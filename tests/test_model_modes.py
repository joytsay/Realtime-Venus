import asyncio
import json
import tempfile
import unittest
import wave
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import av
import numpy as np
from fastapi.testclient import TestClient
from demos.launcher.config import checkpoint_directory
from demos.launcher.variants import select_checkpoint
from demos.launcher.cli import _run
from demos.model.adapter import _load_real_blocking
from demos.server.app import create_app
from demos.server.media import audio_buckets
from demos.server.session import WebAgentSession


def checkpoint(path, kind):
    path.mkdir(parents=True)
    (path/'config.json').write_text(json.dumps({'model_type':kind}))
    for file in ['tokenizer.json','model.safetensors','assets/HT_ref_audio.wav','assets/token2wav/flow.yaml','assets/token2wav/flow.pt','assets/token2wav/hift.pt','assets/token2wav/campplus.onnx','assets/token2wav/speech_tokenizer_v2_25hz.onnx']:
        p=path/file;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'fixture')
    return path


class ModelModeTests(unittest.TestCase):
    def test_repository_root_and_sibling_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            omni=checkpoint(root/'weights/Realtime-Venus-Omni','realtime_venus_omni')
            audio=checkpoint(root/'weights/Realtime-Venus-Audio','minicpmo')
            self.assertEqual(select_checkpoint(root,'audio',configured=str(omni)),audio.resolve())
            self.assertEqual(select_checkpoint(root,'omni',explicit=str(omni.parent)),omni.resolve())
            with self.assertRaisesRegex(ValueError,'contains omni weights'):
                select_checkpoint(root,'audio',explicit=str(omni))

    def test_missing_audio_never_falls_back_to_omni(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            omni=checkpoint(root/'model_weight','realtime_venus_omni')
            with self.assertRaisesRegex(ValueError,'Realtime-Venus-Audio'):
                select_checkpoint(root,'audio',configured=str(omni))

    def test_launcher_forwards_variant_to_both_services(self):
        for mode,kind in [('audio','minicpmo'),('omni','realtime_venus_omni')]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);path=checkpoint(root/'weights',kind)
                with patch.dict('os.environ',{'VENUS_ROOT':tmp}), patch('demos.launcher.cli.subprocess.run'), patch('demos.launcher.cli.setup_backend'), patch('demos.launcher.cli.ensure_ports_available'), patch('demos.launcher.cli.StackSupervisor') as supervisor:
                    _run([mode,'--model-path',str(path)])
                    cfg,model,web=supervisor.call_args.args
                    self.assertEqual(cfg.model_type,mode)
                    self.assertEqual(model[model.index('--model-type')+1],mode)
                    self.assertEqual(web[web.index('--model-type')+1],mode)

    def test_audio_loader_disables_vision_without_video_memory(self):
        for mode in ['audio','omni']:
            base=MagicMock();base.eval.return_value=base;base.cuda.return_value=base
            duplex=SimpleNamespace(tokenizer=object());base.as_duplex.return_value=duplex
            auto=MagicMock();auto.from_pretrained.return_value=base
            with patch.dict('sys.modules',{'torch':SimpleNamespace(bfloat16='bf16'),'transformers':SimpleNamespace(AutoModel=auto,AutoTokenizer=MagicMock()),'librosa':SimpleNamespace()}):
                model,_,_=_load_real_blocking('/fake',40,None,mode)
            self.assertIs(model,duplex)
            if mode=='audio':
                self.assertFalse(auto.from_pretrained.call_args.kwargs['init_vision'])
                self.assertTrue(auto.from_pretrained.call_args.kwargs['init_tts'])
                base.use_memory.assert_not_called()
                base.as_duplex.assert_called_once_with(generate_audio=True)
            else:
                base.use_memory.assert_called_once_with(memory_minutes=40)

    def test_web_status_and_invalid_modes_are_enforced_server_side(self):
        with tempfile.TemporaryDirectory() as tmp:
            for mode,wrong in [('audio','omni'),('omni','audio')]:
                app=create_app(settings_path=Path(tmp)/'settings.json',tokenizer_path='/unused',model_type=mode)
                with TestClient(app) as client:
                    result=client.get('/api/status').json()
                    self.assertEqual(result['model_type'],mode)
                    self.assertEqual(result['model'],'Realtime-Venus-'+mode.capitalize())
                    self.assertEqual(result['input_modes'],['audio','audio_file','text'] if mode=='audio' else ['camera','video','audio_file','text'])
                    with client.websocket_connect('/ws?mode='+wrong) as socket:
                        self.assertEqual(socket.receive_json()['code'],'input_mode')
                    with client.websocket_connect('/ws?mode='+mode+'&source='+('video' if mode=='audio' else 'invalid')) as socket:
                        self.assertEqual(socket.receive_json()['code'],'input_mode')


class AudioDecodeTests(unittest.TestCase):
    def test_stereo_resampling_padding_and_no_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'speech.wav'
            samples=np.full((22050*3//2,2),1200,dtype='<i2')
            with wave.open(str(p),'wb') as w:
                w.setnchannels(2);w.setsampwidth(2);w.setframerate(22050);w.writeframes(samples.tobytes())
            buckets=list(audio_buckets(p))
            self.assertEqual(len(buckets),2)
            self.assertTrue(all(len(b.pcm)==32000 and not b.jpeg for b in buckets))
            self.assertTrue(any(buckets[0].pcm))
            self.assertEqual(buckets[-1].pcm[-8000:],b'\0'*8000)

    def test_corrupt_media_and_video_only_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'bad.wav';p.write_bytes(b'not audio')
            with self.assertRaises(Exception):list(audio_buckets(p))
            p=Path(tmp)/'video.mp4'
            with av.open(str(p),'w') as out:
                stream=out.add_stream('mpeg4',rate=1);stream.width=16;stream.height=16;stream.pix_fmt='yuv420p'
                frame=av.VideoFrame.from_ndarray(np.zeros((16,16,3),dtype=np.uint8),format='rgb24')
                for packet in stream.encode(frame):out.mux(packet)
                for packet in stream.encode():out.mux(packet)
            with self.assertRaisesRegex(ValueError,'音频轨道'):list(audio_buckets(p))


class UploadModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_audio_upload_feeds_pcm_and_never_video_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'audio.wav'
            with wave.open(str(path),'wb') as w:
                w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(b'\1\0'*8000)
            session=WebAgentSession(MagicMock(),MagicMock(),mode='audio',input_source='audio_file')
            session.host=SimpleNamespace(append_audio=AsyncMock(),append_video_frame=AsyncMock())
            session.send=AsyncMock();session._video_idle=AsyncMock()
            temporary=MagicMock()
            with patch('demos.server.session.VIDEO_TAIL_SECONDS',0),patch('demos.server.session.asyncio.sleep',new=AsyncMock()):
                await session._feed_video(path,temporary)
                await session._video_idle_task
            session.host.append_audio.assert_awaited_once()
            session.host.append_video_frame.assert_not_called()
            self.assertIn({'type':'media_status','state':'complete','seconds':1},[call.args[0] for call in session.send.call_args_list])
            temporary.cleanup.assert_called_once()

    async def test_wrong_upload_endpoint_is_rejected(self):
        audio=WebAgentSession(MagicMock(),MagicMock(),mode='audio',input_source='audio_file')
        video=WebAgentSession(MagicMock(),MagicMock(),mode='omni',input_source='video')
        with self.assertRaises(ValueError):await audio.upload_video(None)
        with self.assertRaises(ValueError):await video.upload_audio(None)

if __name__=='__main__':unittest.main()
