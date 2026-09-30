"""Native local transport, agent tools, and provider-aware admission contracts."""
import asyncio
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from demos.launcher.cli import setup_backend
from demos.server.configuration import WebConfiguration
from demos.settings import UserSetup, save_setup
from harness.agents.config import GeneralAgentConfig
from harness.agents.context import FrozenAgentContext
from harness.agents.contracts import AgentInput, AgentRequest
from harness.agents.llamacpp import LlamaCppAgentProvider
from harness.backends import task_backends
from harness.config import LlamaCppConfig
from harness.llm.llamacpp import LlamaCppBackend, LlamaCppDirect, check_server


def local_data():
    return {"workspace": "./tasks", "general_provider": "llamacpp",
            **{role: {"provider": "llamacpp"} for role in ("routing", "responses", "multimodal")}}


def final(text="Done", artifacts=None):
    return json.dumps({"outcome": "completed", "full_result": text,
                       "artifacts": artifacts or [], "assumptions": [], "unresolved": []})


class LocalConfigurationTests(unittest.TestCase):
    def test_all_local_needs_no_codex_and_uses_local_planner(self):
        setup = UserSetup(local_data(), '/tmp/local.json')
        with patch('shutil.which', return_value=None):
            self.assertTrue(setup.check()['ok'])
        self.assertFalse(setup.uses_codex)
        direct, planner = task_backends(setup)
        self.assertIsInstance(direct.direct, LlamaCppDirect)
        self.assertIsInstance(direct.polish.backend, LlamaCppBackend)
        self.assertIsInstance(planner._backend, LlamaCppBackend)
        data = local_data()
        data['responses']['provider'] = 'codex'
        self.assertTrue(UserSetup(data, '/tmp/local.json').uses_codex)

    def test_startup_and_web_gate_use_server_probe_without_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'harness.json'
            save_setup(local_data(), path)
            config = SimpleNamespace(settings=path, runtime=Path(tmp))
            with patch('harness.llm.llamacpp.check_server', return_value={'state':'ready', 'message':'ready'}), patch('demos.launcher.cli.subprocess.run') as run, patch('demos.server.codex_login.CodexLoginCheck.check') as login:
                setup_backend(config, False)
                self.assertTrue(WebConfiguration(path).status())
                run.assert_not_called()
                login.assert_not_called()
            with patch('harness.llm.llamacpp.check_server', return_value={'state':'error', 'message':'offline'}):
                self.assertFalse(WebConfiguration(path).status())
                with self.assertRaisesRegex(RuntimeError, 'offline'):
                    setup_backend(config, False)

    def test_health_rejects_missing_model_and_bad_http(self):
        for payload, ready in [({'data':[{'id':'local-model'}]}, True), ({'data':[]}, False), ({}, False)]:
            response = httpx.Response(200, json=payload, request=httpx.Request('GET','http://localhost/v1/models'))
            with patch('httpx.Client.get', return_value=response):
                status = check_server(LlamaCppConfig(model='local-model'))
                self.assertEqual(status['state']=='ready', ready)
        with patch('httpx.Client.get', side_effect=httpx.ConnectError('unreachable')):
            self.assertEqual(check_server(LlamaCppConfig())['state'], 'error')

    def test_reject_invalid_config(self):
        for value in ('file:///tmp/model', 'http://user:secret@localhost/v1', 'http://localhost/v1?key=x'):
            with self.assertRaises(ValueError):
                LlamaCppConfig(base_url=value)
        for value in (0, True, -1):
            with self.assertRaises(ValueError):
                LlamaCppConfig(max_steps=value)


class LocalTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_chat_contract_and_no_auth(self):
        def handler(request):
            self.assertEqual(str(request.url), 'http://127.0.0.1:8082/v1/chat/completions')
            self.assertNotIn('authorization', request.headers)
            body = json.loads(request.content)
            self.assertEqual(body['model'], 'local-model')
            self.assertEqual(body['response_format'], {'type':'json_object'})
            self.assertFalse(body['stream'])
            return httpx.Response(200, json={'choices':[{'finish_reason':'stop','message':{'content':'{"speech":"hello"}'}}]})
        backend = LlamaCppBackend(LlamaCppConfig(model='local-model'), transport=httpx.MockTransport(handler))
        self.assertEqual(json.loads(await backend.complete('Return JSON', {'text':'hello'}))['speech'], 'hello')

    async def test_truncated_invalid_and_http_errors(self):
        for response in [httpx.Response(503), httpx.Response(200,json={'choices':[{'finish_reason':'length','message':{'content':'{}'}}]}), httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':'not JSON'}}]})]:
            backend = LlamaCppBackend(LlamaCppConfig(), transport=httpx.MockTransport(lambda request:response))
            with self.assertRaises(RuntimeError):
                await backend.complete('test', {})


class ScriptedBackend:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.messages = []

    async def chat(self, messages, schema):
        self.messages.append(list(messages))
        return next(self.replies)


class LocalAgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = GeneralAgentConfig(workspace=self.temp.name)
        self.events = []

    async def emit(self, event):
        self.events.append(event)

    async def test_write_artifact_and_continuation(self):
        backend = ScriptedBackend([
            json.dumps({'tool':'write_file','arguments':{'path':'result.txt','text':'hello'}}),
            final(artifacts=['result.txt']), final('Continued')])
        agent = LlamaCppAgentProvider(self.config, LlamaCppConfig(), backend=backend)
        result = await agent.run(AgentRequest('s','w','Write a file'), self.emit)
        self.assertEqual(Path(result.artifacts[0].path).read_text(), 'hello')
        continued = await agent.run(AgentRequest('s','w2','Continue',parent_work_id='w'), self.emit)
        self.assertEqual(result.thread_id, continued.thread_id)
        self.assertTrue(any(m['content']=='Write a file' for m in backend.messages[-1]))
        self.assertTrue(any(e.kind=='tool' for e in self.events))
        self.assertEqual(self.events[-1].kind, 'completed')
        with self.assertRaises(ValueError):
            await agent.run(AgentRequest('other-session','w3','Continue',parent_work_id='w'), self.emit)
        await agent.aclose()

    async def test_tool_paths_inputs_and_readonly_are_enforced(self):
        root = Path(self.temp.name)/'task'
        context = FrozenAgentContext(AgentRequest('s','w','test', inputs=(AgentInput('a.txt', b'original'),)),root)
        agent = LlamaCppAgentProvider(self.config, LlamaCppConfig())
        for path in ('../escape.txt', '/tmp/escape.txt', 'inputs/a.txt'):
            with self.assertRaises(ValueError):
                agent._tool(context,'write_file',{'path':path,'text':'bad'})
        (root/'outside').symlink_to(Path(self.temp.name))
        with self.assertRaises(ValueError):
            agent._tool(context,'read_file',{'path':'outside/secret'})
        agent.config = replace(self.config,sandbox='read-only')
        with self.assertRaises(ValueError):
            agent._tool(context,'write_file',{'path':'answer.txt','text':'bad'})
        self.assertEqual((root/'inputs/a.txt').read_bytes(), b'original')

    async def test_step_limit_and_duplicate_request(self):
        backend = ScriptedBackend([json.dumps({'tool':'list_files','arguments':{'path':'.'}})])
        agent = LlamaCppAgentProvider(self.config, LlamaCppConfig(max_steps=1), backend=backend)
        result = await agent.run(AgentRequest('s','w','test'),self.emit)
        self.assertEqual(result.outcome,'partial')
        with self.assertRaises(ValueError):
            await agent.run(AgentRequest('s','w','duplicate'),self.emit)
        await agent.aclose()

    async def test_visual_requests_report_limitation_without_inference(self):
        backend = ScriptedBackend([])
        agent = LlamaCppAgentProvider(self.config, LlamaCppConfig(), backend=backend)
        result = await agent.run(AgentRequest('s','w','Describe image',
            context={'visual_evidence':{'status':'missing'}}), self.emit)
        self.assertEqual(result.outcome, 'partial')
        self.assertTrue(result.unresolved)
        self.assertFalse(backend.messages)

    async def test_failed_parent_cannot_continue(self):
        backend = ScriptedBackend(['invalid JSON'])
        agent = LlamaCppAgentProvider(self.config, LlamaCppConfig(), backend=backend)
        with self.assertRaises(ValueError):
            await agent.run(AgentRequest('s','w','test'),self.emit)
        with self.assertRaisesRegex(RuntimeError, 'parent failed'):
            await agent.run(AgentRequest('s','w2','continue',parent_work_id='w'),self.emit)
        await agent.aclose()

    async def test_cancel_closes_inflight_request(self):
        entered = asyncio.Event()
        class Blocking:
            async def chat(self,*args):
                entered.set()
                await asyncio.Future()
        agent = LlamaCppAgentProvider(self.config, LlamaCppConfig(), backend=Blocking())
        pending = asyncio.create_task(agent.run(AgentRequest('s','w','test'),self.emit))
        await entered.wait()
        self.assertEqual((await agent.read_progress('s','w'))['status'],'running')
        await agent.aclose()
        self.assertTrue(pending.cancelled())


if __name__ == '__main__':
    unittest.main()
