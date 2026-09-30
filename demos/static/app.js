import { t, getLanguage, setLanguage } from "./i18n.js?v=20260930-llamacpp";
import { Presence } from "./presence.js";
import { AudioPlayer, encode } from "./audio.js";
import { MediaCapture } from "./capture.js";
import { MicrophoneWaveform } from "./waveform.js?v=20260921-model-modes";
import { TaskTray } from "./tasks.js?v=20260921-model-modes";

const $ = (id) => document.getElementById(id);
const state = {
  mode: "audio",
  modelType: null,
  phase: "idle",
  connected: false,
  connecting: false,
  stopping: false,
  epoch: 0,
  socket: null,
  sessionId: null,
  token: null,
  startedAt: 0,
  configuration: null,
  configured: null,
  checkingSettings: false,
  file: null,
  fileUrl: null,
  upload: null,
  uploadState: "",
  muted: false,
  outputMuted: false,
  transcript: new Map(),
  lastBackpressure: 0,
  online: true,
};
const audioTextKeys = new Set(["replaceVideo", "dropHint", "dropTypes", "videoMode", "intro", "multimodalHint", "videoReady", "startVideo", "chooseVideo", "uploading", "watching", "videoFinished", "videoFailed", "fileTooLarge", "invalidVideo", "videoHint"]);
function mediaText(key) {
  return t(state.modelType === "audio" && audioTextKeys.has(key) ? `audio_${key}` : key);
}
function previewMedia() { return $(state.modelType === "audio" ? "preview-audio" : "preview-video"); }

const devices = { microphone: "", camera: "" };
try {
  Object.assign(
    devices,
    JSON.parse(localStorage.getItem("venus-devices") || "{}"),
  );
} catch {}
const presence = new Presence($("presence-canvas"));
const microphoneWaveform = new MicrophoneWaveform($("mic-visualizer"));
const tasks = new TaskTray({ send, download: downloadArtifact });
const player = new AudioPlayer({
  ack: send,
  onText: appendText,
  onActivity: (active) => {
    if (state.connected) setPhase(active ? "speaking" : "listening");
  },
  onError: () => {
    notify(t("browserAudio"), true);
    void stopSession().catch(() => {});
  },
});
const capture = new MediaCapture($("preview-video"), {
  onChunk: sendInput,
  onLevel: (level) => (presence.inputLevel = level),
  onEnded: () => {
    notify(t("deviceEnded"), true);
    void stopSession().catch(() => {});
  },
  onBackpressure: backpressure,
});
let toastTimer;
function notify(message, error = false) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").classList.toggle("error", error);
  $("toast").hidden = false;
  toastTimer = setTimeout(
    () => ($("toast").hidden = true),
    error ? 7000 : 4200,
  );
}
function humanError(error) {
  const key = {
    NotAllowedError: "permissionDenied",
    PermissionDeniedError: "permissionDenied",
    NotFoundError: "deviceMissing",
    NotReadableError: "deviceBusy",
    InsecureContextError: "insecure",
  }[error?.name];
  return key ? t(key) : error?.message || t("connectionFailed");
}
function setPhase(phase) {
  state.phase = phase;
  document.body.dataset.phase = phase;
  presence.phase = phase;
  render();
}
function send(message) {
  if (state.socket?.readyState === WebSocket.OPEN) {
    state.socket.send(JSON.stringify(message));
    return true;
  }
  return false;
}
function backpressure() {
  if (Date.now() - state.lastBackpressure > 20000) {
    state.lastBackpressure = Date.now();
    notify(t("slowConnection"));
  }
}
function sendInput(pcm, jpeg) {
  if (!state.connected) return;
  if (state.socket.bufferedAmount > 256 * 1024) {
    backpressure();
    return;
  }
  if (jpeg) send({ type: "video_frame", format: "jpeg", data: encode(jpeg) });
  send({
    type: "audio",
    format: "pcm16",
    sample_rate: 16000,
    data: encode(pcm),
  });
}
function render() {
  const busy = state.connected || state.connecting || state.stopping;
  document.body.dataset.mode = state.mode;
  document.body.dataset.modelType = state.modelType || "loading";
  $("frontend-model-name").textContent = state.modelType ? `REALTIME-VENUS-${state.modelType.toUpperCase()}` : "REALTIME-VENUS";
  for (const node of document.querySelectorAll("[data-media-key]")) node.textContent = mediaText(node.dataset.mediaKey);
  $("video-file").accept = state.modelType === "audio" ? "audio/*,.wav,.mp3,.m4a,.flac,.ogg,.opus,.aac,.aiff,.aif,.wma" : "video/*,.mp4,.mov,.webm,.mkv,.avi";
  $("camera-device").closest("label").hidden = state.modelType === "audio";
  $("preview-audio").hidden = state.modelType !== "audio";
  $("preview-video").hidden = state.modelType === "audio";
  $("media-preview").classList.toggle("audio-file", state.modelType === "audio");
  const micActive = Boolean(capture.stream) && state.mode !== "file" && !state.stopping;
  microphoneWaveform.setState(micActive, state.muted);
  document.body.classList.toggle("mic-open", micActive);
  for (const button of document.querySelectorAll(".mode-button")) {
    button.classList.toggle("selected", button.dataset.mode === state.mode);
    button.setAttribute(
      "aria-pressed",
      String(button.dataset.mode === state.mode),
    );
    button.hidden = state.modelType === "audio" && button.dataset.mode === "camera";
    button.disabled = busy || state.checkingSettings || !state.modelType;
  }
  $("start-button").hidden = state.connected;
  $("start-button").disabled = state.connecting || state.stopping || state.checkingSettings || !state.modelType;
  $("setup-banner").hidden = busy || state.configured !== false;
  $("start-label").textContent = mediaText(
    state.checkingSettings
      ? "checkingSettings"
      : state.configured === false && !busy
        ? "completeSettings"
        : state.stopping
      ? "ending"
      : state.connecting
        ? "connecting"
        : state.mode === "file"
          ? state.file
            ? "startVideo"
            : "chooseVideo"
          : "start",
  );
  $("stop-button").hidden = !(state.connected || state.connecting);
  $("stop-button").disabled = state.stopping;
  $("mute-mic").hidden = !state.connected || state.mode === "file";
  $("mute-mic").setAttribute("aria-pressed", String(state.muted));
  $("mute-mic")
    .querySelector("use")
    .setAttribute("href", state.muted ? "#i-mic-off" : "#i-mic");
  $("mute-output").hidden = !state.connected;
  $("mute-output").setAttribute("aria-pressed", String(state.outputMuted));
  $("upload-zone").hidden = state.mode !== "file" || Boolean(state.file);
  const preview =
    (state.mode === "camera" && Boolean(capture.stream)) ||
    (state.mode === "file" && Boolean(state.file));
  $("media-preview").hidden = !preview;
  $("experience").classList.toggle("previewing", preview);
  $("media-preview").classList.toggle("camera", state.mode === "camera");
  $("replace-video").hidden = state.mode !== "file" || !state.file || busy;
  $("preview-label").textContent = mediaText(
    state.mode === "camera"
      ? "yourView"
      : state.connected
        ? "watching"
        : "videoReady",
  );
  previewMedia().controls = state.mode === "file" && !busy;
  const presenceKey = state.connecting
    ? "connecting"
    : state.phase === "speaking"
      ? "speaking"
      : state.connected
        ? state.mode === "file"
          ? "watching"
          : "listening"
        : state.mode === "file"
          ? "videoReady"
          : "presenceIdle";
  $("presence-label").textContent = mediaText(presenceKey);
  let hint =
    state.mode === "camera"
      ? "cameraHint"
      : state.mode === "file"
        ? "videoHint"
        : "startHint";
  if (state.connected)
    hint =
      state.mode === "file"
        ? state.uploadState === "complete"
          ? "videoFinished"
          : "watching"
        : state.muted
          ? "mutedHint"
          : "liveHint";
  if (state.uploadState === "uploading") hint = "uploading";
  if (state.connecting) hint = "connecting";
  if (state.stopping) hint = "ending";
  if (!busy && state.configured === false) hint = "notConfigured";
  $("control-hint").textContent = mediaText(hint);
  const status = $("connection-state");
  status.classList.toggle("pending", state.connecting || state.stopping);
  status.classList.toggle("offline", !state.online);
  status.lastElementChild.textContent = state.stopping
    ? t("ending")
    : state.connecting
      ? t("connecting")
      : state.connected
        ? `${t("live")} · ${elapsed()}`
        : t(!state.online ? "offline" : state.configured === false ? "setupNeeded" : "available");
  $("reset-button").disabled = state.connecting || state.stopping;
  if (!state.transcript.size)
    $("caption-text").textContent = mediaText(
      state.mode === "file" ? "videoHint" : "prompt",
    );
}
function elapsed() {
  const seconds = Math.max(
    0,
    Math.floor((Date.now() - state.startedAt) / 1000),
  );
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
}
function setMode(mode) {
  if (state.connected || state.connecting || state.stopping) return;
  state.mode = mode;
  state.uploadState = "";
  previewMedia().pause();
  previewMedia().srcObject = null;
  if (mode === "file" && state.fileUrl) previewMedia().src = state.fileUrl;
  else {
    previewMedia().removeAttribute("src");
    previewMedia().load();
  }
  $("video-name").textContent =
    mode === "file" && state.file ? state.file.name : "";
  $("video-time").textContent = "";
  render();
}
async function refreshStatus(fresh = false) {
  try {
    const response = await fetch(`/api/status${fresh ? "?fresh=true" : ""}`, { cache: "no-store" });
    if (!response.ok) throw Error();
    const status = await response.json();
    state.online = true;
    if (!["audio", "omni"].includes(status.model_type)) throw Error("Unknown frontend model type");
    if (state.modelType !== status.model_type && !state.connected && !state.connecting) {
      if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
      state.file = state.fileUrl = null;
      for (const media of [$("preview-video"), $("preview-audio")]) {
        media.pause(); media.removeAttribute("src"); media.load();
      }
      state.modelType = status.model_type;
      state.mode = "audio";
    }
    state.configured = status.configured;
    return status;
  } catch {
    state.online = false;
    return null;
  } finally {
    render();
  }
}
async function requireSettings() {
  if (state.checkingSettings) return false;
  state.checkingSettings = true;
  render();
  try {
    const status = await refreshStatus(true);
    if (!status) {
      notify(t("network"), true);
      return false;
    }
    if (!status.configured) {
      await openSettings(true);
      return false;
    }
    return true;
  } finally {
    state.checkingSettings = false;
    render();
  }
}
async function chooseMode(mode) {
  if (state.connected || state.connecting || state.stopping) return;
  if (await requireSettings()) {
    if (mode === "file" || mode === "audio" || (mode === "camera" && state.modelType === "omni")) setMode(mode);
  }
}
async function chooseVideo() {
  if (state.connected || state.connecting || state.stopping) return;
  if (!(await requireSettings())) return;
  if (state.connected || state.connecting || state.stopping) return;
  $("video-file").click();
}
async function startSession() {
  if (state.connected || state.connecting || state.stopping || state.checkingSettings) return;
  if (state.mode === "file" && !state.file) {
    await chooseVideo();
    return;
  }
  if (state.configured === false) {
    await requireSettings();
    return;
  }
  // Unlock playback during the user's click, before any asynchronous network request.
  const primed = player.prime();
  state.checkingSettings = true;
  const epoch = ++state.epoch;
  render();
  try {
    await primed;
    const status = await refreshStatus(true);
    if (epoch !== state.epoch) return;
    if (!status) throw Error(t("network"));
    if (!status.configured) {
      state.checkingSettings = false;
      render();
      await openSettings(true);
      return;
    }
    if (status.busy) throw Error(t("busy"));
    state.checkingSettings = false;
    state.connecting = true;
    setPhase("connecting");
    tasks.clear();
    clearTranscript();
    state.muted = false;
    state.uploadState = "";
    if (state.mode !== "file") {
      const started = await capture.start(state.mode, devices);
      if (!started || epoch !== state.epoch) return;
      render();
      void enumerateDevices();
    }
    if (epoch !== state.epoch) return;
    const ready = await connect(epoch);
    if (epoch !== state.epoch) return;
    state.sessionId = ready.session_id;
    state.token = ready.upload_token;
    state.connected = true;
    state.connecting = false;
    state.startedAt = Date.now();
    capture.active = true;
    setPhase("listening");
    if (state.mode === "file") await uploadVideo(epoch);
  } catch (error) {
    if (epoch !== state.epoch) return;
    await stopSession().catch(() => {});
    if (error.code === "configuration") {
      state.configured = false;
      await openSettings(true);
    } else notify(humanError(error), true);
  } finally {
    state.checkingSettings = false;
    render();
  }
}
function connect(epoch) {
  return new Promise((resolve, reject) => {
    const mode = state.modelType,
      source = state.mode === "file" ? (state.modelType === "audio" ? "audio_file" : "video") : "live";
    const socket = new WebSocket(
      `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws?mode=${mode}&source=${source}`,
    );
    state.socket = socket;
    let settled = false;
    let serverRejected = false;
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        reject(Error(t("connectionFailed")));
        socket.close();
      }
    }, 180000);
    socket.onmessage = (event) => {
      if (epoch !== state.epoch) return;
      let message;
      try {
        message = JSON.parse(event.data);
      } catch {
        return;
      }
      if (message.type === "ready") {
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          resolve(message);
        }
        return;
      }
      if (message.type === "fatal_error") {
        serverRejected = true;
        const error = Error(
          message.code === "busy"
            ? t("busy")
            : message.code === "configuration"
              ? t("notConfigured")
              : message.error || t("connectionFailed"),
        );
        error.code = message.code;
        if (!settled) {
          settled = true;
          clearTimeout(timer);
          reject(error);
        } else if (error.code === "configuration") {
          state.configured = false;
          void stopSession().then(() => openSettings(true)).catch(() => {});
        } else {
          notify(humanError(error), true);
          void stopSession().catch(() => {});
        }
        return;
      }
      if (message.type === "playback") player.enqueue(message);
      else if (message.type === "playback_end") player.finish(message.utterance_id);
      else if (message.type === "text")
        appendText(message.text, message.utterance_id);
      else if (message.type === "works") tasks.update(message.works);
      else if (message.type === "media_status" || message.type === "video_status") videoStatus(message);
      else if (message.type === "command_error") notify(message.error, true);
    };
    socket.onerror = () => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(Error(t("network")));
      }
    };
    socket.onclose = () => {
      clearTimeout(timer);
      if (!settled) {
        settled = true;
        reject(Error(t("network")));
      }
      if (epoch === state.epoch && !state.stopping && !serverRejected) {
        notify(t("network"), true);
        void stopSession(false).catch(() => {});
      }
    };
  });
}
async function stopSession(notifyServer = true) {
  if (state.stopping) return state.stopPromise;
  state.stopping = true;
  state.connecting = false;
  state.connected = false;
  state.epoch++;
  setPhase("ending");
  state.upload?.abort();
  state.upload = null;
  player.clear();
  capture.active = false;
  const socket = state.socket;
  state.stopPromise = (async () => {
    const stopped = capture.stop();
    try {
      if (socket && socket.readyState !== WebSocket.CLOSED) {
        await new Promise((resolve, reject) => {
          const timer = setTimeout(() => {
            socket.close();
            reject(Error(t("sessionClosing")));
          }, 45000);
          socket.addEventListener(
            "close",
            () => {
              clearTimeout(timer);
              resolve();
            },
            { once: true },
          );
          if (notifyServer && socket.readyState === WebSocket.OPEN)
            socket.send(JSON.stringify({ type: "stop_session" }));
          else socket.close();
        });
      }
      await stopped;
    } finally {
      state.socket = null;
      state.sessionId = null;
      state.token = null;
      state.stopping = false;
      state.uploadState = "";
      $("upload-progress").hidden = true;
      previewMedia().pause();
      tasks.end();
      setPhase("idle");
      void refreshStatus();
    }
  })();
  try {
    await state.stopPromise;
  } catch (error) {
    notify(humanError(error), true);
    throw error;
  } finally {
    state.stopPromise = null;
  }
}
async function selectFile(file) {
  if (!file || state.connected || state.connecting || state.stopping) return;
  if (file.size > 200 * 1024 * 1024) {
    notify(mediaText("fileTooLarge"), true);
    return;
  }
  const validFile = state.modelType === "audio"
    ? (file.type.startsWith("audio/") || /\.(wav|mp3|m4a|flac|ogg|opus|aac|aiff|aif|wma)$/i.test(file.name))
    : (file.type.startsWith("video/") || /\.(mp4|mov|webm|mkv|avi)$/i.test(file.name));
  if (!validFile) {
    notify(mediaText("invalidVideo"), true);
    return;
  }
  if (!(await requireSettings())) return;
  if (state.connected || state.connecting || state.stopping) return;
  if (state.fileUrl) URL.revokeObjectURL(state.fileUrl);
  state.file = file;
  state.fileUrl = URL.createObjectURL(file);
  setMode("file");
}
async function uploadVideo(epoch) {
  state.uploadState = "uploading";
  $("upload-progress").hidden = false;
  const bar = $("upload-progress").querySelector("span");
  bar.style.width = "0%";
  render();
  await new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    state.upload = xhr;
    xhr.open(
      "POST",
      `/api/sessions/${encodeURIComponent(state.sessionId)}/${state.modelType === "audio" ? "audio" : "video"}`,
    );
    xhr.setRequestHeader("X-Venus-Session-Token", state.token);
    xhr.setRequestHeader(
      "Content-Type",
      state.file.type || "application/octet-stream",
    );
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable)
        bar.style.width = `${Math.round((event.loaded / event.total) * 100)}%`;
    };
    xhr.onload = () => {
      state.upload = null;
      if (xhr.status >= 200 && xhr.status < 300) resolve();
      else {
        let detail;
        try {
          detail = JSON.parse(xhr.responseText).detail;
        } catch {}
        reject(Error(detail || mediaText("videoFailed")));
      }
    };
    xhr.onerror = () => reject(Error(t("network")));
    xhr.onabort = () =>
      reject(Object.assign(Error("Upload cancelled"), { name: "AbortError" }));
    xhr.send(state.file);
  });
  if (epoch !== state.epoch) return;
  state.uploadState = "feeding";
  $("upload-progress").hidden = true;
  const video = previewMedia();
  video.currentTime = 0;
  await video.play().catch(() => {});
  render();
}
function videoStatus(message) {
  state.uploadState = message.state;
  if (message.seconds !== undefined)
    $("video-time").textContent = `${message.seconds}s`;
  if (message.state === "feeding") {
    const video = previewMedia();
    if (
      Number.isFinite(video.duration) &&
      Math.abs(video.currentTime - message.seconds) > 2
    )
      video.currentTime = Math.min(video.duration, message.seconds);
  }
  if (message.state === "complete") previewMedia().pause();
  if (message.state === "error") {
    notify(mediaText("videoFailed"), true);
    previewMedia().pause();
  }
  render();
}
function clearTranscript() {
  state.transcript.clear();
  $("transcript").replaceChildren();
  const empty = document.createElement("p");
  empty.className = "empty-copy";
  empty.dataset.i18n = "historyEmpty";
  empty.textContent = t("historyEmpty");
  $("transcript").append(empty);
  $("caption").classList.remove("has-speech");
  $("caption-speaker").hidden = true;
  $("caption-text").textContent = mediaText("prompt");
}
function appendText(text, id) {
  if (!text) return;
  const key = id || `text-${state.transcript.size}`;
  let entry = state.transcript.get(key);
  if (!entry) {
    $("transcript").querySelector(".empty-copy")?.remove();
    const node = document.createElement("article");
    node.className = "transcript-entry";
    const heading = document.createElement("header");
    heading.textContent = "VENUS";
    const time = document.createElement("time");
    time.textContent = new Date().toLocaleTimeString(getLanguage(), {
      hour: "2-digit",
      minute: "2-digit",
    });
    heading.append(time);
    const paragraph = document.createElement("p");
    node.append(heading, paragraph);
    $("transcript").append(node);
    entry = { text: "", node, paragraph, time: Date.now() };
    state.transcript.set(key, entry);
  }
  entry.text += text;
  entry.paragraph.textContent = entry.text;
  $("caption").classList.add("has-speech");
  $("caption-speaker").hidden = false;
  $("caption-text").textContent = entry.text;
  $("caption-text").scrollTop = $("caption-text").scrollHeight;
  $("transcript").scrollTop = $("transcript").scrollHeight;
}
async function downloadArtifact(workId, artifact) {
  if (!state.sessionId || !state.token) {
    notify(t("noResult"));
    return;
  }
  try {
    const response = await fetch(
      `/api/sessions/${encodeURIComponent(state.sessionId)}/works/${encodeURIComponent(workId)}/artifacts/${artifact.index}`,
      { headers: { "X-Venus-Session-Token": state.token } },
    );
    if (!response.ok) throw Error(t("noResult"));
    saveBlob(await response.blob(), artifact.name);
  } catch (error) {
    notify(humanError(error), true);
  }
}
function saveBlob(blob, name) {
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = name;
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
}
async function enumerateDevices() {
  if (!navigator.mediaDevices?.enumerateDevices) return;
  try {
    const available = await navigator.mediaDevices.enumerateDevices();
    for (const [key, kind] of [
      ["microphone", "audioinput"],
      ["camera", "videoinput"],
    ]) {
      const select = $(key + "-device");
      select.replaceChildren(new Option(t("defaultDevice"), ""));
      for (const [i, device] of available
        .filter((value) => value.kind === kind)
        .entries())
        select.add(
          new Option(device.label || `${t(key)} ${i + 1}`, device.deviceId),
        );
      select.value = devices[key];
    }
  } catch {}
}
function updateRoutingOptions() {
  const options = $("auto-routing-options");
  const automatic = $("routing-mode").value === "auto";
  options.hidden = !automatic;
  options.disabled =
    !automatic || state.connected || state.connecting || state.stopping;
}
function updateProviders() {
  const busy = state.connected || state.connecting || state.stopping;
  for (const role of ["routing", "response", "multimodal"]) {
    const gemini = $(`${role}-provider`).value === "gemini";
    const local = $(`${role}-provider`).value === "llamacpp";
    $(`${role}-effort`).closest("label").hidden = gemini || local;
    $(`${role}-effort`).disabled = busy || gemini || local;
    $(`${role}-model`).placeholder = gemini ? $("gemini-model").value : "";
  }
  const localGeneral = $("general-provider").value === "llamacpp";
  for (const id of ["backend-model", "backend-effort"]) {
    $(id).closest("label").hidden = localGeneral;
    $(id).disabled = busy || localGeneral;
  }
  const usesCodex = !localGeneral || ["response", "multimodal"].some(role => $(`${role}-provider`).value === "codex") || ($("routing-mode").value === "auto" && $("routing-provider").value === "codex");
  $("backend-binary").closest("label").hidden = !usesCodex;
  $("backend-binary").disabled = busy || !usesCodex;
  const usesLocal = localGeneral || ["routing", "response", "multimodal"].some(role => $(`${role}-provider`).value === "llamacpp");
  for (const id of ["llama-url", "llama-model"]) {
    $(id).closest("label").hidden = !usesLocal;
    $(id).disabled = busy || !usesLocal;
  }
  $("multimodal-codex-audio-hint").hidden = $("multimodal-provider").value !== "codex";
}
$("routing-mode").onchange = () => { updateRoutingOptions(); updateProviders(); };
for (const role of ["routing", "response", "multimodal"]) {
  $(`${role}-provider`).onchange = () => {
    $(`${role}-model`).value = "";
    updateProviders();
  };
}
$("gemini-model").oninput = updateProviders;
$("general-provider").onchange = updateProviders;
function renderCodexLogin() {
  const status = state.configuration?.codex_login?.state || "error";
  $("codex-login-status").textContent = ["ready", "not_required"].includes(status) || state.configuration?.data?.general_provider === "llamacpp"
    ? state.configuration.codex_login.message : t(`codexLogin_${status}`);
  $("codex-login-help").hidden = ["logged_in", "ready", "not_required"].includes(status) || state.configuration?.data?.general_provider === "llamacpp";
}
$("recheck-codex").onclick = async () => {
  const button = $("recheck-codex");
  button.disabled = true;
  $("codex-login-status").textContent = t("codexChecking");
  try {
    const response = await fetch("/api/config", {cache: "no-store"});
    if (!response.ok) throw Error(t("settingsLoadFailed"));
    const result = await response.json();
    state.configuration.codex_login = result.codex_login;
    state.configured = result.configured;
    renderCodexLogin();
    showSettingsIssues(settingsIssues(result.check?.problems || []));
    render();
  } catch {
    $("codex-login-status").textContent = t("settingsLoadFailed");
  } finally { button.disabled = false; }
};
function settingsIssues(serverProblems = []) {
  const issues = [];
  const add = (id, key) => { if (!issues.some(x => x.id === id)) issues.push({id, message: t(key)}); };
  if (!$("backend-workspace").value.trim()) add("backend-workspace", "workspaceRequired");
  if (!$("backend-binary").disabled && !$("backend-binary").value.trim()) add("backend-binary", "codexRequired");
  const usesGemini = $("response-provider").value === "gemini" || $("multimodal-provider").value === "gemini" || ($("routing-mode").value === "auto" && $("routing-provider").value === "gemini");
  if (usesGemini && !state.configuration?.gemini_key_configured && !$("gemini-key").value.trim()) add("gemini-key", "geminiRequired");
  for (const node of $("settings-form").querySelectorAll("input,select")) {
    if (node.disabled || (!usesGemini && node.id === "gemini-model")) continue;
    if (!node.checkValidity() && !issues.some(x => x.id === node.id))
      issues.push({id: node.id, message: `${node.closest("label")?.querySelector("span")?.textContent || node.id}: ${t("fieldRequired")}`});
  }
  for (const message of serverProblems || []) {
    if (/Configure Task workspace/i.test(message)) add("backend-workspace", "workspaceRequired");
    else if (/Codex executable/i.test(message)) add("backend-binary", "codexRequired");
    else if (/Gemini API key/i.test(message)) add("gemini-key", "geminiRequired");
    else issues.push({message: String(message).split(" / ")[getLanguage() === "zh" && String(message).includes(" / ") ? 1 : 0]});
  }
  return issues;
}
function focusSetting(id) {
  const field = $(id);
  if (!field) return;
  const details = field.closest("details");
  if (details) details.open = true;
  field.focus({preventScroll: true});
  field.scrollIntoView({block: "center", behavior: "smooth"});
}
function clearSettingsIssues() {
  $("settings-guidance").hidden = true;
  $("settings-issues").replaceChildren();
  for (const node of $("settings-form").querySelectorAll("[aria-invalid]")) {
    node.removeAttribute("aria-invalid");
    const errorId = `${node.id}-error`;
    $(errorId)?.remove();
    node.setAttribute("aria-describedby", (node.getAttribute("aria-describedby") || "").split(" ").filter(id => id && id !== errorId).join(" "));
  }
}
function showSettingsIssues(issues, focusFirst = false) {
  clearSettingsIssues();
  $("settings-guidance").hidden = false;
  $("settings-guidance-copy").textContent = t(issues.length ? "setupInstructions" : "setupSaveNext");
  for (const issue of issues) {
    const li = document.createElement("li");
    const item = document.createElement(issue.id ? "button" : "span");
    item.textContent = issue.message;
    if (issue.id) {
      item.type = "button";
      item.onclick = () => focusSetting(issue.id);
      const field = $(issue.id);
      field.setAttribute("aria-invalid", "true");
      const hint = document.createElement("small");
      hint.id = `${issue.id}-error`;
      hint.className = "setting-error";
      hint.textContent = issue.message;
      field.after(hint);
      field.setAttribute("aria-describedby", `${field.getAttribute("aria-describedby") || ""} ${hint.id}`.trim());
    }
    li.append(item);
    $("settings-issues").append(li);
  }
  if (focusFirst && issues[0]?.id) focusSetting(issues[0].id);
}
$("settings-form").addEventListener("input", () => {
  if (!$("settings-guidance").hidden) showSettingsIssues(settingsIssues());
});

async function openSettings(required = false) {
  try {
    const response = await fetch("/api/config", { cache: "no-store" });
    if (!response.ok) throw Error(t("settingsLoadFailed"));
    state.configuration = await response.json();
    renderCodexLogin();
    const data = state.configuration.data;
    $("general-provider").value = data.general_provider;
    $("llama-url").value = data.llamacpp.base_url;
    $("llama-model").value = data.llamacpp.model;
    $("backend-workspace").value = data.workspace || "";
    $("gemini-model").value = data.gemini.model;
    $("gemini-key").value = "";
    $("gemini-key-status").textContent = t(state.configuration.gemini_key_configured ? "geminiKeyConfigured" : "geminiKeyMissing");
    for (const [role, section] of [["routing", "routing"], ["response", "responses"], ["multimodal", "multimodal"]]) {
      $(`${role}-provider`).value = data[section].provider;
    }
    $("multimodal-model").value = data.multimodal.model || "";
    $("multimodal-effort").value = data.multimodal.effort;
    $("reply-language").value = data.language;
    $("proactive-progress").value = String(data.feedback.proactive_progress === true);
    $("length-penalty").value = data.duplex.length_penalty;
    $("backend-model").value = data.codex.model || "";
    $("backend-effort").value = data.codex.effort || "";
    $("routing-mode").value = data.routing.mode;
    $("routing-model").value = data.routing.model || "";
    $("routing-effort").value = data.routing.effort;
    $("routing-timeout").value = data.routing.timeout_s;
    $("response-model").value = data.responses.model || "";
    $("response-effort").value = data.responses.effort;
    $("backend-binary").value = data.codex.command[0];
    void enumerateDevices();
    const busy = state.connected || state.connecting || state.stopping;
    for (const node of $("settings-form").querySelectorAll(
      "input,select,button",
    ))
      node.disabled = busy;
    updateRoutingOptions();
    updateProviders();
    state.configured = state.configuration.configured;
    $("settings-message").textContent = busy ? t("settingsActive") : "";
    clearSettingsIssues();
    if (!$("settings-dialog").open) $("settings-dialog").showModal();
    $("recheck-codex").disabled = busy;
    if (!busy && (required || !state.configured))
      showSettingsIssues(settingsIssues(state.configuration.check?.problems), true);
  } catch (error) {
    notify(humanError(error), true);
  }
}
$("settings-form").onsubmit = async (event) => {
  event.preventDefault();
  if (!state.configuration || state.connected) return;
  const issues = settingsIssues();
  if (issues.length) { showSettingsIssues(issues, true); return; }
  clearSettingsIssues();
  const button = $("save-settings");
  button.disabled = true;
  try {
    const data = structuredClone(state.configuration.data);
    data.general_provider = $("general-provider").value;
    data.llamacpp.base_url = $("llama-url").value.trim();
    data.llamacpp.model = $("llama-model").value.trim();
    data.workspace = $("backend-workspace").value.trim();
    data.language = $("reply-language").value;
    data.feedback.proactive_progress = $("proactive-progress").value === "true";
    data.duplex.length_penalty = Number($("length-penalty").value);
    data.codex.model = $("backend-model").value.trim() || null;
    data.codex.effort = $("backend-effort").value || null;
    data.routing.mode = $("routing-mode").value;
    if (data.routing.mode === "auto") {
      data.routing.provider = $("routing-provider").value;
      data.routing.model = $("routing-model").value.trim() || null;
      data.routing.effort = $("routing-effort").value;
      data.routing.timeout_s = Number($("routing-timeout").value);
    }
    data.responses.provider = $("response-provider").value;
    data.multimodal.provider = $("multimodal-provider").value;
    data.multimodal.model = $("multimodal-model").value.trim() || null;
    data.multimodal.effort = $("multimodal-effort").value;
    data.gemini.model = $("gemini-model").value.trim();
    data.gemini.api_key = $("gemini-key").value.trim();
    data.responses.model = $("response-model").value.trim() || null;
    data.responses.effort = $("response-effort").value;
    data.codex.command = [$("backend-binary").value.trim(), "app-server"];
    const response = await fetch("/api/config", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-Venus-Config-Token": state.configuration.token,
      },
      body: JSON.stringify({ data, revision: state.configuration.revision }),
    });
    const result = await response.json();
    if (!response.ok) {
      showSettingsIssues(settingsIssues(result.problems || [result.detail || t("settingsSaveFailed")] ), true);
      return;
    }
    state.configuration = result;
    state.configured = result.configured;
    $("gemini-key").value = "";
    devices.microphone = $("microphone-device").value;
    devices.camera = $("camera-device").value;
    try {
      localStorage.setItem("venus-devices", JSON.stringify(devices));
    } catch {}
    renderCodexLogin();
    if (!result.configured) {
      $("settings-message").textContent = t("settingsSavedLoginNeeded");
      showSettingsIssues(settingsIssues(result.check?.problems), false);
    } else {
      $("settings-dialog").close();
      notify(t("settingsSaved"));
    }
    void refreshStatus();
  } catch (error) {
    showSettingsIssues([{ message: t("settingsSaveFailed") + " " + humanError(error) }], false);
  } finally {
    button.disabled = false;
  }
};
for (const button of document.querySelectorAll(".mode-button"))
  button.onclick = () => void chooseMode(button.dataset.mode);
$("start-button").onclick = () => void startSession();
$("stop-button").onclick = () => void stopSession().catch(() => {});
$("mute-mic").onclick = () => {
  state.muted = !state.muted;
  capture.setMuted(state.muted);
  render();
};
$("mute-output").onclick = () => {
  state.outputMuted = !state.outputMuted;
  player.setMuted(state.outputMuted);
  render();
};
$("video-file").onchange = (event) => void selectFile(event.target.files[0]);
$("replace-video").onclick = () => void chooseVideo();
$("upload-zone").onclick = (event) => {
  event.preventDefault();
  void chooseVideo();
};
$("upload-zone").onkeydown = (event) => {
  if (event.key === "Enter" || event.key === " ") {
    event.preventDefault();
    void chooseVideo();
  }
};
for (const name of ["dragenter", "dragover"])
  $("upload-zone").addEventListener(name, (event) => {
    event.preventDefault();
    $("upload-zone").classList.add("dragging");
  });
for (const name of ["dragleave", "drop"])
  $("upload-zone").addEventListener(name, (event) => {
    event.preventDefault();
    $("upload-zone").classList.remove("dragging");
    if (name === "drop") void selectFile(event.dataTransfer.files[0]);
  });
$("settings-button").onclick = () => void openSettings();
$("complete-settings").onclick = () => void openSettings(true);
$("history-button").onclick = () => $("history-dialog").showModal();
for (const dialog of document.querySelectorAll("dialog")) {
  dialog.querySelector(".dialog-close").onclick = () => dialog.close();
  dialog.addEventListener("click", (event) => {
    const rect = dialog.getBoundingClientRect();
    if (
      event.target === dialog &&
      (event.clientX < rect.left ||
        event.clientX > rect.right ||
        event.clientY < rect.top ||
        event.clientY > rect.bottom)
    )
      dialog.close();
  });
}
$("reset-button").onclick = async () => {
  const restart = state.connected;
  $("history-dialog").close();
  try {
    await player.prime();
    await stopSession();
    if (status.busy) throw Error(t("busy"));
    state.checkingSettings = false;
    state.connecting = true;
    setPhase("connecting");
    tasks.clear();
    clearTranscript();
    if (restart) await startSession();
    else notify(t("resetDone"));
  } catch {}
};
$("save-transcript").onclick = () => {
  if (!state.transcript.size) {
    notify(t("noTranscript"));
    return;
  }
  saveBlob(
    new Blob(
      [
        [...state.transcript.values()]
          .map(
            (entry) =>
              `Venus · ${new Date(entry.time).toLocaleString()}\n${entry.text}`,
          )
          .join("\n\n"),
      ],
      { type: "text/plain;charset=utf-8" },
    ),
    "Venus-conversation.txt",
  );
};
$("language-button").onclick = () =>
  setLanguage(getLanguage() === "en" ? "zh" : "en");
function renderTheme() {
  const label = t(document.documentElement.dataset.theme === "dark" ? "lightTheme" : "darkTheme");
  $("theme-button").setAttribute("aria-label", label);
  $("theme-button").title = label;
}
window.addEventListener("venus-theme", renderTheme);
window.addEventListener("venus-language", () => {
  renderTheme();
  render();
  void enumerateDevices();
});
window.addEventListener("pagehide", () => {
  state.epoch++;
  player.clear();
  state.upload?.abort();
  capture.stop();
  if (state.socket?.readyState === WebSocket.OPEN)
    send({ type: "stop_session" });
  state.socket?.close();
});
navigator.mediaDevices?.addEventListener?.(
  "devicechange",
  () => void enumerateDevices(),
);
setInterval(() => {
  if (state.connected) render();
}, 1000);
setInterval(() => {
  if (!state.connected && !state.connecting && !state.stopping)
    void refreshStatus();
}, 15000);
function animate(time) {
  presence.outputLevel = player.amplitude();
  if (microphoneWaveform.active && !document.hidden)
    microphoneWaveform.draw(capture.readWaveform(), time);
  requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
renderTheme();
render();
void refreshStatus();
