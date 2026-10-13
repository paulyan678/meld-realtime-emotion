"use strict";

const $ = (id) => document.getElementById(id);
let file = null;
let busy = false;
let recording = null;
let socket = null;
let awaitingStreamResult = false;
let startedAt = 0;
let lastEventTime = 0;
let clockTimer = null;

function automaticTranscript() {
  return $("mode-auto").checked;
}

function log(message, error = false) {
  const row = document.createElement("li");
  const time = document.createElement("time");
  time.textContent = new Date().toLocaleTimeString([], { hour12: false });
  const text = document.createElement("span");
  text.textContent = message;
  row.append(time, text);
  if (error) row.className = "error";
  $("event-log").prepend(row);
  while ($("event-log").children.length > 30) $("event-log").lastChild.remove();
}

function message(text, error = false) {
  $("message").textContent = text;
  $("message").classList.toggle("error", error);
}

function updateTranscriptMode() {
  const automatic = automaticTranscript();
  $("transcript").readOnly = automatic;
  $("transcript").placeholder = automatic ? "Your spoken words will appear here." : "Enter the words spoken in this recording.";
  $("transcript-note").textContent = automatic ? "Generated after recording." : "Your transcript";
  $("transcript-helper").textContent = automatic
    ? "You can correct the transcript afterward."
    : "Enter the spoken words, then click Analyze audio.";
  $("edit-transcript").hidden = !automatic || !$("transcript").value.trim();
}

function controls(isBusy) {
  busy = isBusy;
  $("analyze").disabled = isBusy || !file;
  $("record").disabled = isBusy;
  $("audio-file").disabled = isBusy;
  $("mode-auto").disabled = isBusy;
  $("mode-manual").disabled = isBusy;
  $("edit-transcript").disabled = isBusy;
  $("transcript").disabled = isBusy && !recording;
  $("record").hidden = Boolean(recording);
  $("stop").hidden = !recording;
  $("stop").disabled = !recording;
}

function invalidateResult() {
  $("result").hidden = true;
  $("empty-result").hidden = false;
  for (const id of ["emotion", "confidence", "response"]) $(id).textContent = "";
  for (const id of ["latency", "transcription-latency", "modalities"]) $(id).textContent = "—";
  $("scores").replaceChildren();
  $("result-state").textContent = "Awaiting input";
}

function fail(error) {
  invalidateResult();
  const text = error instanceof Error ? error.message : String(error);
  message(text, true);
  log(text, true);
  $("result-state").textContent = "Needs attention";
  controls(false);
}

for (const mode of ["mode-auto", "mode-manual"]) $(mode).addEventListener("change", () => {
  invalidateResult();
  updateTranscriptMode();
  message(automaticTranscript() ? "The next analysis will transcribe the audio again. Previous text will not be reused." : "Enter or correct the transcript, then analyze your audio.");
});
$("edit-transcript").addEventListener("click", () => {
  invalidateResult();
  $("mode-manual").checked = true;
  updateTranscriptMode();
  $("transcript").focus();
  message("Correct the words, then click Analyze audio to reuse this recording.");
});
$("transcript").addEventListener("input", () => {
  if (!busy) invalidateResult();
});

async function health() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error("Server unavailable");
    const data = await response.json();
    $("connection").className = "status-pill online";
    $("connection-label").textContent = data.offline ? "Local · offline inference" : "Local server connected";
  } catch (_) {
    $("connection").className = "status-pill offline";
    $("connection-label").textContent = "Server disconnected";
  }
}

function clearAutomaticTranscript() {
  if (automaticTranscript()) {
    $("transcript").value = "";
    updateTranscriptMode();
  }
}

function selectFile(selected) {
  if (busy || !selected) return;
  invalidateResult();
  file = null;
  clearAutomaticTranscript();
  $("audio-info").textContent = "No WAV selected.";
  $("record-clock").textContent = "00:00";
  if (selected.size > 8 * 1024 * 1024) return fail(new Error("Choose a WAV file smaller than 8 MiB."));
  if (!selected.name.toLowerCase().endsWith(".wav")) return fail(new Error("Choose a .wav recording."));
  file = selected;
  $("audio-info").textContent = `${(file.size / 1024).toFixed(0)} KiB · WAV recording`;
  controls(false);
  message(automaticTranscript() ? "Recording selected. Click Analyze audio to transcribe it and predict an emotion." : "Recording selected. Check its transcript, then analyze.");
  log(`WAV selected: ${file.name}`);
}

$("audio-file").addEventListener("change", (event) => selectFile(event.target.files[0]));
for (const name of ["dragenter", "dragover"]) $("upload-zone").addEventListener(name, (event) => {
  event.preventDefault();
  if (!busy) $("upload-zone").classList.add("dragover");
});
for (const name of ["dragleave", "drop"]) $("upload-zone").addEventListener(name, (event) => {
  event.preventDefault();
  $("upload-zone").classList.remove("dragover");
  if (name === "drop") selectFile(event.dataTransfer.files[0]);
});

function processing(stage) {
  const transcribing = stage === "transcribing";
  $("result-state").textContent = transcribing ? "Transcribing…" : "Analyzing…";
  message(transcribing ? "Transcribing and analyzing…" : "Analyzing…");
}

function displayResult(result) {
  $("empty-result").hidden = true;
  $("result").hidden = false;
  if (typeof result.transcript === "string") $("transcript").value = result.transcript;
  updateTranscriptMode();
  if (result.transcript_source === "automatic") {
    $("transcript-note").textContent = "Recognized locally from this recording";
    $("transcript-helper").textContent = "Use Edit transcript to correct any words, then Analyze audio.";
    log(`Speech transcribed locally${typeof result.transcription_ms === "number" ? ` · ${result.transcription_ms.toFixed(0)} ms` : ""}.`);
  }
  $("emotion").textContent = result.emotion;
  $("confidence").textContent = `${(result.confidence * 100).toFixed(0)}%`;
  $("response").textContent = result.response;
  $("latency").textContent = typeof result.latency_ms === "number" ? `${result.latency_ms.toFixed(0)} ms` : "—";
  $("transcription-latency").textContent = result.transcript_source === "provided" ? "Skipped · your text" : typeof result.transcription_ms === "number" ? `${result.transcription_ms.toFixed(0)} ms` : "—";
  const status = result.modality_status;
  const statusLabels = { text_and_audio: "Text + audio", text_only_silent_audio: "Text · audio is silent", text_only_missing_audio: "Text · audio missing" };
  $("modalities").textContent = typeof status === "string" ? statusLabels[status] || status : Object.entries(status || {}).map(([key, value]) => `${key}: ${value}`).join(" · ") || "Text + audio";
  $("scores").replaceChildren();
  Object.entries(result.probabilities || {}).sort((a, b) => b[1] - a[1]).forEach(([emotion, score]) => {
    const row = document.createElement("div");
    row.className = `score-row${emotion === result.emotion ? " winner" : ""}`;
    const name = document.createElement("span");
    name.className = "score-name";
    name.textContent = emotion;
    const bar = document.createElement("div");
    bar.className = "score-bar";
    bar.setAttribute("role", "meter");
    bar.setAttribute("aria-label", `${emotion} probability`);
    bar.setAttribute("aria-valuemin", "0");
    bar.setAttribute("aria-valuemax", "100");
    bar.setAttribute("aria-valuenow", (score * 100).toFixed(1));
    const fill = document.createElement("div");
    fill.className = "score-fill";
    fill.style.width = `${Math.max(0, Math.min(100, score * 100))}%`;
    bar.append(fill);
    const value = document.createElement("span");
    value.className = "score-number";
    value.textContent = `${(score * 100).toFixed(0)}%`;
    row.append(name, bar, value);
    $("scores").append(row);
  });
  $("result-state").textContent = result.uncertain ? "Low confidence" : "Analysis complete";
  message(result.transcript_source === "automatic" ? "Analysis complete. Check the transcript, or record another utterance." : "Analysis complete. You can try another utterance.");
  log(`Prediction: ${result.emotion} · ${(result.confidence * 100).toFixed(1)}% confidence · ${$("latency").textContent} total processing`);
  controls(false);
  health();
}

$("analyze").addEventListener("click", async () => {
  if (busy) return;
  invalidateResult();
  const automatic = automaticTranscript();
  const transcript = automatic ? "" : $("transcript").value.trim();
  if (!file) return fail(new Error("Choose a WAV or record your voice first."));
  if (!automatic && !transcript) return fail(new Error("Enter the spoken words, or choose Transcribe automatically."));
  clearAutomaticTranscript();
  controls(true);
  processing(automatic ? "transcribing" : "analyzing");
  log(automatic ? "Sending audio to local speech recognition and emotion analysis." : "Sending your transcript + audio to local emotion analysis.");
  const form = new FormData();
  if (!automatic) form.append("text", transcript);
  form.append("audio", file);
  try {
    const response = await fetch("/predict", { method: "POST", body: form });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === "string" ? result.detail : "The server could not analyze this input.");
    displayResult(result);
  } catch (error) {
    fail(error);
  }
});

function eventTime() {
  lastEventTime = Math.max(lastEventTime, Date.now());
  return lastEventTime;
}

function sendEvent(type, fields = {}) {
  if (!socket || socket.readyState !== WebSocket.OPEN) throw new Error("Audio connection closed. Please try again.");
  socket.send(JSON.stringify({ type, timestamp_ms: eventTime(), ...fields }));
}

function pcmBase64(samples) {
  const bytes = new Uint8Array(samples.length * 4);
  const view = new DataView(bytes.buffer);
  for (let i = 0; i < samples.length; i += 1) view.setFloat32(i * 4, Math.max(-1, Math.min(1, samples[i])), true);
  let binary = "";
  for (let i = 0; i < bytes.length; i += 1) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

function recordingFile(current) {
  const buffer = new ArrayBuffer(44 + current.samples * 2);
  const view = new DataView(buffer);
  const writeText = (offset, text) => [...text].forEach((character, i) => view.setUint8(offset + i, character.charCodeAt(0)));
  writeText(0, "RIFF");
  view.setUint32(4, 36 + current.samples * 2, true);
  writeText(8, "WAVE");
  writeText(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, current.context.sampleRate, true);
  view.setUint32(28, current.context.sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeText(36, "data");
  view.setUint32(40, current.samples * 2, true);
  let offset = 44;
  for (const chunk of current.chunks) for (const raw of chunk) {
    const sample = Math.max(-1, Math.min(1, raw));
    view.setInt16(offset, Math.round(sample * (sample < 0 ? 32768 : 32767)), true);
    offset += 2;
  }
  return new File([buffer], `${current.id}.wav`, { type: "audio/wav" });
}

async function releaseMicrophone() {
  clearInterval(clockTimer);
  clockTimer = null;
  const current = recording;
  recording = null;
  if (current) {
    current.processor.onaudioprocess = null;
    current.source.disconnect();
    current.processor.disconnect();
    current.silence.disconnect();
    current.stream.getTracks().forEach((track) => track.stop());
    if (current.context.state !== "closed") await current.context.close();
  }
  return current;
}

function closeSocket(ws) {
  if (!ws) return;
  if (socket === ws) socket = null;
  ws.close();
}

async function streamFailed(ws, error) {
  if (socket !== ws) return;
  awaitingStreamResult = false;
  closeSocket(ws);
  await releaseMicrophone();
  fail(error);
}

async function finishRecording() {
  if (!recording) return;
  $("stop").disabled = true;
  const ws = socket;
  const current = recording;
  file = recordingFile(current);
  $("audio-info").textContent = `${(current.samples / current.context.sampleRate).toFixed(2)} seconds · ready to reanalyze`;
  await releaseMicrophone();
  // A disconnect can finish cleanup while AudioContext.close() is pending.
  // Never send an old utterance to a replacement connection or lock its controls.
  if (socket !== ws) return;
  controls(true);
  const transcript = automaticTranscript() ? "" : $("transcript").value.trim();
  if (!automaticTranscript() && !transcript) {
    closeSocket(socket);
    return fail(new Error("Recording saved for this session. Enter a transcript or choose Transcribe automatically, then click Analyze audio."));
  }
  try {
    if (transcript) sendEvent("text", { utterance_id: current.id, text: transcript });
    awaitingStreamResult = true;
    sendEvent("end", { utterance_id: current.id });
    processing(automaticTranscript() ? "transcribing" : "analyzing");
    log(`Utterance ended: ${current.sequence} audio chunks, ${(current.samples / current.context.sampleRate).toFixed(2)} seconds.`);
  } catch (error) {
    awaitingStreamResult = false;
    closeSocket(socket);
    fail(error);
  }
}

$("record").addEventListener("click", async () => {
  if (busy) return;
  invalidateResult();
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) return fail(new Error("Microphone access requires localhost or HTTPS. You can also upload a WAV."));
  controls(true);
  message("Requesting microphone access…");
  let stream = null;
  let context = null;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true }, video: false });
    context = new AudioContext({ sampleRate: 16_000 });
    await context.resume();
    if (context.sampleRate < 8_000 || context.sampleRate > 48_000) throw new Error("This microphone sample rate is unsupported. Please upload a WAV instead.");
    closeSocket(socket);
    const ws = new WebSocket(`${location.protocol === "https:" ? "wss:" : "ws:"}//${location.host}/stream`);
    socket = ws;
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => { ws.close(); reject(new Error("Audio connection timed out.")); }, 10_000);
      ws.onerror = () => { clearTimeout(timeout); reject(new Error("Could not connect to the local audio server.")); };
      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.type === "ready") { clearTimeout(timeout); resolve(); }
        if (data.type === "error") { clearTimeout(timeout); reject(new Error(data.message)); }
      };
      ws.onclose = () => { clearTimeout(timeout); reject(new Error("Audio connection closed before it was ready.")); };
    });
    const source = context.createMediaStreamSource(stream);
    const processor = context.createScriptProcessor(4096, 1, 1);
    const silence = context.createGain();
    silence.gain.value = 0;
    const id = `mic-${Date.now()}`;
    awaitingStreamResult = false;
    file = null;
    $("audio-file").value = "";
    $("audio-info").textContent = "Live microphone · up to 30 seconds";
    clearAutomaticTranscript();
    recording = { id, stream, context, source, processor, silence, sequence: 0, samples: 0, chunks: [] };
    sendEvent("start", { utterance_id: id, sample_rate: context.sampleRate });
    ws.onmessage = async (event) => {
      if (socket !== ws) return;
      try {
        const data = JSON.parse(event.data);
        if (data.type === "processing") processing(data.stage);
        if (data.type === "result") {
          awaitingStreamResult = false;
          displayResult(data);
          closeSocket(ws);
        }
        if (data.type === "error") await streamFailed(ws, new Error(data.message));
      } catch (error) { await streamFailed(ws, error); }
    };
    ws.onclose = async () => {
      if (socket !== ws) return;
      if (recording || awaitingStreamResult) {
        await streamFailed(ws, new Error("Audio connection closed before the prediction finished. If your recording finished, click Analyze audio to retry."));
      } else socket = null;
    };
    ws.onerror = () => streamFailed(ws, new Error("Audio connection failed. Try another recording, or analyze the saved audio."));
    processor.onaudioprocess = (event) => {
      if (!recording) return;
      const samples = event.inputBuffer.getChannelData(0);
      const remaining = recording.context.sampleRate * 30 - recording.samples;
      const chunk = samples.subarray(0, Math.max(0, remaining));
      if (chunk.length) {
        try {
          if (ws.bufferedAmount > 1_048_576) throw new Error("Audio connection is too slow. Please try uploading a WAV.");
          sendEvent("audio", { utterance_id: id, sequence: recording.sequence, encoding: "pcm_f32le", data: pcmBase64(chunk) });
          recording.chunks.push(new Float32Array(chunk));
          recording.sequence += 1;
          recording.samples += chunk.length;
        } catch (error) { streamFailed(ws, error); }
      }
      if (remaining <= samples.length) finishRecording();
    };
    source.connect(processor);
    processor.connect(silence);
    silence.connect(context.destination);
    startedAt = performance.now();
    $("record-clock").textContent = "00:00";
    clockTimer = setInterval(() => {
      const seconds = Math.min(30, Math.floor((performance.now() - startedAt) / 1000));
      $("record-clock").textContent = `00:${String(seconds).padStart(2, "0")}`;
      if (seconds >= 30) finishRecording();
    }, 250);
    controls(true);
    $("result-state").textContent = "Listening…";
    message(automaticTranscript() ? "Speak now. Click Stop & analyze when you finish; your words will be transcribed automatically." : "Speak now. Check or edit your transcript, then click Stop & analyze.");
    log(`Stream started · ${context.sampleRate} Hz mono · ${automaticTranscript() ? "automatic transcription" : "your transcript"}.`);
  } catch (error) {
    if (stream) stream.getTracks().forEach((track) => track.stop());
    if (context && context.state !== "closed") await context.close();
    recording = null;
    closeSocket(socket);
    fail(error);
  }
});

$("stop").addEventListener("click", finishRecording);
$("clear-log").addEventListener("click", () => $("event-log").replaceChildren());
window.addEventListener("beforeunload", () => {
  if (recording) recording.stream.getTracks().forEach((track) => track.stop());
  if (socket) socket.close();
});
updateTranscriptMode();
controls(false);
health();
