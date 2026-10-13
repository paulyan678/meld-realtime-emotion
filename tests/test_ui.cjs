// Exercise the actual browser event handlers with deterministic browser adapters.
// No microphone permission, network connection, or npm dependency is required.
const assert = require("node:assert/strict");
const { File } = require("node:buffer");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const staticPath = path.join(__dirname, "../src/meld_emotion/static");
const html = fs.readFileSync(path.join(staticPath, "index.html"), "utf8");
const script = fs.readFileSync(path.join(staticPath, "app.js"), "utf8");
const prediction = (source = "automatic") => ({
  type: "result", transcript: "HELLO WORLD", transcript_source: source,
  transcription_ms: source === "automatic" ? 42 : 0, latency_ms: 55,
  emotion: "joy", confidence: 0.6, probabilities: { joy: 0.6, neutral: 0.4 },
  modality_status: "text_and_audio", response: "Hello", uncertain: false,
});
const deferred = () => {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
};

function browser({ predict, microphoneError } = {}) {
  const elements = {};
  class Element {
    constructor(attributes = "") {
      this.listeners = {}; this.value = ""; this.children = []; this.style = {};
      this.hidden = /\bhidden\b/.test(attributes); this.disabled = /\bdisabled\b/.test(attributes);
      this.name = attributes.match(/name="([^"]+)"/)?.[1];
      this.type = attributes.match(/type="([^"]+)"/)?.[1];
      this._checked = /\bchecked\b/.test(attributes);
      this.classList = { toggle() {}, add() {}, remove() {} };
      this.clientWidth = 300; this.clientHeight = 30;
    }
    set checked(value) {
      if (value && this.type === "radio") {
        for (const other of Object.values(elements)) if (other.name === this.name) other._checked = false;
      }
      this._checked = value;
    }
    get checked() { return this._checked; }
    addEventListener(name, handler) { this.listeners[name] = handler; }
    append(...children) { this.children.push(...children); }
    prepend(child) { this.children.unshift(child); }
    replaceChildren(...children) { this.children = children; }
    setAttribute() {}
    focus() { this.focused = true; }
    getContext() { return { clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {} }; }
  }
  for (const tag of html.matchAll(/<[^>]+\bid="([^"]+)"[^>]*>/g)) elements[tag[1]] = new Element(tag[0]);
  const requests = [], sockets = [], contexts = [], tracks = [];
  class Socket {
    static OPEN = 1;
    constructor() {
      this.readyState = 1; this.bufferedAmount = 0; this.events = []; sockets.push(this);
      queueMicrotask(() => this.receive({ type: "ready" }));
    }
    send(data) { this.events.push(JSON.parse(data)); }
    receive(data) { return this.onmessage?.({ data: JSON.stringify(data) }); }
    close() { this.readyState = 3; queueMicrotask(() => this.onclose?.()); }
    fail() { return this.onerror?.(); }
  }
  class AudioContext {
    constructor() { this.sampleRate = 16000; this.state = "running"; this.destination = {}; contexts.push(this); }
    async resume() {}
    async close() { this.state = "closed"; if (this.closeWait) await this.closeWait.promise; }
    createMediaStreamSource() { return { connect() {}, disconnect() {} }; }
    createScriptProcessor() { this.processor = { connect() {}, disconnect() {} }; return this.processor; }
    createGain() { return { gain: {}, connect() {}, disconnect() {} }; }
  }
  const context = vm.createContext({
    document: {
      getElementById: (id) => { assert.ok(elements[id], `Missing DOM element ${id}`); return elements[id]; },
      createElement: () => new Element(),
    },
    window: { addEventListener() {}, devicePixelRatio: 1 },
    navigator: { mediaDevices: { async getUserMedia() {
      if (microphoneError) throw microphoneError;
      const track = { stopped: false, stop() { this.stopped = true; } }; tracks.push(track);
      return { getTracks: () => [track] };
    } } },
    location: { protocol: "http:", host: "localhost" },
    AudioContext, WebSocket: Socket, File, FormData, Float32Array, Uint8Array, DataView, ArrayBuffer, Error,
    btoa: (value) => Buffer.from(value, "binary").toString("base64"),
    performance, setTimeout, clearTimeout,
    // Timers are not needed: tests deliver samples and click Stop explicitly.
    setInterval: () => 1, clearInterval() {},
    async fetch(url, options) {
      if (url === "/health") return { ok: true, json: async () => ({ offline: true }) };
      requests.push(options);
      if (predict) return predict(options);
      return { ok: true, json: async () => prediction(options.body.has("text") ? "provided" : "automatic") };
    },
  });
  vm.runInContext(script, context);
  return {
    elements, requests, sockets, contexts, tracks,
    click: (id) => elements[id].listeners.click(),
    upload: (file = new File(["RIFF"], "speech.wav")) => elements["audio-file"].listeners.change({ target: { files: [file] } }),
    mode(id) { elements[id].checked = true; elements[id].listeners.change(); },
    audio(samples = new Float32Array(1600).fill(0.25)) {
      contexts.at(-1).processor.onaudioprocess({ inputBuffer: { getChannelData: () => samples } });
    },
  };
}

test("default automatic upload omits text and exposes the recognized transcript", async () => {
  const ui = browser();
  assert.equal(ui.elements["mode-auto"].checked, true);
  assert.equal(ui.elements.transcript.readOnly, true);
  assert.equal(ui.elements.analyze.disabled, true);
  ui.upload();
  await ui.click("analyze");
  assert.equal(ui.requests[0].body.has("text"), false);
  assert.equal(ui.elements.transcript.value, "HELLO WORLD");
  assert.equal(ui.elements["edit-transcript"].hidden, false);
  assert.equal(ui.elements["transcription-latency"].textContent, "42 ms");
});

test("explicit correction sends the edited text and reuses the same audio", async () => {
  const ui = browser();
  const audio = new File(["original audio"], "speech.wav");
  ui.upload(audio);
  await ui.click("analyze");
  ui.click("edit-transcript");
  assert.equal(ui.elements["mode-manual"].checked, true);
  assert.equal(ui.elements.transcript.readOnly, false);
  ui.elements.transcript.value = "Corrected words";
  await ui.click("analyze");
  assert.equal(ui.requests[1].body.get("text"), "Corrected words");
  assert.equal(await ui.requests[1].body.get("audio").text(), "original audio");
  assert.equal(ui.elements["transcription-latency"].textContent, "Skipped · your text");
});

test("automatic retries and fresh input never reuse previous transcript text", async () => {
  const ui = browser();
  ui.upload();
  await ui.click("analyze");
  ui.mode("mode-manual");
  ui.elements.transcript.value = "Old correction";
  ui.mode("mode-auto");
  await ui.click("analyze");
  assert.equal(ui.requests[1].body.has("text"), false);
  ui.upload(new File(["new audio"], "next.wav"));
  assert.equal(ui.elements.transcript.value, "");
  assert.equal(ui.elements["edit-transcript"].hidden, true);
  await ui.click("analyze");
  assert.equal(ui.requests[2].body.has("text"), false);
});

test("microphone auto path streams PCM without text and retains WAV for correction", async () => {
  const ui = browser();
  ui.upload();
  await ui.click("analyze");
  assert.equal(ui.elements.result.hidden, false);
  ui.elements.transcript.value = "Stale previous result";
  await ui.click("record");
  const ws = ui.sockets[0];
  assert.equal(ui.elements.result.hidden, true);
  assert.equal(ui.elements.emotion.textContent, "");
  assert.equal(ui.elements.latency.textContent, "—");
  assert.equal(ui.elements.transcript.value, "");
  assert.equal(ui.elements.stop.disabled, false);
  assert.equal(ui.elements.analyze.disabled, true);
  const samples = new Float32Array(1600).fill(0.25);
  samples[0] = -1; samples[1599] = 1;
  ui.audio(samples);
  await ui.click("stop");
  assert.deepEqual(ws.events.map((event) => event.type), ["start", "audio", "end"]);
  assert.equal(Buffer.from(ws.events[1].data, "base64").readFloatLE(0), -1);
  assert.equal(ui.tracks[0].stopped, true);
  assert.equal(ui.contexts[0].state, "closed");
  await ws.receive({ type: "processing", stage: "transcribing" });
  assert.equal(ui.elements["result-state"].textContent, "Transcribing…");
  await ws.receive(prediction());
  ui.click("edit-transcript");
  ui.elements.transcript.value = "Actual spoken words";
  await ui.click("analyze");
  const form = ui.requests[1].body;
  assert.equal(form.get("text"), "Actual spoken words");
  const wav = Buffer.from(await form.get("audio").arrayBuffer());
  assert.equal(wav.toString("ascii", 0, 4), "RIFF");
  assert.equal(wav.readUInt32LE(24), 16000);
  assert.equal(wav.readUInt32LE(40), 3200);
  assert.equal(wav.readInt16LE(44), -32768);
  assert.equal(wav.readInt16LE(wav.length - 2), 32767);
});

test("pending HTTP request blocks duplicate input, and failure permits retry", async () => {
  const response = deferred();
  let count = 0;
  const ui = browser({ predict: () => ++count === 1
    ? { ok: true, json: async () => prediction() }
    : response.promise });
  ui.upload();
  await ui.click("analyze");
  assert.equal(ui.elements.result.hidden, false);
  const pending = ui.click("analyze");
  assert.equal(ui.elements.result.hidden, true);
  assert.equal(ui.elements.response.textContent, "");
  assert.equal(ui.elements.scores.children.length, 0);
  assert.equal(ui.elements.latency.textContent, "—");
  assert.equal(ui.elements.record.disabled, true);
  assert.equal(ui.elements["mode-manual"].disabled, true);
  await ui.click("analyze");
  assert.equal(ui.requests.length, 2);
  response.resolve({ ok: false, json: async () => ({ detail: "No speech recognized" }) });
  await pending;
  assert.equal(ui.elements.message.textContent, "No speech recognized");
  assert.equal(ui.elements.analyze.disabled, false);
  assert.equal(ui.elements.record.disabled, false);
  assert.equal(ui.elements["mode-manual"].disabled, false);
});

test("new audio clears the old result through failure and a corrected retry", async () => {
  const failure = deferred();
  const retry = deferred();
  let count = 0;
  const ui = browser({ predict: () => {
    count += 1;
    if (count === 1) return { ok: true, json: async () => prediction() };
    return count === 2 ? failure.promise : retry.promise;
  } });
  const assertNoResult = () => {
    assert.equal(ui.elements.result.hidden, true);
    for (const id of ["emotion", "confidence", "response"]) assert.equal(ui.elements[id].textContent, "");
    for (const id of ["latency", "transcription-latency", "modalities"]) assert.equal(ui.elements[id].textContent, "—");
    assert.equal(ui.elements.scores.children.length, 0);
  };
  ui.upload();
  await ui.click("analyze");
  assert.equal(ui.elements.result.hidden, false);
  assert.equal(ui.elements.emotion.textContent, "joy");
  assert.equal(ui.elements.response.textContent, "Hello");
  assert.equal(ui.elements.scores.children.length, 2);

  const newAudio = new File(["audio needing a manual transcript"], "next.wav");
  ui.upload(newAudio);
  assertNoResult();
  assert.equal(ui.elements["result-state"].textContent, "Awaiting input");
  const pending = ui.click("analyze");
  assertNoResult();
  failure.resolve({ ok: false, json: async () => ({ detail: "No speech recognized" }) });
  await pending;
  assertNoResult();
  assert.equal(ui.elements["result-state"].textContent, "Needs attention");
  assert.equal(ui.elements.analyze.disabled, false);

  ui.mode("mode-manual");
  ui.elements.transcript.value = "I feel sad today";
  const retried = ui.click("analyze");
  assertNoResult();
  assert.equal(ui.elements.transcript.value, "I feel sad today");
  assert.equal(ui.requests[2].body.get("text"), "I feel sad today");
  assert.equal(await ui.requests[2].body.get("audio").text(), await newAudio.text());
  retry.resolve({ ok: true, json: async () => ({
    ...prediction("provided"), transcript: "I feel sad today", emotion: "sadness",
    confidence: 0.7, probabilities: { sadness: 0.7, neutral: 0.3 }, response: "That sounds difficult.",
  }) });
  await retried;
  assert.equal(ui.elements.result.hidden, false);
  assert.equal(ui.elements.emotion.textContent, "sadness");
  assert.equal(ui.elements.confidence.textContent, "70%");
  assert.equal(ui.elements.response.textContent, "That sounds difficult.");
  assert.equal(ui.elements["result-state"].textContent, "Analysis complete");
});

test("invalid file selection cannot display or reanalyze the previous recording", async () => {
  const ui = browser();
  ui.upload();
  await ui.click("analyze");
  assert.equal(ui.elements.result.hidden, false);
  ui.upload(new File(["not audio"], "notes.txt"));
  assert.equal(ui.elements.result.hidden, true);
  assert.equal(ui.elements.emotion.textContent, "");
  assert.equal(ui.elements.response.textContent, "");
  assert.equal(ui.elements.scores.children.length, 0);
  assert.equal(ui.elements.analyze.disabled, true);
  assert.equal(ui.elements.message.textContent, "Choose a .wav recording.");
  await ui.click("analyze");
  assert.equal(ui.requests.length, 1);
  assert.equal(ui.elements.message.textContent, "Choose a WAV or record your voice first.");
});

test("microphone permission failure restores usable controls", async () => {
  const ui = browser({ microphoneError: new Error("Microphone denied") });
  await ui.click("record");
  assert.equal(ui.elements.record.disabled, false);
  assert.equal(ui.elements.record.hidden, false);
  assert.equal(ui.elements.stop.hidden, true);
  assert.equal(ui.elements.analyze.disabled, true);
  assert.equal(ui.elements.message.textContent, "Microphone denied");
});

test("disconnect while stopping cannot send stale end to a new recording", async () => {
  const ui = browser();
  await ui.click("record");
  ui.audio();
  const oldSocket = ui.sockets[0];
  const closeWait = deferred();
  ui.contexts[0].closeWait = closeWait;
  const stopped = ui.click("stop");
  await oldSocket.fail();
  assert.equal(ui.elements.analyze.disabled, false, "captured audio remains available after disconnect");
  await ui.click("record");
  const newSocket = ui.sockets[1];
  assert.equal(ui.elements.stop.disabled, false, "new recording has a usable Stop button");
  closeWait.resolve();
  await stopped;
  assert.deepEqual(newSocket.events.map((event) => event.type), ["start"]);
  assert.equal(ui.elements["result-state"].textContent, "Listening…");
  ui.audio();
  await ui.click("stop");
  assert.deepEqual(newSocket.events.map((event) => event.type), ["start", "audio", "end"]);
  await newSocket.receive({ type: "error", message: "No speech recognized" });
  assert.equal(ui.elements.analyze.disabled, false);
  assert.equal(ui.elements.record.disabled, false);
});
