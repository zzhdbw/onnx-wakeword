/**
 * Voicute Wake Word — Web Inference (ONNX Runtime Web)
 *
 * Matches Android WakeWordEngine logic:
 *   Audio → Mel → Embedding → N classifiers → sigmoid → logit → softmax → trigger
 */
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const SAMPLE_RATE = 16000;
const MEL_HOP_SAMPLES = 160;
const MEL_WIN_SAMPLES = 400;
const EMB_WINDOW = 76;
const EMB_STEP = 8;
const N_MELS = 32;

let melSession = null;
let embSession = null;
let models = [];       // { name, session, embFrames }
let maxEmbFrames = 0;
let audioSamplesNeeded = 0;

// ── Load models from model_info.json ──

async function loadModel(url) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`Failed to load ${url}: ${resp.status}`);
    const buf = await resp.arrayBuffer();
    return await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
}

async function loadModels(modelInfoUrl, melUrl, embUrl) {
    melSession = await loadModel(melUrl);
    embSession = await loadModel(embUrl);

    // Base path for model files relative to model_info.json
    const basePath = modelInfoUrl.substring(0, modelInfoUrl.lastIndexOf('/') + 1);

    const infoResp = await fetch(modelInfoUrl);
    const info = await infoResp.json();

    if (info.multi_model && info.models) {
        for (const m of info.models) {
            const session = await loadModel(basePath + m.model_file);
            models.push({ name: m.wake_word, session, embFrames: m.emb_frames });
            if (m.emb_frames > maxEmbFrames) maxEmbFrames = m.emb_frames;
        }
    } else if (info.model_type === 'multi') {
        // Multi-class model: single ONNX with softmax output
        // Handled in process() separately
        for (const name of info.wake_words) {
            models.push({ name, session: await loadModel(info.model_file), embFrames: info.emb_frames, multiClass: true });
        }
        maxEmbFrames = info.emb_frames;
    } else {
        const session = await loadModel(info.model_file);
        models.push({ name: info.wake_word, session, embFrames: info.emb_frames || 16 });
        maxEmbFrames = info.emb_frames || 16;
    }
    audioSamplesNeeded = (EMB_WINDOW + (maxEmbFrames - 1) * EMB_STEP) * MEL_HOP_SAMPLES + MEL_WIN_SAMPLES;
    console.log(`[wakeword] ${models.length} model(s), maxEmbFrames=${maxEmbFrames}, audioSamplesNeeded=${audioSamplesNeeded}`);
}

// ── Process one audio chunk ──

async function process(audioData) {
    if (!melSession || !embSession || models.length === 0) return null;

    // 1. Mel spectrogram
    const melIn = new ort.Tensor('float32', audioData, [1, audioData.length]);
    const melOut = await melSession.run({ input: melIn });
    const mel = melOut[Object.keys(melOut)[0]].data;
    const frames = Math.floor((audioData.length - MEL_WIN_SAMPLES) / MEL_HOP_SAMPLES) + 1;
    if (frames < EMB_WINDOW) return null;

    // 2. Transform: x/10 + 2 (matches Android)
    const mel2d = new Float32Array(frames * N_MELS);
    for (let i = 0; i < frames * N_MELS; i++) mel2d[i] = mel[i] / 10.0 + 2.0;

    // 3. Embedding batch
    let startFrame = frames - EMB_WINDOW - (maxEmbFrames - 1) * EMB_STEP;
    if (startFrame < 0) startFrame = 0;
    const embBatch = new Float32Array(maxEmbFrames * EMB_WINDOW * N_MELS);
    for (let w = 0; w < maxEmbFrames; w++) {
        let off = startFrame + w * EMB_STEP;
        if (off + EMB_WINDOW > frames) off = frames - EMB_WINDOW;
        for (let f = 0; f < EMB_WINDOW; f++) {
            for (let m = 0; m < N_MELS; m++) {
                embBatch[w * EMB_WINDOW * N_MELS + f * N_MELS + m] = mel2d[(off + f) * N_MELS + m];
            }
        }
    }

    const embIn = new ort.Tensor('float32', embBatch, [maxEmbFrames, EMB_WINDOW, N_MELS, 1]);
    const embOut = await embSession.run({ input_1: embIn });
    const embeddings = embOut[Object.keys(embOut)[0]].data;  // [maxEmbFrames * 96]

    // 4. Run each classifier with its emb_frames slice
    const sigmoidProbs = [];
    const wordNames = [];
    for (const model of models) {
        const ef = model.embFrames;
        const sliceStart = maxEmbFrames - ef;
        const wakeInput = new Float32Array(ef * 96);
        let idx = 0;
        for (let w = sliceStart; w < sliceStart + ef; w++) {
            for (let d = 0; d < 96; d++) {
                wakeInput[idx++] = embeddings[w * 96 + d];
            }
        }
        const wakeIn = new ort.Tensor('float32', wakeInput, [1, ef, 96]);
        const wakeOut = await model.session.run({ input: wakeIn });
        sigmoidProbs.push(wakeOut[Object.keys(wakeOut)[0]].data[0]);
        wordNames.push(model.name);
    }

    // 5. Sigmoid → logit → softmax (matches Android)
    const K = models.length;
    const logits = new Array(K + 1);
    let maxLogit = -Infinity;
    for (let i = 0; i < K; i++) {
        const p = Math.max(1e-6, Math.min(1 - 1e-6, sigmoidProbs[i]));
        logits[i] = Math.log(p / (1 - p));
        if (logits[i] > maxLogit) maxLogit = logits[i];
    }
    logits[K] = 0;  // background class

    let sumExp = 0;
    const softmax = new Array(K + 1);
    for (let i = 0; i <= K; i++) {
        softmax[i] = Math.exp(logits[i] - maxLogit);
        sumExp += softmax[i];
    }

    let bestScore = -1, bestName = null;
    const allProbs = {};
    for (let i = 0; i < K; i++) {
        softmax[i] /= sumExp;
        allProbs[wordNames[i]] = softmax[i];
        if (softmax[i] > bestScore) {
            bestScore = softmax[i];
            bestName = wordNames[i];
        }
    }
    const bgProb = softmax[K] / sumExp;

    return {
        wakeWord: bestName,
        probability: bestScore,
        background: bgProb,
        allProbs,
    };
}

// ── Relative confidence gate ──

function createConfidenceGate(opts) {
    const {
        bgEmaAlpha = 0.002,
        relativeRatio = 3.0,
        absoluteThreshold = 0.5,
        cooldownMs = 3500,
        initialBackground = 0.01,
    } = opts || {};

    let backgroundEMA = initialBackground;
    let lastDetectTime = 0;

    function reset() {
        backgroundEMA = initialBackground;
        lastDetectTime = 0;
    }

    function onScore(score) {
        backgroundEMA = bgEmaAlpha * score + (1 - bgEmaAlpha) * backgroundEMA;
        if (backgroundEMA < 0.001) backgroundEMA = initialBackground;
        const ratio = score / backgroundEMA;
        const now = Date.now();
        if (score > absoluteThreshold && ratio > relativeRatio && (now - lastDetectTime) > cooldownMs) {
            lastDetectTime = now;
            return true;
        }
        return false;
    }

    return { onScore, reset };
}

// ── Microphone streaming ──

function createDetector() {
    let audioCtx = null;
    let stream = null;
    let listening = false;
    let ringBuf = null;
    let ringPos = 0;
    let lastRingPos = 0;
    let processorNode = null;
    let busy = false;

    function resample(audio, fromRate) {
        if (fromRate === SAMPLE_RATE) return audio;
        const ratio = fromRate / SAMPLE_RATE;
        const newLen = Math.round(audio.length / ratio);
        const out = new Float32Array(newLen);
        for (let i = 0; i < newLen; i++) {
            const src = i * ratio;
            const lo = Math.floor(src);
            const hi = Math.min(lo + 1, audio.length - 1);
            const frac = src - lo;
            out[i] = audio[lo] * (1 - frac) + audio[hi] * frac;
        }
        return out;
    }

    async function startMic(onResult) {
        stream = await navigator.mediaDevices.getUserMedia({
            audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
        });
        audioCtx = new AudioContext({ sampleRate: SAMPLE_RATE });
        const hwRate = audioCtx.sampleRate;
        const needed = Math.ceil((audioSamplesNeeded / SAMPLE_RATE) * hwRate);
        const ringSize = hwRate * 3;
        const stride = Math.round(0.25 * hwRate);
        ringBuf = new Float32Array(ringSize);
        ringPos = 0;
        lastRingPos = 0;
        listening = true;
        busy = false;
        let totalCollected = 0;

        const source = audioCtx.createMediaStreamSource(stream);
        processorNode = audioCtx.createScriptProcessor(2048, 1, 1);

        async function runIfReady() {
            if (!listening || busy) return;
            const newSamples = (ringPos - lastRingPos + ringSize) % ringSize;
            if (newSamples < stride) return;
            busy = true;
            const pos = ringPos;
            lastRingPos = pos;
            const raw = new Float32Array(needed);
            for (let i = 0; i < needed; i++) {
                raw[i] = ringBuf[(pos - needed + i + ringSize) % ringSize];
            }
            try {
                const chunk = resample(raw, hwRate);
                const result = await process(chunk);
                if (listening && totalCollected >= needed) {
                    onResult(result ? result.probability : 0, result ? result.allProbs : {});
                }
            } catch (e) {
                console.warn('[wakeword] inference error', e);
            } finally {
                busy = false;
            }
        }

        processorNode.onaudioprocess = (e) => {
            if (!listening) return;
            const input = e.inputBuffer.getChannelData(0);
            for (let i = 0; i < input.length; i++) {
                ringBuf[ringPos] = input[i];
                ringPos = (ringPos + 1) % ringSize;
            }
            totalCollected += input.length;
            runIfReady();
        };
        source.connect(processorNode);
        processorNode.connect(audioCtx.destination);
        console.log('[wakeword] Mic started, hwRate:', hwRate);
    }

    function stopMic() {
        listening = false;
        if (processorNode) { processorNode.disconnect(); processorNode = null; }
        if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
        if (audioCtx && audioCtx.state !== 'closed') { audioCtx.close(); audioCtx = null; }
        ringBuf = null;
        busy = false;
    }

    function isLoaded() { return !!melSession && models.length > 0; }
    function getModelNames() { return models.map(m => m.name); }

    return { loadModels, process, startMic, stopMic, isLoaded, getModelNames, createConfidenceGate };
}
