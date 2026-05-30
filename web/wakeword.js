/**
 * Voicute Wake Word — Web Inference (ONNX Runtime Web)
 *
 * Usage:
 *   1. npm install onnxruntime-web
 *   2. Place melspectrogram.onnx + embedding_model.onnx + your_model.onnx in /models/
 *   3. Update model_info.json with your wake word(s)
 *   4. Call startListening(callback)
 *
 * Single model:  sigmoid output directly
 * Multi model:   sigmoid → logit → softmax with background class
 */

const SAMPLE_RATE = 16000;
const MEL_HOP_SAMPLES = 160;
const EMB_WINDOW = 76;
const EMB_STEP = 8;

let melSession = null;
let embSession = null;
let models = [];       // { name, session, embFrames }
let maxEmbFrames = 0;

// ── Load models ──

async function loadModel(path) {
    const response = await fetch(path);
    const buffer = await response.arrayBuffer();
    return await ort.InferenceSession.create(buffer);
}

async function init(modelBasePath = '/models/') {
    melSession = await loadModel(modelBasePath + 'melspectrogram.onnx');
    embSession = await loadModel(modelBasePath + 'embedding_model.onnx');

    const infoResp = await fetch(modelBasePath + 'model_info.json');
    const info = await infoResp.json();

    if (info.multi_model && info.models) {
        for (const m of info.models) {
            const session = await loadModel(modelBasePath + m.model_file);
            models.push({ name: m.wake_word, session, embFrames: m.emb_frames });
            if (m.emb_frames > maxEmbFrames) maxEmbFrames = m.emb_frames;
        }
    } else {
        const session = await loadModel(modelBasePath + info.model_file);
        models.push({ name: info.wake_word, session, embFrames: info.emb_frames || 16 });
        maxEmbFrames = info.emb_frames || 16;
    }
    console.log(`Loaded ${models.length} model(s), maxEmbFrames=${maxEmbFrames}`);
}

// ── Inference ──

function detect(audioSamples) {
    if (!melSession || !embSession || models.length === 0) return null;

    const audioTensor = new ort.Tensor('float32', audioSamples, [1, audioSamples.length]);
    const melOut = melSession.run({ input: audioTensor });
    const mel = melOut.output.data;  // shape [1, 1, frames, 32]
    const frames = mel.length / 32;

    // Transform: x/10 + 2
    const mel2d = new Float32Array(frames * 32);
    for (let i = 0; i < frames * 32; i++) mel2d[i] = mel[i] / 10.0 + 2.0;

    // Build embedding batch
    const startFrame = Math.max(0, frames - EMB_WINDOW - (maxEmbFrames - 1) * EMB_STEP);
    const embBatch = new Float32Array(maxEmbFrames * EMB_WINDOW * 32);
    for (let w = 0; w < maxEmbFrames; w++) {
        let offset = Math.min(startFrame + w * EMB_STEP, frames - EMB_WINDOW);
        for (let f = 0; f < EMB_WINDOW; f++) {
            for (let m = 0; m < 32; m++) {
                embBatch[w * EMB_WINDOW * 32 + f * 32 + m] = mel2d[(offset + f) * 32 + m];
            }
        }
    }

    const embTensor = new ort.Tensor('float32', embBatch, [maxEmbFrames, EMB_WINDOW, 32, 1]);
    const embOut = embSession.run({ input_1: embTensor });
    const embeddings = embOut.output.data;  // [maxEmbFrames * 96]

    // Run classifiers
    const sigmoidProbs = [];
    for (const model of models) {
        const sliceStart = maxEmbFrames - model.embFrames;
        const input = new Float32Array(model.embFrames * 96);
        let idx = 0;
        for (let w = sliceStart; w < sliceStart + model.embFrames; w++) {
            for (let d = 0; d < 96; d++) {
                input[idx++] = embeddings[w * 96 + d];
            }
        }

        const wakeTensor = new ort.Tensor('float32', input, [1, model.embFrames, 96]);
        const wakeOut = model.session.run({ input: wakeTensor });
        sigmoidProbs.push(wakeOut.output.data[0]);
    }

    // Sigmoid → Softmax
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
    for (let i = 0; i < K; i++) {
        softmax[i] /= sumExp;
        if (softmax[i] > bestScore) {
            bestScore = softmax[i];
            bestName = models[i].name;
        }
    }

    return {
        wakeWord: bestName,
        probability: bestScore,
        background: softmax[K] / sumExp,
        allProbs: softmax.slice(0, K).reduce((acc, p, i) => {
            acc[models[i].name] = p; return acc;
        }, {}),
    };
}

// ── Microphone streaming ──

async function startListening(callback, threshold = 0.5) {
    const stream = await navigator.mediaDevices.getUserMedia({
        audio: { sampleRate: SAMPLE_RATE, channelCount: 1, echoCancellation: false,
                  noiseSuppression: false, autoGainControl: false }
    });

    const audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);

    const buffer = new Int16Array(SAMPLE_RATE * 4);
    let bufferPos = 0;
    let lastDetect = 0;
    const SAMPLES_NEEDED = (EMB_WINDOW + (maxEmbFrames - 1) * EMB_STEP) * MEL_HOP_SAMPLES + 400;

    processor.onaudioprocess = (e) => {
        const input = e.inputBuffer.getChannelData(0);
        for (let i = 0; i < input.length; i++) {
            buffer[bufferPos] = Math.max(-32768, Math.min(32767, Math.round(input[i] * 32768)));
            bufferPos = (bufferPos + 1) % buffer.length;
        }

        if (bufferPos < SAMPLES_NEEDED) return;

        const audioChunk = new Float32Array(SAMPLES_NEEDED);
        for (let i = 0; i < SAMPLES_NEEDED; i++) {
            audioChunk[i] = buffer[(bufferPos - SAMPLES_NEEDED + i + buffer.length) % buffer.length] / 32768.0;
        }

        const result = detect(audioChunk);
        if (result && result.probability > threshold && (Date.now() - lastDetect) > 3500) {
            lastDetect = Date.now();
            callback(result);
        }
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
    return { stream, audioContext, source, processor };
}
