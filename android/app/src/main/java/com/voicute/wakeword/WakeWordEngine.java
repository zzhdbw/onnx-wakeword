package com.voicute.wakeword;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Multi-model wake word engine with shared mel + embedding computation.
 *
 * Loads N wake word models from assets/model_info.json, computes mel and
 * embedding ONCE per audio frame using the largest emb_frames, then runs
 * each model's classifier on its own slice of the shared embedding output.
 */
public class WakeWordEngine {

    private static final String TAG = "WakeWordEngine";
    private static final String MEL_MODEL = "melspectrogram.onnx";
    private static final String EMB_MODEL = "embedding_model.onnx";

    // Audio parameters (constant)
    static final int SAMPLE_RATE = 16000;
    static final float MEL_HOP_SEC = 0.010f;
    static final float MEL_WIN_SEC = 0.025f;
    static final int MEL_HOP_SAMPLES = (int) (SAMPLE_RATE * MEL_HOP_SEC);
    static final int EMB_WINDOW = 76;
    static final int EMB_STEP = 8;
    static final int N_MELS = 32;

    /** Detection result with specific wake word name. */
    public static class DetectionResult {
        public final String wakeWord;
        public final float probability;
        /** Mean probability across ALL models — represents background noise level. */
        public final float backgroundMean;

        public DetectionResult(String wakeWord, float probability, float backgroundMean) {
            this.wakeWord = wakeWord;
            this.probability = probability;
            this.backgroundMean = backgroundMean;
        }
    }

    private static class ModelSlot {
        final String wakeWord;
        final String modelFile;
        final int embFrames;
        OrtSession session;

        ModelSlot(String wakeWord, String modelFile, int embFrames) {
            this.wakeWord = wakeWord;
            this.modelFile = modelFile;
            this.embFrames = embFrames;
        }
    }

    private final List<ModelSlot> models = new ArrayList<>();
    private String[] wakeWordNames;
    private int maxEmbFrames;
    private int melFramesNeeded;
    private int audioSamplesNeeded;
    // Multi-class model (single ONNX with softmax)
    private boolean multiClassModel = false;
    private int nClasses = 0;
    private String multiClassModelFile = null;
    private OrtSession multiClassSession = null;

    public int getMelFramesNeeded() { return melFramesNeeded; }
    public int getAudioSamplesNeeded() { return audioSamplesNeeded; }

    /** Pipe-separated display string of all wake words. */
    public String getWakeWordDisplay() {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < wakeWordNames.length; i++) {
            if (i > 0) sb.append(" | ");
            sb.append(wakeWordNames[i]);
        }
        return sb.toString();
    }

    /** Number of wake word models loaded. */
    public int getModelCount() { return models.size(); }

    private final OrtEnvironment env;
    private OrtSession melSession;
    private OrtSession embSession;

    private boolean loaded;
    private int debugLogCount = 0;
    private static final int DEBUG_LOG_MAX = 3;

    public WakeWordEngine(Context context) {
        env = OrtEnvironment.getEnvironment();
        try {
            AssetManager am = context.getAssets();
            byte[] infoBytes;
            try (InputStream is = am.open("model_info.json")) {
                infoBytes = new byte[is.available()];
                int offset = 0;
                while (offset < infoBytes.length) {
                    int read = is.read(infoBytes, offset, infoBytes.length - offset);
                    if (read < 0) break;
                    offset += read;
                }
            }
            JSONObject info = new JSONObject(new String(infoBytes, "UTF-8"));

            // --- Multi-class model (single ONNX, softmax output) ---
            String modelType = info.optString("model_type", "");
            multiClassModel = modelType.equals("multi");

            if (multiClassModel) {
                // Multi-class: one model with N+1 outputs (words + background)
                JSONArray wordArray = info.getJSONArray("wake_words");
                nClasses = info.getInt("n_classes");
                String mFile = info.getString("model_file");
                int ef = info.getInt("emb_frames");
                wakeWordNames = new String[wordArray.length()];
                for (int i = 0; i < wordArray.length(); i++) {
                    wakeWordNames[i] = wordArray.getString(i);
                }
                maxEmbFrames = ef;
                multiClassModelFile = mFile;
                Log.i(TAG, "Multi-class model: " + nClasses + " classes emb_frames=" + ef
                        + " file=" + mFile);
            } else if (info.optBoolean("multi_model", false) && info.has("models")) {
                JSONArray modelArray = info.getJSONArray("models");
                for (int i = 0; i < modelArray.length(); i++) {
                    JSONObject m = modelArray.getJSONObject(i);
                    String word = m.getString("wake_word");
                    String file = m.getString("model_file");
                    int ef = m.getInt("emb_frames");
                    models.add(new ModelSlot(word, file, ef));
                    Log.i(TAG, "Registered: " + word + " emb_frames=" + ef + " file=" + file);
                }
            } else {
                String word = info.getString("wake_word");
                String file = info.getString("model_file");
                int ef = info.optInt("emb_frames", 16);
                models.add(new ModelSlot(word, file, ef));
                Log.i(TAG, "Single model: " + word + " emb_frames=" + ef);
            }

            if (!multiClassModel) {
                wakeWordNames = new String[models.size()];
                maxEmbFrames = 0;
                for (int i = 0; i < models.size(); i++) {
                ModelSlot m = models.get(i);
                wakeWordNames[i] = m.wakeWord;
                if (m.embFrames > maxEmbFrames) maxEmbFrames = m.embFrames;
                }
            }

            melFramesNeeded = EMB_WINDOW + (maxEmbFrames - 1) * EMB_STEP;
            audioSamplesNeeded = melFramesNeeded * MEL_HOP_SAMPLES + (int) (SAMPLE_RATE * MEL_WIN_SEC);
            Log.i(TAG, "maxEmbFrames=" + maxEmbFrames
                    + " melFramesNeeded=" + melFramesNeeded
                    + " audioSamplesNeeded=" + audioSamplesNeeded);

            // Shared feature extractors
            melSession = loadModel(context, MEL_MODEL);
            embSession = loadModel(context, EMB_MODEL);

            if (multiClassModel) {
                multiClassSession = loadModel(context, multiClassModelFile);
            } else {
                // Each wake word classifier
                for (ModelSlot m : models) {
                    m.session = loadModel(context, m.modelFile);
                }
            }

            loaded = true;
            Log.i(TAG, "All " + models.size() + " models loaded");
        } catch (Exception e) {
            Log.e(TAG, "Failed to load models", e);
            loaded = false;
        }
    }

    private OrtSession loadModel(Context context, String filename) throws IOException, OrtException {
        AssetManager am = context.getAssets();
        byte[] modelBytes;
        try (InputStream is = am.open(filename)) {
            modelBytes = new byte[is.available()];
            int offset = 0;
            while (offset < modelBytes.length) {
                int read = is.read(modelBytes, offset, modelBytes.length - offset);
                if (read < 0) break;
                offset += read;
            }
        }
        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        return env.createSession(modelBytes, opts);
    }

    public boolean isLoaded() { return loaded; }

    /**
     * Run inference on raw 16-bit PCM audio.
     *
     * @param audio 16-bit mono PCM, 16 kHz
     * @return best matching DetectionResult or null on error
     */
    public DetectionResult process(short[] audio) {
        if (!loaded) return null;

        try {
            // 1. Convert to float
            float[] floatAudio = new float[audio.length];
            for (int i = 0; i < audio.length; i++) {
                floatAudio[i] = (float) audio[i];
            }

            // 2. Mel spectrogram (shared — single call for all models)
            OnnxTensor melIn = OnnxTensor.createTensor(env,
                    FloatBuffer.wrap(floatAudio), new long[]{1, audio.length});
            OrtSession.Result melOut = melSession.run(
                    Collections.singletonMap("input", melIn));
            float[][][][] mel = (float[][][][]) melOut.get(0).getValue();
            melIn.close();
            melOut.close();

            int frames = mel[0][0].length;
            if (frames < EMB_WINDOW) return null;

            // 3. Apply transform: x/10 + 2 (shared)
            float[][] mel2d = new float[frames][N_MELS];
            for (int f = 0; f < frames; f++) {
                for (int m = 0; m < N_MELS; m++) {
                    mel2d[f][m] = mel[0][0][f][m] / 10.0f + 2.0f;
                }
            }

            // 4. Embedding batch using maxEmbFrames (shared)
            int startFrame = frames - EMB_WINDOW - (maxEmbFrames - 1) * EMB_STEP;
            if (startFrame < 0) startFrame = 0;

            float[][][][] embBatch = new float[maxEmbFrames][EMB_WINDOW][N_MELS][1];
            for (int w = 0; w < maxEmbFrames; w++) {
                int offset = startFrame + w * EMB_STEP;
                if (offset + EMB_WINDOW > frames) offset = frames - EMB_WINDOW;
                for (int f = 0; f < EMB_WINDOW; f++) {
                    for (int m = 0; m < N_MELS; m++) {
                        embBatch[w][f][m][0] = mel2d[offset + f][m];
                    }
                }
            }

            OnnxTensor embIn = OnnxTensor.createTensor(env, embBatch);
            OrtSession.Result embOut = embSession.run(
                    Collections.singletonMap("input_1", embIn));
            float[][][][] embeddings = (float[][][][]) embOut.get(0).getValue();
            embIn.close();
            embOut.close();

            // 5. Multi-class model path
            if (multiClassModel) {
                // Flatten all embedding frames into single input vector
                int flatDim = maxEmbFrames * 96;
                float[][] multiIn = new float[1][flatDim];
                int idx = 0;
                for (int w = 0; w < maxEmbFrames; w++) {
                    for (int d = 0; d < 96; d++) {
                        multiIn[0][idx++] = embeddings[w][0][0][d];
                    }
                }

                OnnxTensor mIn = OnnxTensor.createTensor(env, multiIn);
                OrtSession.Result mOut = multiClassSession.run(
                        Collections.singletonMap("input", mIn));
                float[][] probs = (float[][]) mOut.get(0).getValue();
                mIn.close();
                mOut.close();

                // probs[0] = [p_word0, p_word1, ..., p_background]
                int K = nClasses - 1;  // number of wake words (exclude background)
                float bestScore = -1f;
                String bestWord = null;
                for (int i = 0; i < K; i++) {
                    if (probs[0][i] > bestScore) {
                        bestScore = probs[0][i];
                        bestWord = wakeWordNames[i];
                    }
                }
                float backgroundProb = probs[0][K];

                return new DetectionResult(bestWord, bestScore, backgroundProb);
            }

            // 6. Binary model path — collect sigmoid outputs
            float[] sigmoidProbs = new float[models.size()];
            for (int i = 0; i < models.size(); i++) {
                ModelSlot model = models.get(i);
                int sliceStart = maxEmbFrames - model.embFrames;
                float[][][] wakeInput = new float[1][model.embFrames][96];
                for (int w = 0; w < model.embFrames; w++) {
                    for (int d = 0; d < 96; d++) {
                        wakeInput[0][w][d] = embeddings[sliceStart + w][0][0][d];
                    }
                }

                OnnxTensor wakeIn = OnnxTensor.createTensor(env, wakeInput);
                OrtSession.Result wakeOut = model.session.run(
                        Collections.singletonMap("input", wakeIn));
                float[][][] score = (float[][][]) wakeOut.get(0).getValue();
                sigmoidProbs[i] = score[0][0][0];
                wakeIn.close();
                wakeOut.close();
            }

            // 6. Convert sigmoid → logit → softmax (multi-class with background)
            int K = models.size();
            float[] logits = new float[K + 1];  // +1 = background class
            float maxLogit = -Float.MAX_VALUE;

            for (int i = 0; i < K; i++) {
                // Inverse sigmoid: logit = ln(p / (1-p))
                float p = Math.max(sigmoidProbs[i], 1e-6f);
                p = Math.min(p, 1f - 1e-6f);
                logits[i] = (float) Math.log(p / (1f - p));
                if (logits[i] > maxLogit) maxLogit = logits[i];
            }
            logits[K] = 0f;   // background class logit

            // Softmax with numerical stability
            float sumExp = 0f;
            float[] softmaxProbs = new float[K + 1];
            for (int i = 0; i <= K; i++) {
                softmaxProbs[i] = (float) Math.exp(logits[i] - maxLogit);
                sumExp += softmaxProbs[i];
            }
            for (int i = 0; i <= K; i++) {
                softmaxProbs[i] /= sumExp;
            }

            // 7. Find best non-background word
            float bestScore = -1f;
            String bestWord = null;
            for (int i = 0; i < K; i++) {
                if (softmaxProbs[i] > bestScore) {
                    bestScore = softmaxProbs[i];
                    bestWord = models.get(i).wakeWord;
                }
            }
            float backgroundProb = softmaxProbs[K];

            if (debugLogCount < DEBUG_LOG_MAX) {
                Log.d(TAG, String.format(Locale.US,
                        "[DEBUG %d] frames=%d maxEmb=%d best=%s softmax=%.2f%% bg=%.2f%%",
                        debugLogCount, frames, maxEmbFrames, bestWord, bestScore * 100, backgroundProb * 100));
                debugLogCount++;
            }

            return new DetectionResult(bestWord, bestScore, backgroundProb);

        } catch (OrtException e) {
            Log.e(TAG, "Inference error", e);
            return null;
        }
    }

    public void close() {
        try {
            if (melSession != null) melSession.close();
            if (embSession != null) embSession.close();
            for (ModelSlot m : models) {
                if (m.session != null) m.session.close();
            }
            if (env != null) env.close();
        } catch (OrtException e) {
            Log.e(TAG, "Error closing sessions", e);
        }
    }
}
