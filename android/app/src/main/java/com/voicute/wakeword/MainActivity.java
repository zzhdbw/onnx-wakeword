package com.voicute.wakeword;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.graphics.Typeface;
import android.media.AudioFormat;
import android.media.AudioRecord;
import android.media.MediaRecorder;
import android.media.audiofx.NoiseSuppressor;
import android.os.Bundle;
import android.os.Handler;
import android.os.Looper;
import android.os.PowerManager;
import android.util.Log;
import android.view.Gravity;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.SeekBar;
import android.widget.TextView;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    private static final String TAG = "Voicute";
    private static final int PERM_REQ = 1001;

    private static final long DETECT_COOLDOWN_MS = 3500;
    private static final long INFERENCE_INTERVAL_MS = 30;

    private static final int THRESHOLD_OFFSET = 0;

    private static final int CONF_HISTORY_SIZE = 128;
    private final float[] confHistory = new float[CONF_HISTORY_SIZE];
    private final String[] confHistoryWord = new String[CONF_HISTORY_SIZE];
    private final long[] confHistoryTime = new long[CONF_HISTORY_SIZE];
    private int confHistoryIdx = 0;
    private static final long CONF_HISTORY_WINDOW_MS = 1500;


    private WakeWordEngine engine;
    private AudioRecord recorder;
    private Thread captureThread;
    private Thread inferThread;
    private Handler mainHandler;

    private volatile boolean running;
    private volatile boolean captureAlive;
    private volatile float latestScore;
    private volatile float latestProb;
    private volatile String latestWord = "";
    private volatile long lastInferMs;
    private volatile float threshold = 0.60f;

    private TextView statusText;
    private TextView scoreText;
    private TextView detectText;
    private TextView counterText;
    private TextView thresholdText;
    private SeekBar thresholdSeekBar;
    private Button toggleButton;
    private Button clearButton;

    private LinearLayout logContainer;
    private ScrollView logScrollView;
    private NoiseSuppressor noiseSuppressor;

    private long lastDetectTime;
    private long lastUiUpdate;
    private long lastDebugLog;
    private long lastAudioActivityMs;
    private int detectCount;
    private SimpleDateFormat timeFmt = new SimpleDateFormat("HH:mm:ss", Locale.US);

    private PowerManager.WakeLock wakeLock;
    private static final long CAPTURE_RESTART_IDLE_MS = 300_000;

    private final int RING_SIZE = WakeWordEngine.SAMPLE_RATE * 3;
    private short[] ringBuffer;
    private volatile int ringPos;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        statusText = findViewById(R.id.statusText);
        scoreText = findViewById(R.id.scoreText);
        detectText = findViewById(R.id.detectText);
        counterText = findViewById(R.id.counterText);
        thresholdText = findViewById(R.id.thresholdText);
        thresholdSeekBar = findViewById(R.id.thresholdSeekBar);
        toggleButton = findViewById(R.id.toggleButton);
        clearButton = findViewById(R.id.clearButton);
        logContainer = findViewById(R.id.logContainer);
        logScrollView = findViewById(R.id.logScrollView);

        mainHandler = new Handler(Looper.getMainLooper());

        thresholdSeekBar.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
            @Override
            public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                threshold = (THRESHOLD_OFFSET + progress) / 100.0f;
                thresholdText.setText(String.format(Locale.US, "%d%%", THRESHOLD_OFFSET + progress));
            }
            @Override
            public void onStartTrackingTouch(SeekBar seekBar) {}
            @Override
            public void onStopTrackingTouch(SeekBar seekBar) {}
        });
        threshold = (THRESHOLD_OFFSET + thresholdSeekBar.getProgress()) / 100.0f;
        thresholdText.setText(String.format(Locale.US, "%d%%",
                THRESHOLD_OFFSET + thresholdSeekBar.getProgress()));

        toggleButton.setOnClickListener(v -> {
            if (running) {
                stopListening();
            } else {
                startListening();
            }
        });

        clearButton.setOnClickListener(v -> clearLog());

        resetDisplay();

        new Thread(() -> {
            engine = new WakeWordEngine(MainActivity.this);
            runOnUiThread(() -> {
                if (engine.isLoaded()) {
                    String display = engine.getWakeWordDisplay();
                    statusText.setText("已加载 " + engine.getModelCount() + " 个模型 - 点击开始");
                    // Update the layout subtitle with actual wake words
                    TextView subtitle = findViewById(R.id.wakeWordSubtitle);
                    if (subtitle != null) {
                        subtitle.setText("唤醒词: " + display);
                    }
                    toggleButton.setEnabled(true);
                } else {
                    statusText.setText("模型加载失败");
                }
            });
        }).start();
        toggleButton.setEnabled(false);
    }

    private void resetDisplay() {
        detectCount = 0;
        lastDetectTime = 0;
        lastUiUpdate = 0;
        lastInferMs = 0;
        detectText.setText("");
        scoreText.setText("");
        counterText.setText("");
    }

    private void clearLog() {
        logContainer.removeAllViews();
        detectCount = 0;
        counterText.setText("");
        detectText.setText("");
    }

    private void startListening() {
        if (!hasPermission()) {
            ActivityCompat.requestPermissions(this,
                    new String[]{Manifest.permission.RECORD_AUDIO}, PERM_REQ);
            return;
        }

        running = true;
        resetDisplay();
        toggleButton.setText("停止");
        String display = engine.getWakeWordDisplay();
        statusText.setText("正在监听 \"" + display + "\"...");

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        PowerManager pm = (PowerManager) getSystemService(Context.POWER_SERVICE);
        if (pm != null) {
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "Voicute:WakeWord");
            wakeLock.acquire();
        }

        ringBuffer = new short[RING_SIZE];
        ringPos = 0;

        captureThread = new Thread(this::captureLoop, "AudioCapture");
        captureThread.start();

        inferThread = new Thread(this::inferenceLoop, "Inference");
        inferThread.start();
    }

    private void stopListening() {
        running = false;
        toggleButton.setText("开始");
        statusText.setText("已停止");

        getWindow().clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (wakeLock != null && wakeLock.isHeld()) {
            wakeLock.release();
            wakeLock = null;
        }

        for (Thread t : new Thread[]{captureThread, inferThread}) {
            if (t != null) {
                t.interrupt();
                try { t.join(500); } catch (InterruptedException ignored) {}
            }
        }
        captureThread = null;
        inferThread = null;

        if (noiseSuppressor != null) {
            noiseSuppressor.setEnabled(false);
            noiseSuppressor.release();
            noiseSuppressor = null;
        }
        if (recorder != null) {
            recorder.stop();
            recorder.release();
            recorder = null;
        }
    }

    private void restartCapture() {
        if (recorder != null) {
            try { recorder.stop(); } catch (Exception ignored) {}
            try { recorder.release(); } catch (Exception ignored) {}
            recorder = null;
        }
        if (noiseSuppressor != null) {
            try { noiseSuppressor.setEnabled(false); } catch (Exception ignored) {}
            try { noiseSuppressor.release(); } catch (Exception ignored) {}
            noiseSuppressor = null;
        }
        captureAlive = false;
        captureThread = new Thread(this::captureLoop, "AudioCapture");
        captureThread.start();
    }

    // --- Audio capture ---

    private void captureLoop() {
        try {
            captureAlive = false;
            int bufferSize = AudioRecord.getMinBufferSize(
                    WakeWordEngine.SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT);
            bufferSize = Math.max(bufferSize, engine.getAudioSamplesNeeded() * 2);

            recorder = new AudioRecord(
                    MediaRecorder.AudioSource.VOICE_RECOGNITION,
                    WakeWordEngine.SAMPLE_RATE,
                    AudioFormat.CHANNEL_IN_MONO,
                    AudioFormat.ENCODING_PCM_16BIT,
                    bufferSize);

            if (recorder.getState() != AudioRecord.STATE_INITIALIZED) {
                mainHandler.post(() -> statusText.setText("麦克风初始化失败"));
                running = false;
                return;
            }

            if (NoiseSuppressor.isAvailable()) {
                noiseSuppressor = NoiseSuppressor.create(recorder.getAudioSessionId());
                if (noiseSuppressor != null) noiseSuppressor.setEnabled(true);
                Log.i(TAG, "NoiseSuppressor enabled: " + (noiseSuppressor != null));
            }

            recorder.startRecording();
            captureAlive = true;

            short[] readChunk = new short[WakeWordEngine.MEL_HOP_SAMPLES * 8];
            long lastCaptureLog = 0;
            long recordStartMs = System.currentTimeMillis();
            float maxRmsSinceStart = 0;
            int zeroReadCount = 0;
            boolean micWarningShown = false;

            while (running && !Thread.interrupted()) {
                int read = recorder.read(readChunk, 0, readChunk.length);
                if (read <= 0) {
                    zeroReadCount++;
                    if (zeroReadCount > 10 && !micWarningShown) {
                        micWarningShown = true;
                        Log.w(TAG, "AudioRecord.read returns 0 — mic likely occupied by another app");
                        mainHandler.post(() -> statusText.setText("麦克风被占用，请关闭其他使用麦克风的应用"));
                    }
                    continue;
                }
                zeroReadCount = 0;

                for (int i = 0; i < read; i++) {
                    ringBuffer[ringPos] = readChunk[i];
                    ringPos = (ringPos + 1) % RING_SIZE;
                }

                float rms = rmsOf(readChunk, read);
                if (rms > maxRmsSinceStart) maxRmsSinceStart = rms;
                if (rms > 200) {
                    lastAudioActivityMs = System.currentTimeMillis();
                }

                // Mic occupancy check: after 3s, if RMS never exceeded 50, mic is likely locked
                if (!micWarningShown && (System.currentTimeMillis() - recordStartMs) > 3000
                        && maxRmsSinceStart < 50) {
                    micWarningShown = true;
                    Log.w(TAG, "Mic occupancy detected: max RMS=" + String.format(Locale.US, "%.1f", maxRmsSinceStart));
                    mainHandler.post(() -> statusText.setText("麦克风被占用，请关闭其他使用麦克风的应用"));
                }

                long now = System.currentTimeMillis();
                if (now - lastCaptureLog > 5000) {
                    lastCaptureLog = now;
                    Log.d(TAG, "Capture alive | ringPos=" + ringPos
                            + " rms=" + String.format(Locale.US, "%.1f", rms));
                }
            }

            recorder.stop();
            recorder.release();
            recorder = null;
            captureAlive = false;
        } catch (Exception e) {
            Log.e(TAG, "Capture thread crashed", e);
            captureAlive = false;
            if (recorder != null) {
                try { recorder.stop(); } catch (Exception ignored) {}
                try { recorder.release(); } catch (Exception ignored) {}
                recorder = null;
            }
        }
    }

    private float rmsOf(short[] buf, int len) {
        double sum = 0;
        for (int i = 0; i < len; i++) sum += (double) buf[i] * buf[i];
        return (float) Math.sqrt(sum / len);
    }

    // --- Inference ---

    private void inferenceLoop() {
        try { Thread.sleep(500); } catch (InterruptedException e) { return; }

        long lastHeartbeat = 0;

        while (running && !Thread.interrupted()) {
            long t0 = System.currentTimeMillis();

            if (!captureAlive && running) {
                Log.w(TAG, "Capture thread dead — restarting capture");
                restartCapture();
                try { Thread.sleep(800); } catch (InterruptedException e) { break; }
                if (!captureAlive) {
                    mainHandler.post(() -> statusText.setText("麦克风错误 - 请重启应用"));
                    break;
                }
            }

            if (running && captureAlive && lastAudioActivityMs > 0
                    && (t0 - lastAudioActivityMs) > CAPTURE_RESTART_IDLE_MS
                    && (t0 - lastDetectTime) > CAPTURE_RESTART_IDLE_MS) {
                Log.w(TAG, "Audio idle 5min — restarting capture to wake hardware");
                restartCapture();
                lastAudioActivityMs = t0;
            }

            int pos = ringPos;

            short[] audioChunk = new short[engine.getAudioSamplesNeeded()];
            for (int i = 0; i < engine.getAudioSamplesNeeded(); i++) {
                int idx = (pos - engine.getAudioSamplesNeeded() + i + RING_SIZE) % RING_SIZE;
                audioChunk[i] = ringBuffer[idx];
            }

            WakeWordEngine.DetectionResult result = engine.process(audioChunk);

            long t1 = System.currentTimeMillis();
            lastInferMs = t1 - t0;

            String word = "";
            float prob = 0;
            float bgProb = 0;
            if (result != null) {
                word = result.wakeWord != null ? result.wakeWord : "";
                prob = result.probability;        // softmax probability of best word
                bgProb = result.backgroundMean;   // softmax probability of background class
            }

            latestScore = prob;
            latestProb = prob;
            latestWord = word;

            // Record in confidence history ring buffer
            confHistory[confHistoryIdx] = prob;
            confHistoryWord[confHistoryIdx] = word;
            confHistoryTime[confHistoryIdx] = t1;
            confHistoryIdx = (confHistoryIdx + 1) % CONF_HISTORY_SIZE;

            // Find peak over recent window
            float peakProb = 0;
            String peakWord = "";
            for (int i = 0; i < CONF_HISTORY_SIZE; i++) {
                if (confHistoryTime[i] > 0 && (t1 - confHistoryTime[i]) < CONF_HISTORY_WINDOW_MS) {
                    if (confHistory[i] > peakProb) {
                        peakProb = confHistory[i];
                        peakWord = confHistoryWord[i] != null ? confHistoryWord[i] : "";
                    }
                }
            }

            if (t1 - lastDebugLog > 1000) {
                lastDebugLog = t1;
                float maxVal = 0, sumSq = 0;
                for (short s : audioChunk) {
                    float abs = Math.abs((float) s);
                    if (abs > maxVal) maxVal = abs;
                    sumSq += (float) s * (float) s;
                }
                float rms = (float) Math.sqrt(sumSq / audioChunk.length);
                Log.d(TAG, String.format(Locale.US,
                        "word=%s prob=%.1f%% peak=%.1f%% thr=%.0f%% bg=%.1f%% | infer=%dms pcm max=%.0f rms=%.1f capture=%b",
                        peakWord, prob * 100, peakProb * 100, threshold * 100, bgProb * 100, lastInferMs, maxVal, rms, captureAlive));
            }

            if (t1 - lastHeartbeat > 30000) {
                lastHeartbeat = t1;
                Log.i(TAG, "Inference alive | captureAlive=" + captureAlive
                        + " ringPos=" + ringPos);
            }

            // Detection: softmax probability > threshold + cooldown
            if (peakProb > threshold && !peakWord.isEmpty()
                    && (t1 - lastDetectTime) > DETECT_COOLDOWN_MS) {
                lastDetectTime = t1;
                detectCount++;
                final float detectedProb = peakProb;
                final String detectedWord = peakWord;
                for (int i = 0; i < CONF_HISTORY_SIZE; i++) {
                    confHistory[i] = 0;
                    confHistoryWord[i] = "";
                    confHistoryTime[i] = 0;
                }
                mainHandler.post(() -> showDetection(detectedWord, detectedProb));
            }

            if ((t1 - lastUiUpdate) > 400) {
                lastUiUpdate = t1;
                mainHandler.post(this::updateUI);
            }

            long elapsed = System.currentTimeMillis() - t0;
            long sleepMs = INFERENCE_INTERVAL_MS - elapsed;
            if (sleepMs > 0) {
                try { Thread.sleep(sleepMs); } catch (InterruptedException e) { break; }
            }
        }
    }

    // --- UI callbacks ---

    private void showDetection(String word, float prob) {
        detectText.setText(String.format(Locale.US,
                "%s 检测到! (%.0f%%)", word, prob * 100));
        detectText.setAlpha(1f);
        counterText.setText(String.format(Locale.US, "已检测到 %d 次", detectCount));
        addLogEntry(word, prob);

        mainHandler.postDelayed(() -> {
            detectText.animate().alpha(0f).setDuration(400).start();
        }, 1200);
    }

    private void addLogEntry(String word, float prob) {
        String now = timeFmt.format(new Date());

        LinearLayout row = new LinearLayout(this);
        row.setOrientation(LinearLayout.HORIZONTAL);
        row.setPadding(8, 6, 8, 6);
        row.setBackgroundColor(detectCount % 2 == 0 ? 0xFF1a1a2e : 0xFF16213e);

        TextView idxView = new TextView(this);
        idxView.setText(String.format(Locale.US, "#%d", detectCount));
        idxView.setTextSize(11);
        idxView.setTextColor(0xFF888888);
        idxView.setWidth(60);
        row.addView(idxView);

        TextView msgView = new TextView(this);
        msgView.setText(word);
        msgView.setTextSize(12);
        msgView.setTextColor(0xFFe94560);
        msgView.setTypeface(null, Typeface.BOLD);
        LinearLayout.LayoutParams msgParams = new LinearLayout.LayoutParams(
                0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f);
        msgView.setLayoutParams(msgParams);
        row.addView(msgView);

        TextView timeView = new TextView(this);
        timeView.setText(now);
        timeView.setTextSize(10);
        timeView.setTextColor(0xFF999999);
        timeView.setWidth(100);
        timeView.setGravity(Gravity.END);
        row.addView(timeView);

        TextView probView = new TextView(this);
        probView.setText(String.format(Locale.US, "%.0f%%", prob * 100));
        probView.setTextSize(11);
        probView.setTextColor(prob > 0.8f ? 0xFF16c79a : 0xFFffd93d);
        probView.setTypeface(null, Typeface.BOLD);
        probView.setWidth(60);
        probView.setGravity(Gravity.END);
        row.addView(probView);

        logContainer.addView(row, 0);

        if (logContainer.getChildCount() > 50) {
            logContainer.removeViewAt(logContainer.getChildCount() - 1);
        }

        logScrollView.post(() -> logScrollView.fullScroll(ScrollView.FOCUS_UP));
    }

    private void updateUI() {
        float peak = 0;
        long now = System.currentTimeMillis();
        for (int i = 0; i < CONF_HISTORY_SIZE; i++) {
            if (confHistoryTime[i] > 0 && (now - confHistoryTime[i]) < CONF_HISTORY_WINDOW_MS) {
                if (confHistory[i] > peak) peak = confHistory[i];
            }
        }
        scoreText.setText(String.format(Locale.US,
                "word:%s prob:%.1f%% peak:%.0f%% %dms",
                latestWord, latestProb * 100, peak * 100, lastInferMs));
    }

    private boolean hasPermission() {
        return ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)
                == PackageManager.PERMISSION_GRANTED;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] perms, int[] results) {
        super.onRequestPermissionsResult(requestCode, perms, results);
        if (requestCode == PERM_REQ && results.length > 0
                && results[0] == PackageManager.PERMISSION_GRANTED) {
            startListening();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        stopListening();
        if (engine != null) {
            new Thread(() -> engine.close()).start();
        }
    }
}
