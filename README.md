# 轻量离线关键词识别 / 唤醒词识别 / KWS

> 关键词识别、唤醒词识别、离线语音识别、Keyword Spotting、Wake Word、KWS、ONNX、TFLite、ESP32

**Offline · < 1MB · < 10ms inference · No cloud · No data upload** | [中文说明](README_ZH.md)

ONNX is a universal model format. Grab the mel + embedding + classifier ONNX files and run wake word detection on **any device** with ONNX Runtime — Android, Linux, Web. TFLite for ESP32. No internet, no cloud dependency.

Use cases: smart home voice control, AI toys, PTT radio voice activation, desktop assistants, ESP32 offline voice modules.

## Get ONNX Models

### Train your own

All of these can export ONNX wake word models:

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — open source, home automation optimized, multi-wake-word
- [NanoWakeWord](https://github.com/arcosoph/nanowakeword) — open source, 11 architectures (DNN/TCN/Conformer), tiny (40KB)
- [Porcupine](https://github.com/Picovoice/porcupine) — commercial, high accuracy
- [MicroWakeWord](https://github.com/kahrendt/microWakeWord) — designed for ESP32, TinyML
- [TensorFlow KWS](https://www.tensorflow.org/tutorials/audio/simple_audio) — Google tutorial, research-friendly
- Any KWS framework that exports ONNX

### Don't want to train?

[voicute.com](https://www.voicute.com) — type a keyword, get an ONNX model. First one free.

## ONNX Model Files

Three files needed regardless of training tool:

| File | Description |
|------|-------------|
| `melspectrogram.onnx` | Audio → mel spectrogram. Shared across all wake words. **Included in this repo.** |
| `embedding_model.onnx` | Mel spectrogram → features. Shared across all wake words. **Included in this repo.** |
| `your_model.onnx` | Features → wake word probability. One per word. From voicute.com or your own training. |

> melspectrogram and embedding are universal modules. You only need to swap the classifier ONNX to change wake words.

**Using TFLite?** ONNX → TFLite via [onnx2tf](https://github.com/PINTO0309/onnx2tf) for ESP32 and TFLite Micro devices. See [`esp32/`](esp32/).

Plus a `model_info.json` config file.

Multi-class models (one ONNX for multiple wake words) output N+1 softmax probabilities.

## Model Config

### Single wake word
```json
{
  "wake_word": "hello computer",
  "model_file": "hello_computer.onnx",
  "emb_frames": 16,
  "threshold": 0.5
}
```

### Multiple wake words
```json
{
  "multi_model": true,
  "models": [
    { "wake_word": "turn on lights", "model_file": "lights.onnx", "emb_frames": 16 },
    { "wake_word": "hello computer", "model_file": "computer.onnx", "emb_frames": 16 }
  ]
}
```

### Multi-class model (single ONNX, multiple words)
```json
{
  "model_type": "multi",
  "wake_words": ["punch in", "start break"],
  "model_file": "multi.onnx",
  "emb_frames": 16,
  "n_classes": 3
}
```

## Inference Pipeline

```
Audio(16kHz) → MelSpectrogram → Embedding → Classifier(s) → Softmax → Wake Word
```

Single model: classifier outputs sigmoid probability.
Multi-model: each classifier sigmoid → logit → +background class → softmax → argmax.

## Platform Support

| Directory | Platform | Status |
|-----------|----------|:------:|
| [`android/`](android/) | Android (Java) | ✅ |
| [`web/`](web/) | Web (JavaScript) | ✅ |
| [`linux/`](linux/) | Linux x86_64 / arm64 (Python, ONNX) | ✅ |
| [`linux/`](linux/) | Raspberry Pi 32-bit (Python, TFLite) | ✅ |
| [`esp32/`](esp32/) | ESP32-S3/P4 | 🚧 TFLite |

### Android

Place models in `assets/`, depends on `onnxruntime-android`. See [`WakeWordEngine.java`](android/app/src/main/java/com/voicute/wakeword/WakeWordEngine.java).

```java
WakeWordEngine engine = new WakeWordEngine(context);
DetectionResult result = engine.process(audioChunk);
// result.wakeWord, result.probability
```

### Web

双击项目根目录的 `run.bat`，浏览器打开 `http://localhost:8080/web/`。

或手动：`python -m http.server 8080`，无需复制模型文件。

### Linux

```bash
pip install onnxruntime numpy sounddevice
python linux/infer.py --model-dir ./models/
```

## Included Demo Models

`models/` contains 4 Chinese demo wake words from [voicute.com](https://www.voicute.com):

| Wake Word | Model File |
|-----------|-----------|
| 曼波 | `manbo_v8_wakeword.onnx` |
| 你好电脑 | `nihaodiannao_v8_wakeword.onnx` |
| 开始播放 | `kaishibofang_v8_wakeword.onnx` |
| 来福 | `laifu_v8_wakeword.onnx` |

Configured as multi-model — test multi-word detection out of the box.
