# Lightweight Offline Wake Word / Keyword Spotting

> **Offline · Model < 130KB · No cloud · No audio upload · ESP32/Android/Linux/Web**

[中文说明](README_ZH.md)

Plug in ONNX models and run wake word detection on any device with ONNX Runtime. No internet, no cloud dependency.

Use cases: smart home voice control, AI toys, desktop assistants, ESP32 offline voice modules.

## Web Demo

![Screenshot](web/screenshot.png)

Built-in anti-false-trigger panel with 5-layer detection toggles, L5 ratio slider, threshold control, and confidence bar.

## Get ONNX Models

### Train your own

All of these can export ONNX wake word models:

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — open source, home automation, multi-wake-word
- [MicroWakeWord](https://github.com/kahrendt/microWakeWord) — designed for ESP32, TinyML
- [NanoWakeWord](https://github.com/arcosoph/nanowakeword) — 11 architectures, tiny (40KB)
- Any KWS framework that exports ONNX

### Quick start (no training)

[voicute.com](https://www.voicute.com) — enter a wake word, get an ONNX model. First one free.

## Required Files

| File | Description |
|------|-------------|
| `melspectrogram.onnx` | Audio → Mel spectrogram (shared). **Provided in `models/`** |
| `your_model.onnx` | Wake word detection. Get from training or online service |

Plus a `model_info.json` config file.

## Model Config

```json
{
  "model_type": "dscnn",
  "mel_time": 98,
  "multi_model": true,
  "models": [
    {"wake_word": "曼波", "model_file": "dscnn_multiscale_manbo.onnx", "cons_frames": 3}
  ]
}
```

## Project Structure

```
onnx-wakeword/
├── android/     # Android (Java, ONNX Runtime)
├── web/         # Web (JavaScript, ONNX Runtime Web)
├── linux/       # Linux / Raspberry Pi (Python)
├── esp32/       # ESP32-S3/P4
└── models/      # Shared mel model + demo wake word models
```

## Platform Quick Start

| Platform | Directory | Entry Point |
|----------|-----------|-------------|
| Android | `android/` | `WakeWordEngine.java` |
| Web | `web/` | `wakeword.js` → `VoicuteWakeWord.create()` |
| Linux | `linux/` | `wakeword_engine.py` → `WakeWordEngine()` |

### Web

```html
<script src="wakeword.js"></script>
<script>
  const engine = VoicuteWakeWord.create();
  await engine.load('model_info.json', 'melspectrogram.onnx');
  // Supports local paths, URLs, and COS ZIP packages
  engine.set_L1(true);
  await engine.start((word, prob) => {
    console.log(`Detected: ${word} (${(prob*100).toFixed(0)}%)`);
  });
</script>
```

### Linux

```bash
pip install onnxruntime numpy pyaudio
```

```python
from wakeword_engine import WakeWordEngine
engine = WakeWordEngine()
engine.load('model_info.json', 'melspectrogram.onnx')
engine.set_L1(True)
engine.start(lambda word, prob, info: print(f'{word}'))
```

### Android

Copy models to `assets/`, build with Android Studio.

```java
WakeWordEngine engine = new WakeWordEngine(context);
DetectionResult result = engine.process(audioChunk);
```

## Anti-False-Trigger Layers

5 configurable layers. Enable all for production:

| Layer | Default | What it prevents |
|:---:|:---:|------|
| L1 Consecutive | on / 2~3 frames | Keyboard clicks, chair creaks (transient noise) |
| L2 Peak/BG | off | Model hallucination on quiet background |
| L3 Cooldown | off | Duplicate triggers from a single utterance |
| L4 Burst | off | Dense false triggers from audio feedback loops |
| L5 Energy Jump | off | Video playback, background music |

L5 ratio: adjustable 2.0-8.0x (default 3.0) via `engine.set_L5_ratio(v)`.

## Version History

**v9.0 (2026-06)**
- End-to-end inference architecture, ~128KB models
- Web/Linux/Android unified SDK API
- L1-L5 five-layer detection pipeline, individually toggleable
- L5 energy jump detection (curRms/preMin), blocks video/music false triggers
- COS ZIP direct loading
- Quiet-room guard (preMin<50 && rms<80)
- Auto cons_frames (2-char=2, 3+char=3)
