# 轻量离线关键词识别 / 唤醒词识别 / KWS

> **离线运行 · 模型 < 1MB · 推理 < 10ms · 不耗流量 · 不上传音频**

ONNX 是通用模型格式。拿到 mel + embedding + classifier 三个 ONNX 文件，就能在 Android、ESP32、Linux、Web 等任何支持 ONNX Runtime 的设备上跑**关键词识别**（Keyword Spotting / 唤醒词 / Wake Word）。不联网，不挑训练工具，不依赖云端。

适合：智能家居语音控制、AI 玩具、对讲机声控、桌面语音助手、ESP32 离线语音模块。

## 获取 ONNX 模型

### 自己训练

以下工具都能导出 ONNX 唤醒词模型：

- [OpenWakeWord](https://github.com/dscripka/openWakeWord) — 开源，家居场景优化，支持多唤醒词，社区活跃
- [NanoWakeWord](https://github.com/arcosoph/nanowakeword) — 开源，11 种架构可选(DNN/TCN/Conformer 等)，轻量至极(40KB)
- [Porcupine](https://github.com/Picovoice/porcupine) — Picovoice 商业引擎，精度高，免费额度有限
- [MicroWakeWord](https://github.com/kahrendt/microWakeWord) — 专为 ESP32 等 MCU 设计，TinyML
- [TensorFlow KWS](https://www.tensorflow.org/tutorials/audio/simple_audio) — Google 官方教程，适合自定义研究
- 任何支持导出 ONNX 的 KWS 框架均可

### 不想折腾训练

[voicute.com](https://www.voicute.com) 输入关键词 → 自动生成 ONNX 模型。首次免费。

## ONNX 模型文件

无论用哪个工具训练，最终都需要三个文件：

| 文件 | 说明 |
|------|------|
| `melspectrogram.onnx` | 音频 → 梅尔频谱，通用模块，**本仓库 `models/` 已提供** |
| `embedding_model.onnx` | 梅尔频谱 → 特征向量，通用模块，**本仓库 `models/` 已提供** |
| `你的模型.onnx` | 特征向量 → 唤醒词概率，从 voicute.com 下载，或自己训练 |

> melspectrogram 和 embedding 是所有唤醒词共用的通用模型，训练/下载时不需要重复获取。只需更换分类器 ONNX 即可切换唤醒词。

**使用 TFLite？** ONNX 可通过 [onnx2tf](https://github.com/PINTO0309/onnx2tf) 一键转换为 TFLite 格式，适配 ESP32 等 TFLite Micro 设备。详见 [`esp32/`](esp32/)。

外加一个 `model_info.json` 描述模型配置。

如果用一个模型区分多个唤醒词（多分类），只需一个分类器 ONNX，输出 N+1 个 softmax 概率。

## 模型配置

### 单个唤醒词
```json
{
  "wake_word": "你好灯灯",
  "model_file": "nihaodengdeng.onnx",
  "emb_frames": 16,
  "threshold": 0.5
}
```

### 多个唤醒词（独立训练，推理时合并）
```json
{
  "multi_model": true,
  "models": [
    { "wake_word": "打开灯光", "model_file": "dakaidengguang.onnx", "emb_frames": 16 },
    { "wake_word": "你好电脑", "model_file": "nihaodiannao.onnx", "emb_frames": 16 }
  ]
}
```

### 多分类模型（一次训练多个词）
```json
{
  "model_type": "multi",
  "wake_words": ["punch in", "start break"],
  "model_file": "multi.onnx",
  "emb_frames": 16,
  "n_classes": 3
}
```

## 推理架构

```
音频(16kHz) → MelSpectrogram → Embedding → 分类器 → Softmax → 唤醒词
```

单模型：分类器输出 sigmoid 概率
多模型：各分类器 sigmoid → logit → 拼背景类 → softmax → 取最大

## 各平台调用

| 目录 | 平台 | 状态 |
|------|------|:---:|
| [`android/`](android/) | Android (Java) | ✅ |
| [`web/`](web/) | Web (JavaScript) | ✅ |
| [`linux/`](linux/) | Linux (Python) | ✅ |
| [`esp32/`](esp32/) | ESP32-S3/P4 | 🚧 |

### Android

复制模型到 `assets/`，依赖 `onnxruntime-android`，调用 [`WakeWordEngine.java`](android/WakeWordEngine.java)。

```java
WakeWordEngine engine = new WakeWordEngine(context);
DetectionResult result = engine.process(audioChunk);
// result.wakeWord → 识别到的唤醒词
// result.probability → softmax 概率
```

### Web

```javascript
import { init, detect } from './web/wakeword.js';
await init('/models/');
const result = detect(audioSamples);
// result.wakeWord, result.probability, result.allProbs
```

### Linux

```bash
pip install onnxruntime numpy sounddevice
python linux/infer.py --model-dir ./models/
```

## 嵌入文件说明

`models/` 目录下的 `melspectrogram.onnx` 和 `embedding_model.onnx` 是预提取的共享特征模型。替换为你自己训练的同名文件即可。
