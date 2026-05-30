# 模型目录

放置训练好的 ONNX 唤醒词模型。

## 共享模型（必须）

这两个文件是通用的，**本项目已提供**，无需额外下载：

| 文件 | 说明 |
|------|------|
| `melspectrogram.onnx` | 音频 → 梅尔频谱，通用模块，**本项目已提供** |
| `embedding_model.onnx` | 梅尔频谱 → 特征向量，通用模块，**本项目已提供** |

## 如何使用

### 单个唤醒词

1. 把训练好的模型文件放入 `models/`
2. 编辑 `model_info.json`：

```json
{
  "wake_word": "你的唤醒词",
  "model_file": "your_model.onnx",
  "emb_frames": 16,
  "threshold": 0.5
}
```

### 多个唤醒词

当同时使用多个唤醒词时，各分类器独立运行，通过 softmax 实现互斥（不会同时触发）。

```json
{
  "multi_model": true,
  "models": [
    { "wake_word": "打开灯光", "model_file": "dakaidengguang.onnx", "emb_frames": 16 },
    { "wake_word": "你好电脑", "model_file": "nihaodiannao.onnx",   "emb_frames": 16 },
    { "wake_word": "开始播放", "model_file": "kaishibofang.onnx",   "emb_frames": 16 }
  ]
}
```

### 使用已提供的演示模型

当前 `models/` 目录已包含 4 个中文演示模型（来自 [voicute.com](https://www.voicute.com)），可直接测试：

| 唤醒词 | 模型文件 |
|--------|----------|
| 曼波 | `manbo_v8_wakeword.onnx` |
| 你好电脑 | `nihaodiannao_v8_wakeword.onnx` |
| 开始播放 | `kaishibofang_v8_wakeword.onnx` |
| 来福 | `laifu_v8_wakeword.onnx` |

这 4 个模型已配置为 `multi_model` 模式，可直接体验多词同时唤醒。

## `emb_frames` 是什么

训练时自动确定的嵌入帧数，记录在训练输出的 `model_info.json` 中。常见值 16~23。**必须和模型训练时一致**，填写错误会导致无法识别。
