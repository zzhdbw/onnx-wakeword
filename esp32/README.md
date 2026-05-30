# ESP32-S3 / ESP32-P4 唤醒词

ESP32 原生支持 TFLite Micro，对 ONNX Runtime 支持尚不完善。推荐使用 TFLite 格式部署。

## 格式转换

从 ONNX 模型转换为 TFLite：

```bash
pip install onnx2tf
onnx2tf convert -i your_model.onnx -o ./tflite_output/
```

也可参考 [NanoWakeWord ESP32 示例](https://github.com/arcosoph/nanowakeword/tree/main/examples/esp32)。

## 推理流程

ESP32 上的推理与 ONNX 版本架构相同：

```
音频(16kHz) → MelSpectrogram → Embedding → 分类器 → 唤醒词
```

MelSpectrogram 和 Embedding 为通用模块，可复用。仅分类器因唤醒词不同而需替换。

## 关键指标

| 指标 | 值 |
|------|:---:|
| 模型大小 | < 200KB |
| 推理时间 | < 50ms (ESP32-S3) |
| 内存占用 | < 500KB |
| 功耗 | < 50mW 持续推理 |

## 参考资源

- [ESP-SR](https://github.com/espressif/esp-sr) — Espressif 官方语音识别库
- [MicroWakeWord](https://github.com/kahrendt/microWakeWord) — 专为 ESP32 设计的唤醒词引擎
- [TFLite Micro ESP32](https://docs.espressif.com/projects/esp-tflite-micro) — 官方 TFLite Micro 文档
