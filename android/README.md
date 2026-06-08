# Android 唤醒词 Demo

完整 Android 项目，Android Studio 打开即可运行。

## 使用

1. 复制 ONNX 模型和 `model_info.json` 到 `app/src/main/assets/`
2. Android Studio 打开 `android/` 目录，编译运行

## 模型配置

DS-CNN 架构（推荐）：

```json
{
  "model_type": "dscnn",
  "mel_time": 98,
  "multi_model": true,
  "models": [
    {"wake_word": "曼波", "model_file": "dscnn_multiscale_manbo.onnx", "cons_frames": 3},
    {"wake_word": "你好电脑", "model_file": "dscnn_multiscale_nihaodiannao.onnx", "cons_frames": 3}
  ]
}
```

## 核心类

| 文件 | 说明 |
|------|------|
| `WakeWordEngine.java` | 模型加载、mel 特征提取、推理（支持 DS-CNN 和 NWW） |
| `DetectionLogic.java` | L1-L5 五层防误触发检测 |
| `AudioCapture.java` | 麦克风采集、锁-free 环形缓冲区 |
| `MainActivity.java` | Demo UI |

## L1-L5 检测层

| 层 | 默认 | 说明 |
|:---:|:---:|------|
| L1 | on / 3帧 | 连续帧过滤瞬态噪声（键盘、椅子） |
| L2 | off | 峰值/背景比，防静音幻觉 |
| L3 | off | 1.5s 冷却 |
| L4 | off | 爆发封锁（3次/3s → 5s） |
| L5 | off | 能量跳变（防视频/音乐误触发） |

L5 倍数滑块：2.0-5.0x，默认 3.0。

## 依赖

- `onnxruntime-android` (1.20+)
- Android 8.0+ (API 26+)

## 注意

`assets/` 下的 `.onnx` 和 `model_info.json` 仅本地测试用，**不要提交到 git**（已配 `.gitignore`）。
