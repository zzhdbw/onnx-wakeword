# Linux / macOS / Windows 命令行推理

Python 脚本，麦克风实时唤醒词检测。**支持 x86_64 和 32-bit ARM（树莓派）。**

## 架构选择

| 平台 | 推理脚本 | 推理引擎 | 安装 |
|------|----------|----------|------|
| **x86_64 / 64-bit ARM** (Pi 4/5 64位) | `infer.py` | ONNX Runtime | `pip install onnxruntime` |
| **32-bit ARM** (Pi 3B / Zero / Zero 2W) | `infer_tflite.py` | TFLite Runtime | `pip install tflite-runtime` |

> **为什么 32 位用 TFLite？** ONNX Runtime 不提供 32 位 ARM (armv7l) 预编译包，源码编译复杂。TFLite Runtime 有官方 `linux_armv7l` wheel，一等公民支持。

## 64 位系统

```bash
pip install onnxruntime numpy sounddevice
python linux/infer.py --model-dir ./models/
```

## 32 位树莓派

### 1. 系统依赖

```bash
sudo apt install libportaudio2
```

### 2. Python 依赖

```bash
pip install tflite-runtime numpy sounddevice
```

### 3. 转换模型（在 x86_64 机器上做，树莓派编译 onnx2tf 困难）

```bash
pip install onnx2tf
python linux/convert_to_tflite.py --model-dir ./models/
```

然后将 `models/` 目录下的 `.tflite` 文件复制到树莓派。

### 4. 运行

```bash
python linux/infer_tflite.py --model-dir ./models/
```

## 参数

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `--model-dir` | `../models/` | 模型目录路径 |
| `--threshold` | `0.5` | 触发阈值 (0~1，softmax 概率) |
| `--cooldown` | `3.5` | 冷却时间（秒） |
| `--list-devices` | — | 列出可用麦克风设备 |
