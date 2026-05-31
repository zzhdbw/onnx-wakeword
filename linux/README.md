# Linux / macOS / Windows 命令行推理

Python 脚本，麦克风实时唤醒词检测。

## 安装

需要 **Python 3.8+**。

```bash
pip install onnxruntime numpy sounddevice
```

## 使用

1. 将模型文件放入 `models/` 目录（或自定义路径）
2. 运行：

```bash
python linux/infer.py --model-dir ./models/
```

| 参数 | 默认值 | 说明 |
|------|:---:|------|
| `--model-dir` | `../models/` | 模型目录路径 |
| `--threshold` | `0.5` | 触发阈值 (0~1) |
| `--cooldown` | `3.5` | 冷却时间（秒） |
| `--list-devices` | — | 列出可用麦克风设备 |
