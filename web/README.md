# Web 唤醒词演示

浏览器本地运行 ONNX 唤醒词模型。不联网，不上传音频。

## 启动

| 系统 | 操作 |
|------|------|
| Windows | 双击 `run.bat` |
| Mac / Linux | 终端运行 `bash run.sh` |

然后浏览器打开 `http://localhost:8080/web/`。

## 安装

只需要 **Python 3.6+**，无需安装任何第三方包（使用内置 `http.server` 模块）。

```bash
# 确认 Python 已安装
python --version
# Python 3.10.0   ← 输出类似即可

# 如果未安装，从官网下载:
# https://www.python.org/downloads/
```

无需 `pip install` 任何东西，无需 Node.js。

## 文件说明

| 文件 | 说明 |
|------|------|
| `index.html` | 演示页面 UI |
| `wakeword.js` | 推理引擎（mel + embedding + softmax） |
| `run.bat` | Windows 一键启动 |
| `run.sh` | Mac/Linux 一键启动 |
