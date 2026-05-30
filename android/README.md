# Android 唤醒词 Demo

完整 Android 项目，用 Android Studio 打开即可运行。

## 使用

1. **复制模型文件到 assets**

将以下文件放入 `app/src/main/assets/`：

```
assets/
├── melspectrogram.onnx      # 必须（项目已提供）
├── embedding_model.onnx     # 必须（项目已提供）
├── model_info.json          # 编辑唤醒词配置
└── your_model.onnx          # 你的模型
```

2. **编辑 model_info.json**

```json
{
  "wake_word": "你的唤醒词",
  "model_file": "your_model.onnx",
  "emb_frames": 16
}
```

3. **Android Studio 打开项目，运行**

## 项目结构

```
android/
├── app/
│   ├── build.gradle              # ONNX Runtime 依赖
│   └── src/main/
│       ├── AndroidManifest.xml
│       ├── assets/               # ← 模型放这里
│       ├── java/com/voicute/wakeword/
│       │   ├── WakeWordEngine.java  # 推理引擎
│       │   └── MainActivity.java    # 界面
│       └── res/layout/
│           └── activity_main.xml
├── build.gradle
└── settings.gradle
```

## 关键代码

`WakeWordEngine.java` 是推理核心，可以单独提取到其他项目：

```java
// 初始化
WakeWordEngine engine = new WakeWordEngine(context);

// 处理音频（16kHz, mono, PCM 16-bit，每 ~30ms 调用一次）
short[] chunk = getAudioFromMicrophone();
DetectionResult result = engine.process(chunk);

if (result != null && result.probability > 0.5) {
    Log.i("WakeWord", "检测到: " + result.wakeWord);
}
```
