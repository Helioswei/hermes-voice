# Hermes Voice

macOS 上的轻量语音助手守护进程——喊一声"小九"就唤醒，然后语音对话。

## 架构

```
你说: "小九"  ──→  Sherpa-ONNX KWS 唤醒 (<100ms)
  │ beep
  ▼
你说: "今天天气怎么样"  ──→  VAD 录音 → faster-whisper tiny STT
  │
  ▼
Hermes API (/v1/chat/completions)  ──→  macOS TTS 朗读回复
  │
  ▼
30 秒跟随时窗（可直接说指令，无需再喊"小九"）
```

**技术栈：**
- **唤醒词：** Sherpa-ONNX KWS（Zipformer 3.3M，<100ms 延迟）
- **录音：** AVAudioEngine + voice processing（原生 AEC 回声消除）
- **VAD：** Silero VAD（语音活动检测）
- **STT：** faster-whisper tiny（本地语音转文字）
- **TTS：** AVSpeechSynthesizer（macOS 原生中文语音）
- **LLM：** Hermes API Server（`/v1/chat/completions`）

**关键设计：**
- AEC 回声消除：TTS 朗读时麦克风继续监听，原生 voice processing 从信号层消除回声
- 唤醒与指令分离：KWS 专做唤醒（高精度低延迟），whisper 专做指令转写

## 快速开始

```bash
# 1. 安装
bash scripts/install.sh

# 2. 启用 Hermes API Server
#    在 ~/.hermes/.env 中添加：
#      API_SERVER_ENABLED=true
#      API_SERVER_KEY=hermes-voice-key

# 3. 启动
bash scripts/start.sh
```

首次启动自动下载模型到 `models/` 目录：

| 模型 | 大小 | 用途 |
|------|------|------|
| Sherpa-ONNX KWS | ~17MB | 唤醒词检测 |
| Whisper tiny | ~148MB | 语音转文字 |

## 状态机

| 状态 | 说明 |
|------|------|
| **LISTENING** | KWS 持续监听"小九"。检测到 → 提示音 → VAD 录音 → STT → API → TTS |
| **AWAKE** | 跟随时窗（30 秒）。说话直接 VAD 录音 → STT → API，无需唤醒词。每次 TTS 回复重置计时器 |

## 配置

`config.yaml`：

```yaml
session_timeout: 30          # 跟随时窗（秒）
samplerate: 16000            # 音频采样率
silence_timeout: 1.5         # VAD 静音判定
max_record_sec: 15           # 单次录音上限
model_size: tiny             # whisper 模型（tiny/base/small）
model_source: modelscope     # huggingface / modelscope / local
kws_keywords:                # 自定义唤醒词（可多个）
  - "小九"
kws_threshold: 0.25          # 唤醒阈值（0.1-0.9，越高越严格）
hermes_url: "http://localhost:8642"
hermes_api_key: "hermes-voice-key"
```

## 日志说明

| 标记 | 含义 |
|------|------|
| `唤醒词 →` | KWS 检测到你说唤醒词 |
| `麦克风→` | 麦克风拾音 → whisper 转写结果 |
| `API →` | Hermes 服务端返回的文字 |
| `朗读 →` | TTS 正在朗读的文字 |
| `状态机 →` | 状态切换（就绪/跟随时窗/待唤醒） |

## 依赖

- Python 3.12+（推荐 3.13）
- Homebrew（安装 portaudio）
- Hermes Gateway（运行中）
- macOS 13+（AVAudioEngine voice processing）

## 项目结构

```
.
├── config.yaml
├── requirements.txt
├── models/                    # 模型文件（自动下载）
│   ├── kws/                   # Sherpa-ONNX 唤醒模型
│   └── whisper/               # faster-whisper STT 模型
├── scripts/
│   ├── install.sh
│   └── start.sh
├── docs/
│   ├── design.md              # 初始设计
│   └── sherpa-onnx-kws-design.md  # KWS 方案设计
└── voice/
    ├── main.py                # 主守护进程 + 状态机
    ├── av_recorder.py         # AVAudioEngine 录音（AEC 回声消除）
    ├── wake_word_engine.py    # Sherpa-ONNX KWS 唤醒词引擎
    ├── wake_word.py           # faster-whisper STT
    ├── hermes_client.py       # Hermes API HTTP 客户端
    └── tts.py                 # macOS 语音合成
```

## 架构演进

| 版本 | 唤醒方案 | 优势 |
|------|----------|------|
| v1 | VAD + whisper tiny + 文本匹配 | 纯本地 |
| v2（当前） | Sherpa-ONNX KWS + AVAudioEngine AEC | 唤醒 <100ms，回声消除 |
