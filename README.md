# Hermes Voice

在 macOS 上"喊一声小九就唤醒，然后语音对话"的轻量守护进程。

## 架构

```text
你说："小九，今天天气怎么样"
  │
  ▼
Silero VAD 检测到说话 → 缓冲音频
  │
  ▼
faster-whisper tiny 转写 → 匹配"小九"
  │
  ▼
去掉"小九"前缀 → Hermes API → TTS 朗读回复
  │
  ▼
进入 30 秒跟随时窗 —— 可直接说指令，无需再喊"小九"
```

**技术栈：** Python + sounddevice（录音）+ silero-vad（语音检测）+ faster-whisper（本地 STT）+ AVSpeechSynthesizer（macOS TTS）

## 快速开始

```bash
# 1. 克隆并安装
git clone <repo> hermes-voice
cd hermes-voice
bash scripts/install.sh

# 2. 启用 Hermes API Server（Hermes Gateway 的 HTTP API）
#    在 ~/.hermes/.env 中添加：
#      API_SERVER_ENABLED=true
#      API_SERVER_KEY=your-key-here
#    然后重启 gateway：
#      hermes gateway restart

# 3. 设置 config.yaml 中的 API Key（必须与第 2 步的 key 一致）
#    vim config.yaml → hermes_api_key: "your-key-here"

# 4. 启动
bash scripts/start.sh
```

首次启动会自动下载以下模型，之后离线运行：

| 模型 | 大小 | 下载源 | 缓存位置 |
|------|------|--------|----------|
| Whisper tiny（语音转文字） | ~150MB 下载 → 转写后 ~75MB | ModelScope `openai-mirror/whisper-tiny` | `~/.cache/hermes-voice/` |
| Silero VAD（语音检测） | ~2MB | ONNX 模型自动加载 | `~/.cache/silero-vad/` |

首次启动：① 从 ModelScope 下载 whisper 模型（~430MB）→ ② 自动转成 CTranslate2 格式（~75MB）→ ③ 后续启动秒开，无需再次下载。macOS 会弹出**麦克风权限**授权框，需要允许。

## 状态机

| 状态 | 说明 |
|------|------|
| **LISTENING** | 等待"小九"唤醒词。VAD + whisper tiny + 文本匹配 |
| **AWAKE** | 跟随时窗（30 秒）。说话直接提交 Hermes，无需唤醒词 |

唤醒后每次 TTS 回复完成重置 30 秒计时器。超时后自动回到 LISTENING。

## 配置

`config.yaml`：

```yaml
session_timeout: 30          # 跟随时窗（秒）
samplerate: 16000            # 音频采样率
silence_timeout: 1.5         # VAD 静音判定
model_size: tiny             # whisper 模型（tiny/base/small）
hermes_url: "http://localhost:8642"
hermes_api_key: "your-key-here"
```

## 依赖

- Python 3.12+（推荐 3.13）
- Homebrew（安装 portaudio）
- Hermes Gateway（运行中）
- macOS 13+（AVSpeechSynthesizer & silero-vad）

## 项目结构

```
.
├── config.yaml               # 配置文件
├── requirements.txt
├── scripts/
│   ├── install.sh            # 一键安装
│   └── start.sh              # 一键启动
└── voice/
    ├── main.py               # 主守护进程 + 状态机
    ├── recorder.py           # sounddevice + silero-vad 录音
    ├── wake_word.py          # whisper tiny + "小九"匹配
    ├── stt.py                # 文本条件处理
    ├── hermes_client.py      # Hermes API HTTP 客户端
    └── tts.py                # AVSpeechSynthesizer
```

## 局限性（v1）

- 无打断功能：TTS 朗读期间说话不会被识别
- 单线程：whisper 推理 / API 等待 / TTS 播放期间不响应新语音
- 唤醒延迟约 1-3 秒（whisper tiny 推理耗时）
- 嘈杂环境可能误触发或漏触发
