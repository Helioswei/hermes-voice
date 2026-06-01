# Hermes Voice

macOS 上的轻量语音助手守护进程——喊一声"小九"就唤醒，然后语音对话。

## 特性

| 特性 | 说明 |
|------|------|
| **多唤醒词** | 可配置多个唤醒词，默认"小九""轩轩"。基于 Sherpa-ONNX KWS，<100ms 唤醒延迟 |
| **多引擎 STT** | 支持 SenseVoice（默认，ONNX 推理）和 faster-whisper（CTranslate2）两种语音识别引擎 |
| **AEC 回声消除** | 利用 macOS AVAudioEngine 原生 voice processing，TTS 朗读时麦克风可继续监听 |
| **TTS 打断（Barge-in）** | TTS 朗读期间说话会自动打断并响应新指令；短促噪音（咳嗽）不会误触发 |
| **跟随时窗** | 唤醒后进入 30 秒跟随时窗，可直接说指令无需重复唤醒词。每次 TTS 回复重置计时 |
| **多轮对话** | HermesClient 保持消息上下文，跟随时窗内对话连续性 |
| **声控模型管理** | 自动下载、缓存、转换模型（ModelScope / HuggingFace / 本地三种来源） |
| **Hermes Gateway 一键启动** | `start.sh` 自动配置并启动 Hermes API Server |

## 技术栈

```
你说: "小九" ──→ Sherpa-ONNX KWS 唤醒词引擎 (<100ms)
  │ beep
  ▼
你说: "今天天气怎么样" ──→ AVAudioEngine 录音 (AEC) + Silero VAD 语音检测
  │
  ▼
SenseVoice / faster-whisper STT ──→ Hermes API (LLM) ──→ AVSpeechSynthesizer TTS
  │
  ▼
30 秒跟随时窗（直接说下一句，无需唤醒词）
```

**组件：**

| 模块 | 作用 |
|------|------|
| `voice/wake_word_engine.py` | Sherpa-ONNX KWS 唤醒词检测 |
| `voice/av_recorder.py` | AVAudioEngine 录音 + Silero VAD 语音活动检测 + AEC + 打断检测 |
| `voice/stt_engine.py` | STT 引擎抽象层（工厂模式） |
| `voice/stt_sensevoice.py` | SenseVoice ONNX 语音识别（阿里达摩院） |
| `voice/stt_whisper.py` | faster-whisper CTranslate2 语音识别（OpenAI Whisper） |
| `voice/models.py` | 模型下载、缓存、格式转换（支持 ModelScope / HuggingFace / 本地） |
| `voice/hermes_client.py` | Hermes API Server HTTP 客户端（多轮对话上下文） |
| `voice/tts.py` | macOS AVSpeechSynthesizer 中文语音合成（支持打断回调） |
| `voice/main.py` | 守护进程入口 + 两状态状态机（LISTENING / AWAKE） |

## 架构

### 状态机

```
LISTENING ── [检测到唤醒词] ──→ AWAKE
    ▲                               │
    │                  [30秒无对话 / 超时]
    └───────────────────────────────┘
```

| 状态 | 说明 |
|------|------|
| **LISTENING** | KWS 持续监听唤醒词。检测到 → 提示音 → VAD 录音 → STT → API → TTS → 进入 AWAKE |
| **AWAKE** | 跟随时窗。说话直接 VAD → STT → API → TTS，无需唤醒词。每次回复重置 30s 倒计时 |

### 打断流程

TTS 播放期间，麦克风持续监听（AEC 消除回声）。检测到用户连续说话 >0.5s 触发打断：

1. TTS 立即停止
2. 等待 VAD 完成当前语句（用户说完 + 1.5s 静音确认）
3. 转录打断音频 → API 请求 → TTS 回复新结果
4. 打断回复中再打断 → 递归处理

短促噪音（咳嗽、清嗓子）不会触发打断——VAD 必须持续检测到语音活动才算。

## 快速开始

```bash
# 1. 安装依赖
bash scripts/install.sh

# 2. 启用 Hermes API Server
#    在 ~/.hermes/.env 中：
#      API_SERVER_ENABLED=true
#      API_SERVER_KEY=hermes-voice-key

# 3. 启动（一键：自动启动 Gateway + 启动助手）
bash scripts/start.sh
```

首次启动自动下载模型到 `models/` 目录。

## 配置

详见 `config.yaml`，主要选项：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `stt_engine` | `sensevoice` | 语音引擎：`sensevoice` / `whisper` |
| `stt_model_size` | `small` | Whisper 模型规格（whisper 引擎） |
| `kws_keywords` | `["小九"]` | 唤醒词列表（可多个） |
| `kws_threshold` | `0.25` | 唤醒灵敏度（0.1–0.9） |
| `bargein_threshold` | `0.55` | TTS 期间 VAD 阈值（越高越严格） |
| `bargein_duration` | `0.5` | 打断确认时长（秒，咳嗽 <500ms 被过滤） |
| `session_timeout` | `30` | AWAKE 跟随时窗超时（秒） |
| `silence_timeout` | `1.5` | VAD 静音判定（秒） |

## 日志标记

| 标记 | 含义 |
|------|------|
| `唤醒词 →` | KWS 检测到唤醒词 |
| `麦克风→` | 麦克风 → STT 转写结果 |
| `打断→` | TTS 期间用户说话 → 打断转录 |
| `API →` | Hermes 服务端返回的文字 |
| `朗读 →` | TTS 正在朗读 |
| `状态机 →` | 状态切换 |
| `TTS 被用户打断` | TTS 被用户语音打断 |

## 依赖

- Python 3.12+
- Homebrew（安装 portaudio）
- Hermes Gateway（运行中，`start.sh` 自动管理）
- macOS 13+（AVAudioEngine voice processing / AVSpeechSynthesizer）

## 项目结构

```
.
├── config.yaml             # 配置文件
├── requirements.txt
├── models/                 # 模型文件（自动下载缓存）
│   ├── kws/                # Sherpa-ONNX 唤醒模型
│   ├── sensevoice/         # SenseVoice ONNX 模型
│   └── whisper-*/          # Whisper CTranslate2 模型（whisper 引擎）
├── scripts/
│   ├── install.sh          # 安装依赖
│   └── start.sh            # 一键启动（自动管理 Hermes Gateway）
├── docs/
│   ├── design.md
│   └── sherpa-onnx-kws-design.md
└── voice/
    ├── main.py             # 守护进程 + 状态机
    ├── av_recorder.py      # AVAudioEngine 录音 + VAD + 打断检测
    ├── wake_word_engine.py # Sherpa-ONNX KWS 唤醒
    ├── stt_engine.py       # STT 引擎抽象层
    ├── stt_sensevoice.py   # SenseVoice 语音识别
    ├── stt_whisper.py      # faster-whisper 语音识别
    ├── models.py           # 模型管理（下载/缓存/转换）
    ├── hermes_client.py    # Hermes API 客户端
    └── tts.py              # macOS 语音合成
```
