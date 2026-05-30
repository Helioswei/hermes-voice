# Hermes Voice — 设计文档

> 状态：✅ 已决策（2026-05-30）
> 创建日期：2026-05-30

---

## 1. 概述

在 macOS 上实现"喊一声小九就唤醒，然后语音对话"的体验。

**一句话：** 一个轻量守护进程监听麦克风，检测到唤醒词"小九"后录音 → STT → POST 到 Hermes API Server → TTS 朗读回复。

---

## 2. 架构

```
┌────────────────────────────────────────────────┐
│                  macOS                          │
│                                                  │
│  ┌──────────────────────┐                       │
│  │  小九语音守护进程     │                       │
│  │  (hermes_voice.py)   │                       │
│  │                      │                       │
│  │  ① 监听麦克风        │                       │
│  │  ② 检测"小九"唤醒词  │                       │
│  │  ③ 录音（VAD）       │                       │
│  │  ④ STT → 文字       │                       │
│  │  ⑤ POST → Hermes API │──── HTTP ──┐         │
│  │  ⑥ TTS 朗读回复      │            │         │
│  └──────────────────────┘            │         │
│                                       │         │
│  ┌──────────────────────┐            │         │
│  │  Hermes Gateway      │◄───────────┘         │
│  │  (hermes gateway)    │                       │
│  │  localhost:8642      │                       │
│  │  /v1/chat/completions│                       │
│  └──────────────────────┘                       │
│                                                  │
└────────────────────────────────────────────────┘
```

### 关键设计原则

- **零修改 Hermes 源码** — 所有 Hermes 能力通过 API Server 复用
- **轻量守护进程** — 单 Python 文件，无外部服务依赖
- **本地优先** — 唤醒词检测和 STT 全部本地运行，不依赖云服务
- **跟随时窗** — 唤醒后 30 秒内可直接说指令，无需重复"小九"
- **可打断（v2）** — 预留全双工打断能力，v1 不实现

---

## 3. 组件详解

### 3.1 唤醒词检测器

检测用户说"小九"后触发语音对话流程。

| 方案 | 描述 | 模型大小 | 精度 | 离线 | 费用 |
|---|---|---|---|---|---|
| **A: macOS SFSpeechRecognizer** | macOS 原生语音识别框架，通过 pyobjc 调用 | 系统内置 | 高（中文支持好） | ✅（macOS 12+ on-device） | 免费 |
| **B: openwakeword** | 开源唤醒词引擎，预训练模型全英文 | ~50MB | 中（不支持中文） | ✅ | 免费 |
| **C: Porcupine (Picovoice)** | 商业级唤醒词，有 Python 绑定 | ~2MB | 高 | ✅ | 免费内置词，自定义词需付费 |
| **D: VAD + whisper tiny 文本匹配** | Silero VAD + whisper tiny + "小九"文本匹配 | ~77MB | 高（中文精准） | ✅ | 免费 |

**决策：方案 D（VAD + whisper tiny 文本匹配）** ✅ — 纯本地、零 hack、中文精准。流程：

1. **Silero VAD**（~2MB）持续监听麦克风，检测到有人说话
2. 缓冲整段语音到内存
3. **faster-whisper tiny**（~75MB）将这段语音转文字
4. 文本匹配是否包含"小九"（精确匹配，0% 误触发）
5. 播放提示音（通过 `afplay` 系统命令或 `AVSpeechSynthesizer` 播放短促音效）
6. 是 → **直接复用唤醒词转写结果**，去掉"小九"前缀后走 STT 流程（无需再跑 whisper 推理）
7. 否 → 丢弃缓冲，继续 VAD 监听

> 下载量：~77MB（VAD 2MB + whisper tiny 75MB），纯本地，不依赖云端。

#### 状态机与跟随时窗

检测到"小九"后，守护进程进入 **AWAKE（唤醒）** 状态，开启 30 秒跟随时窗：

| 状态 | 触发条件 | 行为 |
|------|---------|------|
| **LISTENING** | 启动 / 跟随时窗超时 | VAD → whisper tiny → 匹配"小九" |
| **AWAKE** | 匹配到"小九" | VAD → whisper tiny → **直接提交** Hermes API（不过滤关键词） |

- 每次 TTS 回复完成后重置 30 秒计时器
- AWAKE 状态下再说"小九" → 仅延长超时，不重复唤醒
- 30 秒内无语音活动 → 回到 LISTENING

### 3.2 录音器

管理音频缓冲区，VAD 检测到说话时开始缓冲，检测到静音后停止并交付音频。

- 使用 `sounddevice`（Hermes 官方 voice mode 也在用）
- 采样率 16kHz，单声道，16-bit PCM
- VAD（语音活动检测）使用 `silero-vad` 判断说话结束（与唤醒词检测器复用同一模型）
- 超时：15 秒无语音自动结束
- 在 v1 中，唤醒词检测和跟随时窗共用同一套录音缓冲，不额外开录音会话

### 3.3 STT（语音转文字）

| 方案 | 模型大小 | 速度 | 中文精度 | 说明 |
|---|---|---|---|---|
| **faster-whisper base** | ~150MB | 快（int8） | 中 | Hermes 官方在用，首次自动下载 |
| **faster-whisper small** | ~500MB | 中 | 高 | 中文效果更好 |
| **macOS 听写 API** | 内置 | 很快 | 高 | 需联网，系统 API 调用 |
| **Groq Whisper API** | 云端 | 极快 | 高 | 需 GROQ_API_KEY |

**决策：复用唤醒词检测器的 faster-whisper tiny 转写结果（v1）** ✅ — v1 唤醒词和 STT 共用一次 whisper tiny 推理，零额外延迟。根据状态决定文本处理方式：

| 状态 | 用户说话 | 处理方式 |
|------|---------|---------|
| LISTENING | "小九今天天气怎么样" | 去掉"小九"前缀，提交"今天天气怎么样" |
| AWAKE | "帮忙查PM2.5" | 整段提交，不过滤关键词 |

后续如需更高精度，可升级到 small（~500MB）对同一段缓冲音频做二次转写。

### 3.4 Hermes API 客户端

把 STT 得到的文字 POST 到 Hermes API Server。

```python
POST http://localhost:8642/v1/chat/completions
Authorization: Bearer {API_SERVER_KEY}

{
  "model": "hermes-agent",
  "messages": [
    {"role": "user", "content": "今天天气怎么样"}
  ],
  "stream": false
}
```

- 非流式返回，等完整回复后再 TTS
- 持有多轮对话上下文（通过 messages 数组维持 session）

### 3.5 TTS（文字转语音）

| 方案 | 音质 | 延迟 | 离线 | 说明 |
|---|---|---|---|---|
| **macOS `say` 命令** | 中（系统默认语音） | 极低 | ✅ | 零依赖，支持中文（Ting-Ting） |
| **Edge TTS** | 好 | 低 | ❌ | Hermes 官方支持，免费 |
| **macOS AVSpeechSynthesizer** | 好（增强语音） | 极低 | ✅ | Python pyobjc 调用，系统内置 |
| **NeutTS** | 好 | 中 | ✅ | 本地神经 TTS，需下载模型 |

**决策：macOS AVSpeechSynthesizer** ✅ — 系统内置、零下载、中文支持好、音质优于 `say`。

---

## 4. 目录结构

```
/Users/helios/AIWork/hermes-voice/
├── README.md                  # 项目说明
├── docs/
│   └── design.md              # 本设计文档
├── voice/
│   ├── __init__.py
│   ├── main.py                # 主入口（守护进程）
│   ├── wake_word.py           # 唤醒词检测
│   ├── recorder.py            # 录音 + VAD
│   ├── stt.py                 # 语音转文字（faster-whisper）
│   ├── hermes_client.py       # Hermes API Server 客户端
│   └── tts.py                 # 文字转语音
├── scripts/
│   ├── install.sh             # 一键安装脚本
│   └── start.sh               # 启动脚本
├── config.yaml                # 配置文件
└── requirements.txt           # Python 依赖
```

---

## 5. 用户交互流程

```
─── 第一轮：唤醒 ───

1️⃣  用户说："小九，今天天气怎么样"
                      │
2️⃣  Silero VAD 检测到说话 → 缓冲整段音频
                      │
3️⃣  用户说话完毕（VAD 静音检测 1.5s）
                      │
4️⃣  whisper tiny 转写 → 匹配"小九" → 进入 AWAKE 状态
                      │
5️⃣  💡 提示音 → 去掉"小九"前缀 → "今天天气怎么样"
                      │
6️⃣  POST → Hermes API Server（/v1/chat/completions）
                      │
7️⃣  Hermes 处理 → 回复"今天晴，25度"
                      │
8️⃣  AVSpeechSynthesizer 朗读回复
                      │
9️⃣  重置 30 秒跟随时窗，VAD 继续监听
                      │
─── 第二轮：跟随时窗内（无需"小九"）───
                      │
1️⃣0️⃣ 用户说："帮忙查PM2.5"
                      │
1️⃣1️⃣  Silero VAD 检测到说话 → 缓冲整段音频
                      │
1️⃣2️⃣  whisper tiny 转写 → AWAKE 状态 → 直接提交
                      │
1️⃣3️⃣  POST（带上对话上下文）→ Hermes 回复 PM2.5
                      │
1️⃣4️⃣  TTS 朗读 → 重置跟随时窗
                      │
─── 跟随时窗超时 ───
                      │
1️⃣5️⃣ ⏰ 30 秒无对话 → 回到 LISTENING，等待下一次"小九"
```

> 注：AWAKE 状态期间 TTS 朗读时用户说话不会被处理（打断功能为 v2 特性，v1 不做）。跟随时窗仅指安静等待期间的连续对话。

---

## 6. 依赖与安装

### Python 依赖

```
silero-vad                          # 语音活动检测（Silero VAD）
sounddevice                         # 音频录制
faster-whisper                      # STT（自动下载 ~75MB tiny 模型）
pyobjc-framework-AVFoundation       # AVSpeechSynthesizer TTS 调用
numpy                               # 音频数据处理
pyyaml                              # 配置解析
```

### 系统依赖

- `portaudio` — `brew install portaudio`（sounddevice 需要）
- `ffmpeg` — `brew install ffmpeg`（音频格式转换，可选）

### 与 Hermes 配合

- 需要 Hermes Gateway 运行中（`hermes gateway`）
- 需要在 `~/.hermes/.env` 中配置 `API_SERVER_ENABLED=true` 和 `API_SERVER_KEY`
- 需要 Hermes 已安装语音依赖（可选，但我们自己处理 STT/TTS）

---

## 7. 启动方式

```bash
# 终端 1：启动 Hermes Gateway
hermes gateway

# 终端 2：启动 Hermes Voice
cd /Users/helios/AIWork/hermes-voice
python -m voice.main
```

或者用一个脚本封装：

```bash
scripts/start.sh    # 自动启动 gateway（若未运行）+ 守护进程
```

---

## 8. 决策记录

| 项目 | 决策 | 说明 |
|---|---|---|
| 唤醒词引擎 | VAD + whisper tiny 文本匹配 | Silero VAD（2MB）+ whisper tiny（75MB），文本精确匹配"小九" |
| STT 模型 | 复用唤醒词 faster-whisper tiny 转写结果（v1） | 零额外推理，后续可升级 small 做二次转写 |
| TTS | macOS AVSpeechSynthesizer | 内置零下载，`pyobjc` 调用 |
| 多轮对话 | ✅ 跟随时窗 30s | 唤醒后 30 秒内直接说话，无需重复"小九"，每次 TTS 后重置 |
| 打断功能 | ❌ v1 不做 | v2 升级特性，AWAKE 期间 TTS 播放中不响应 |
| 热键备用 | ✅ 保留 | 按键 + 唤醒词双触发 |
| 与官方语音模式 | 共存 | 代码共存，但运行时不同时占麦克风 |
| 启动方式 | 一键脚本 | 同时启动 gateway + 守护进程 |

---

## 9. 局限性（v1 已知问题）

- 唤醒词检测在嘈杂环境中可能误触发（Silero VAD 对噪音敏感）或漏触发（whisper tiny 在背景噪音下转写精度下降）
- 无打断功能：TTS 朗读期间用户说话不会被识别（v2 修复；跟随时窗仅适用于安静等待期间）
- 守护进程单线程，whisper 推理 / Hermes API 等待 / TTS 播放期间不会响应新语音
- 守护进程和 Hermes gateway 是两个独立进程，需一键脚本管理
- 首次启动需下载 whisper tiny ~75MB + Silero VAD ~2MB = ~77MB（几秒钟）
- VAD+whisper 方案唤醒延迟约 1-3 秒（whisper tiny 推理耗时，VAD 本身约 200ms），比专用唤醒词引擎慢
