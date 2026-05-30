# Hermes Voice — 唤醒词方案设计

> 状态：✅ 已实现
> 创建日期：2026-05-30
> 更新日期：2026-05-30

## 1. 目标

替换原有的 VAD + whisper tiny + 文本匹配 唤醒链路，用 Sherpa-ONNX Keyword Spotting 专门做唤醒词检测，同时使用 AVAudioEngine 原生 voice processing 做回声消除。

**已解决的问题：**

| 问题 | 解决方案 |
|------|----------|
| whisper 转写"嗨小九"为"来 下酒" | Sherpa-ONNX KWS 专做唤醒词，<100ms，不依赖 STT |
| 噪音被 VAD 误触发 → whisper 幻觉中文 | KWS 模型专为关键词检测训练，噪声下误触发极低 |
| "小酒"命中同音字 → 误唤醒 | 不再需要文本匹配，KWS 直接声学匹配 |
| TTS 回声被麦克风捕捉 → 二次触发 | AVAudioEngine voice processing 原生 AEC 消除回声 |
| 唤醒延迟 1-3 秒 | KWS <100ms 流式匹配 |

## 2. 架构

```
┌─ 唤醒阶段（Sherpa-ONNX KWS）──────────────────┐
│                                                  │
│  AVAudioEngine 音频流 (voice processing AEC)     │
│       │                                          │
│       ▼                                          │
│  ┌──────────────────────┐                        │
│  │  WakeWordEngine      │  每 512 样本送入 KWS   │
│  │  (sherpa-onnx KWS)   │  延迟 <100ms           │
│  └──────┬───────────────┘                        │
│         │ 检测到 "小九"                            │
│         ▼                                        │
│    提示音 → VAD 录音 → whisper STT               │
│                                                  │
└──────────────────────────────────────────────────┘
         │
         ▼
┌─ 回声消除（AVAudioEngine voice processing）─────┐
│                                                  │
│  TTS 扬声器输出                                   │
│       │                                          │
│       ▼  AEC 已知扬声器信号                       │
│  ┌──────────────────────┐                        │
│  │  麦克风信号 - TTS回声  │ = 干净人声             │
│  └──────────────────────┘                        │
│                                                  │
│  TTS 朗读期间用户可直接打断说话                     │
│                                                  │
└──────────────────────────────────────────────────┘
```

## 3. 组件实现

### 3.1 WakeWordEngine (`voice/wake_word_engine.py`)

封装 sherpa-onnx Keyword Spotter，使用 ppinyin 格式的关键词匹配。

- 模型存储在 `models/kws/`，首次运行从 ModelScope 自动下载
- 关键词通过 `pypinyin` 自动转换为 ppinyin token 格式
- 可配置多个唤醒词（如 `["小九", "嗨小九"]`）

**API：**
- `process_chunk(audio: np.ndarray) -> bool` — 实时检测
- `reset()` — 检测到后重置流

**配置：**
- `config.yaml` → `kws_keywords` — 唤醒词列表
- `config.yaml` → `kws_threshold` — 检测阈值（0.1-0.9）

### 3.2 AVRecorder (`voice/av_recorder.py`)

替换 sounddevice，使用 macOS 原生 AVAudioEngine。

- `setVoiceProcessingEnabled_(True)` 开启硬件 AEC
- 48kHz → 16kHz 实时降采样
- 内部 VAD 缓冲池（512 样本切块）
- API 与旧 AudioRecorder 兼容

**AEC 工作原理：**
```
扬声器播放 TTS → 麦克风同时开启
  → AVAudioEngine 持有扬声器信号副本
  → 实时从麦克风信号中减去 (TTS × 房间传递函数)
  → 输出的信号只包含用户声音
```

### 3.3 STT (`voice/wake_word.py`)

保留 faster-whisper tiny 做指令转写，移除了唤醒词检测相关代码。

- 模型存储在 `models/whisper/`
- `transcribe(audio)` — 语音转文字
- 参数优化：`beam_size=5, vad_filter=True`

### 3.4 移除的模块

- `voice/stt.py` — 状态相关的文本过滤逻辑（不再需要）
- `voice/recorder.py` — sounddevice 录音器（被 av_recorder 取代）
- `voice/wake_word.py` 中的 `contains_wake_word` / `strip_wake_word` / 同音字集合

## 4. 模型文件

```
models/
├── kws/                          # Sherpa-ONNX KWS (~17MB)
│   ├── encoder-*.onnx
│   ├── decoder-*.onnx
│   ├── joiner-*.onnx
│   └── tokens.txt
└── whisper/                      # faster-whisper STT (~148MB)
    ├── model.bin
    ├── config.json
    ├── tokenizer.json
    ├── preprocessor_config.json
    └── vocabulary.json
```

模型均从 ModelScope 自动下载，存储在 `models/` 下。

## 5. 配置

`config.yaml` 唤醒相关配置：

```yaml
kws_keywords:                 # 自定义唤醒词
  - "小九"
kws_threshold: 0.25           # 阈值
```

## 6. 运行流程

```
LISTENING:
  AVAudioEngine 音频流 (AEC 已启用)
    → WakeWordEngine 实时检测每 512 样本
    → 检出 "小九" → 提示音 → VAD 录音 → whisper STT
    → Hermes API → TTS 朗读（AEC 抵消回声）
    → 进入 AWAKE

AWAKE:
  TTS 朗读时麦克风持续监听（AEC 防止回声触发）
    → VAD 检测语音 → whisper STT → API → TTS
    → 30s 无语音 → 回到 LISTENING
```

## 7. 决策记录

| 项目 | 决策 | 说明 |
|------|------|------|
| 唤醒词引擎 | Sherpa-ONNX KWS | Zipformer 3.3M，<100ms，开源 Apache 2.0 |
| 回声消除 | AVAudioEngine voice processing | macOS 原生 AEC，信号层分离 |
| STT | faster-whisper tiny | 保留，只做指令转写 |
| TTS | macOS AVSpeechSynthesizer | 零下载，pyobjc 调用 |
| 模型存储 | `models/` 目录 | 首次自动下载，不提交 git |
| 唤醒词配置 | `config.yaml` → `kws_keywords` | 支持多个，自动转 ppinyin |
