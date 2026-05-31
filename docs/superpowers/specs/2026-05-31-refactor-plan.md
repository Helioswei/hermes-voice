# Hermes Voice — Phase 1 重构计划

> 状态：📋 计划
> 创建日期：2026-05-31
> 涉及：死代码清理 / 重复代码复用 / TTS Barge-in / 容错增强

---

## 概述

本次 Phase 1 聚焦 4 项工作，按依赖顺序排列：

1. **死代码清理** — 删除不再使用的旧模块（无依赖）
2. **重复代码复用** — 抽离模型下载逻辑到共享模块（依赖 #1 完成后的清理）
3. **TTS Barge-in** — 实现打断能力（依赖 #2，但核心改动独立）
4. **容错增强** — 启动时检查麦克风权限、模型完整性、配置有效性

---

## 1. 死代码清理

### 删除的文件

| 文件 | 原因 | 风险 |
|------|------|------|
| `voice/recorder.py` | 被 `av_recorder.py` 取代，`main.py` 已使用 `AVRecorder` | ✅ `git grep` 确认零引用 |
| `voice/stt.py` | 仅有 deprecation 注释，`main.py` 使用 `stt_engine.py` 工厂 | ✅ 零引用 |
| `voice/wake_word.py` | 旧 `WakeWordDetector` (whisper包装)，被 `WakeWordEngine` (KWS) 取代 | ⚠️ 功能和 `stt_whisper.py` 有重叠 |

### 删除后验证

1. `git grep` 确认没有任何 `import` 引用了这三个模块
2. `python -m voice.main --help` 验证启动不报 ImportError
3. 删除后提交一个 cleanup commit

---

## 2. 重复代码复用

### 现状问题

删除 #1 的三个文件后，`stt_whisper.py` 中的模型下载/转换逻辑成为唯一的副本，但 `stt_sensevoice.py` 也有自己的模型下载逻辑。两者模式相同（首次运行自动下载 + 缓存到 `models/`）。

### 方案

创建 `voice/models.py` 共享模块，抽取所有模型管理逻辑：

```
voice/models.py          # 新建：模型下载/缓存/校验
  │
  ├── resolve_whisper_model()
  │    → 复用 stt_whisper.py 中 _resolve_whisper_model / _prepare_whisper_modelscope
  │
  ├── resolve_sensevoice_model()
  │    → 从 stt_sensevoice.py 移入，不改逻辑
  │
  ├── model_path(relative_path)
  │    → 统一计算 project_root / models/ 路径
  │
  └── _has_gpu()
       → 统一 GPU 检测
```

### 接口设计

```python
# voice/models.py

def resolve_whisper_model(
    model_size: str,
    source: str = "modelscope",
    path: str | None = None,
) -> str:
    """返回 CTranslate2 whisper 模型目录路径。
    
    - source="local" → 直接使用 path 路径
    - source="modelscope" → 从 ModelScope 下载并转换为 CTranslate2 格式
    - source="huggingface" → 让 faster-whisper 自动处理
    """

def resolve_sensevoice_model() -> tuple[str, str]:
    """返回 (model.onnx 路径, tokens.txt 路径)。
    
    从 HF mirror 自动下载并缓存到 models/sensevoice/。
    """

def _model_dir(subdir: str) -> str:
    """返回 models/<subdir>/ 的绝对路径。"""
```

### 修改文件

| 文件 | 变更 |
|------|------|
| `voice/models.py` | 新建 |
| `voice/stt_whisper.py` | 删除 `_resolve_whisper_model`、`_prepare_whisper_modelscope`、`_has_gpu`，改为 `from .models import resolve_whisper_model` |
| `voice/stt_sensevoice.py` | 删除 `_resolve_sensevoice_model`，改为 `from .models import resolve_sensevoice_model` |

---

## 3. TTS Barge-in

### 设计

三层防御实现可靠的用户打断：

```
麦克风信号
    │
    ▼
① AVAudioEngine AEC (硬件层)
   → 消除 TTS 扬声器回声主体
    │
    ▼
② VAD 阈值动态提高 (软件层)
   → TTS 期间 VAD 阈值从 0.5 -> configurable (默认 0.65)
   → 减少 AEC 残余回声的误触发
    │
    ▼
③ 语音持续 300ms 确认 (时间门)
   → 连续 VAD 活跃超过 300ms 才触发打断
   → 排除爆音/噪音瞬态误报
    │
    ▼
✅ 确认是用户开口 → 停止 TTS → 消费 VAD 缓冲音频 → 走正常流程
```

### 改动文件

#### `voice/tts.py` — `speak()` 加中断检查

```python
def speak(self, text, interrupt_check=None):
    """Speak text.
    
    Args:
        text: 要说的话
        interrupt_check: 可选回调，每次循环调用，返回 True 则停止 TTS
    
    Returns:
        True if completed naturally, False if interrupted.
    """
    ...
    self.synthesizer.speakUtterance_(utterance)
    delegate.finished = False
    
    while not delegate.finished:
        if interrupt_check and interrupt_check():
            self.stop()
            return False  # 用户打断
        NSRunLoop.currentRunLoop().runUntilDate_(
            NSDate.dateWithTimeIntervalSinceNow_(0.1)
        )
    return True  # 正常读完
```

#### `voice/av_recorder.py` — 增加打断检测和音频消费

改动点：

1. **TTS 期间标志**：加一个 `_tts_active` 标志，TTS 开始时置 True，结束后置 False
2. **动态 VAD 阈值**：`_tts_active` 时使用 `_tts_vad_threshold` (configurable, 默认 0.65) 代替 0.5
3. **打断检测**：加一个 `check_interrupt()` 公共方法，判断当前 VAD 检测是否超过持续时长

```python
# 新增方法
def check_interrupt(self, min_duration=0.3):
    """Check if user has been speaking continuously for min_duration.
    
    Returns:
        True if user is confirmed speaking (exceeds duration).
    """
    with self._lock:
        if self._speaking and self._speech_start_time is not None:
            elapsed = time.monotonic() - self._speech_start_time
            return elapsed >= min_duration
        return False

# 改动 _on_tap 中的部分
def _on_tap(self, buf, when):
    ...
    # ── VAD
    while len(self._vad_buffer) >= 512:
        ...
        if prob > threshold:  # threshold 变动态：0.65 if tts_active else 0.5
            if not self._speaking:
                self._speech_start_time = now  # 新增：记录说话起始时间
                ...
```

4. **`consume_utterance()`**：原子取走 `_utterance_result` 中的音频

```python
def consume_utterance(self):
    """原子取走 VAD 完整的语音数据，不清空 VAD 状态。"""
    with self._lock:
        if self._utterance_result is not None:
            audio = self._utterance_result
            self._utterance_result = None
            self._utterance_ready.clear()
            return audio
        return None
```

#### `voice/main.py` — `_speak_and_recover()` 改为两路等待

```python
def _speak_and_recover(recorder, tts, text):
    """Speak TTS reply. Returns audio bytes if interrupted, None otherwise."""

    # 1. 通知 AVRecorder 进入 TTS 模式（提高 VAD 阈值）
    recorder.set_tts_active(True)

    # 2. 定义打断检测回调
    def check_barge_in():
        return recorder.check_interrupt(
            min_duration=config.get("bargein_duration", 0.3)
        )

    # 3. 开始 TTS，monitor 是否被打断
    completed = tts.speak(text, interrupt_check=check_barge_in)

    # 4. 退出 TTS 模式
    recorder.set_tts_active(False)

    if not completed:
        logger.info("TTS 被用户打断")
        # 等 VAD 完成当前语句的缓冲
        audio = recorder.read_utterance(idle_timeout=3.0)  
        # ↑ 这里最多等 3 秒用户说完
        #   或用户已经说完了，立即返回
        return audio

    return None  # TTS 正常播完
```

### 打断后的通用处理

两个状态共用同一个打断后处理逻辑：

```python
def _process_interruption(recorder, stt, hermes, tts, interrupted_audio):
    """处理 TTS 打断后的新语音（被用户打断时调用的后续流程）。"""
    if interrupted_audio is None:
        return
    # 等 VAD 完成当前语句缓冲（如果还没完成）
    audio = recorder.read_utterance(idle_timeout=3.0)
    if audio is None:
        return

    text = _to_simplified(stt.transcribe(audio))
    if not text or not text.strip():
        return

    logger.info("打断→ %s", text)
    reply = hermes.send(text)
    logger.info("API → %s", reply)
    _speak_and_recover(recorder, tts, reply)
```

### AWAKE 状态中的使用

```python
elif state == "AWAKE":
    audio = recorder.read_utterance(idle_timeout=session_timeout)
    if audio is None:
        hermes.clear_context()
        state = "LISTENING"
        continue

    text = _to_simplified(stt.transcribe(audio))
    if not text or not text.strip():
        continue

    logger.info("麦克风→ %s", text)
    reply = hermes.send(text)
    logger.info("API → %s", reply)

    interrupted = _speak_and_recover(recorder, tts, reply)
    _process_interruption(recorder, stt, hermes, tts, interrupted)
    # → 回到 read_utterance 继续监听
```

### LISTENING 状态中的使用

```python
if state == "LISTENING":
    ...
    reply = hermes.send(text)
    logger.info("API → %s", reply)

    interrupted = _speak_and_recover(recorder, tts, reply)
    state = "AWAKE"
    recorder.start()
    logger.info("进入跟随时窗")

    # 即使是唤醒后的第一句回复被打断，也处理打断语音
    _process_interruption(recorder, stt, hermes, tts, interrupted)
```

### 配置项

```yaml
# config.yaml 新增
bargein_threshold: 0.65      # TTS 期间 VAD 阈值 (0.0-1.0，越高越严格)
bargein_duration: 0.3        # 打断确认时长（秒）
```

---

## 4. 容错增强

### 4.1 麦克风权限检查

在 `main()` 启动时、`AVRecorder.start()` 之前检查 macOS 麦克风权限。

```python
def check_mic_permission():
    """Check macOS microphone permission. Request if not determined."""
    # 使用 AVFoundation.AVMediaTypeAudio 检查授权状态
    # AVAuthorizationStatusNotDetermined → 请求权限
    # AVAuthorizationStatusDenied → 打印错误信息并退出
```

### 4.2 模型完整性校验

在创建 STT 引擎和 KWS 引擎时，检查模型文件是否存在且非空。

```python
# 在 voice/models.py 中
def validate_model_files(model_dir, required_files):
    """检查 required_files 列表中的文件是否存在且大小 > 0。"""
```

调用点：
- `WakeWordEngine.__init__()` → 检查 encoder/decoder/joiner ONNX 文件
- `WhisperSTT.__init__()` → 检查 model.bin
- `SenseVoiceSTT.__init__()` → 检查 model.onnx

### 4.3 配置验证

启动时验证 config.yaml 必要字段是否存在、值类型是否合法。

```python
def validate_config(cfg):
    """Validate required config keys and types."""
    required = {
        "samplerate": int,
        "silence_timeout": (int, float),
        "hermes_url": str,
    }
    for key, expected_type in required.items():
        if key not in cfg:
            raise ValueError(f"Missing required config: {key}")
```

### 4.4 异常恢复

涉及 #3 的改动，`_speak_and_recover()` 可能返回被打断的音频，需要确保打断流程中的异常也能回到稳定状态。在 `main.py` 中已有 `except Exception` 兜底（`state = "LISTENING"`），需要确认打断路径也在保护范围内。

---

## 执行顺序

```
Phase 1 ─────────────────────────────────────────────
│
├─ Step 1: 🗑️ 死代码清理
│   ├─ 删除 recorder.py, stt.py, wake_word.py
│   └─ Verify: git grep 零引用 + 启动测试
│
├─ Step 2: 🔁 代码复用
│   ├─ 新建 voice/models.py
│   ├─ 修改 stt_whisper.py → 导入 models
│   ├─ 修改 stt_sensevoice.py → 导入 models
│   └─ Verify: Whisper + SenseVoice STT 正常
│
├─ Step 3: 💬 TTS Barge-in
│   ├─ 改 voice/tts.py — speak() 加 interrupt_check
│   ├─ 改 voice/av_recorder.py — 动态 VAD 阈值 + check_interrupt() + consume_utterance()
│   ├─ 改 voice/main.py — _speak_and_recover() 两路等待
│   └─ Verify: TTS 期间说话能打断，正常流程不受影响
│
└─ Step 4: 🛡️ 容错
    ├─ 加麦克风权限检查
    ├─ 加模型文件校验
    ├─ 加配置验证
    └─ Verify: 缺失权限/模型/配置时友好报错
```

---

## 配置变更汇总

```yaml
# config.yaml 新增字段
bargein_threshold: 0.65      # TTS 期间 VAD 阈值（默认 0.65，0.5-0.9 可调）
bargein_duration: 0.3        # 打断确认时长秒数（默认 0.3，范围 0.1-1.0）
```

## 不完成的边界

- ❌ 不引入测试框架（当前无测试，后续单独 phase）
- ❌ 不改动 HermesClient 逻辑
- ❌ 不改变 config.yaml 现有字段名称（仅新增）
- ❌ 不引入第三方打断检测库（完全依赖已有 VAD + AEC）
