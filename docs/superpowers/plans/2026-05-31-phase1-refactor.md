# Phase 1 重构 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up dead code, DRY model resolution, add TTS barge-in, and add error resilience startup checks.

**Architecture:** Four independent phases run sequentially: (1) delete unused files, (2) extract shared model utilities into `voice/models.py`, (3) add barge-in capability to TTS with 3-layer protection (AEC + dynamic VAD threshold + 300ms hold), (4) add startup validation for mic permission, model files, and config.

**Tech Stack:** Python 3.12+, macOS AVAudioEngine, sherpa-onnx, faster-whisper, AVSpeechSynthesizer

**Files to touch:**
| Action | File |
|--------|------|
| Delete | `voice/recorder.py` |
| Delete | `voice/stt.py` |
| Delete | `voice/wake_word.py` |
| Create | `voice/models.py` |
| Modify | `voice/stt_whisper.py` |
| Modify | `voice/stt_sensevoice.py` |
| Modify | `voice/tts.py` |
| Modify | `voice/av_recorder.py` |
| Modify | `voice/main.py` |
| Modify | `config.yaml` |

---

### Task 1: 死代码清理

**Files:**
- Delete: `voice/recorder.py`
- Delete: `voice/stt.py`
- Delete: `voice/wake_word.py`

- [ ] **Step 1: 删除三个已废弃的文件**

```bash
git rm voice/recorder.py voice/stt.py voice/wake_word.py
```

- [ ] **Step 2: 验证项目能正常启动**

```bash
source .venv/bin/activate
python -c "from voice.main import main; print('OK')"
```

Expected output:
```
OK
```

- [ ] **Step 3: 提交**

```bash
git add -A
git commit -m "chore: remove dead code (recorder.py, stt.py, wake_word.py)

These modules have been replaced:
- recorder.py → av_recorder.py (AVAudioEngine with AEC)
- stt.py → stt_engine.py + stt_whisper.py / stt_sensevoice.py
- wake_word.py → wake_word_engine.py (Sherpa-ONNX KWS)

Git grep confirmed zero imports to these files.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: 创建 voice/models.py — 统一模型管理

**Files:**
- Create: `voice/models.py`
- Modify: `voice/stt_whisper.py`
- Modify: `voice/stt_sensevoice.py`

- [ ] **Step 1: 创建 voice/models.py**

```python
"""Model management utilities.

Centralizes model resolution, download, and caching for all STT engines.
"""

import os
import shutil
import glob


def _project_root():
    """Return absolute path to project root (parent of voice/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def model_dir(subdir: str) -> str:
    """Return absolute path to ``models/<subdir>/``."""
    path = os.path.join(_project_root(), "models", subdir)
    os.makedirs(path, exist_ok=True)
    return path


def _has_gpu():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# ── Whisper ──────────────────────────────────────────────────────────

def resolve_whisper_model(model_size, source="modelscope", path=None):
    """Return CTranslate2 whisper model directory path.

    Parameters
    ----------
    model_size : str
        ``tiny`` / ``base`` / ``small`` / ``medium``.
    source : str
        ``modelscope`` (download + convert, default), ``huggingface`` (auto),
        or ``local`` (pre-downloaded CTranslate2 directory).
    path : str or None
        Model path when ``source="local"``, otherwise unused.

    Returns
    -------
    str
        Path to CTranslate2 model directory.
    """
    if source == "local":
        if not path or not os.path.isdir(path):
            raise FileNotFoundError(
                f"source=local but path not found: {path}"
            )
        return path

    if source == "modelscope":
        return _prepare_whisper_modelscope(model_size)

    # huggingface → let faster-whisper handle it automatically
    return model_size


def _prepare_whisper_modelscope(model_size, repo=None):
    """Download whisper from ModelScope, convert to CTranslate2, cache to models/."""
    from modelscope import snapshot_download

    if not repo:
        repo = f"openai-mirror/whisper-{model_size}"

    ct2_path = model_dir(f"whisper-{model_size}")

    if os.path.isfile(os.path.join(ct2_path, "model.bin")):
        return ct2_path

    raw_path = snapshot_download(repo, ignore_file_pattern=["*.msgpack", "*.h5"])

    from ctranslate2.converters import TransformersConverter

    if os.path.isdir(ct2_path):
        shutil.rmtree(ct2_path)

    print(f"Converting {repo} to CTranslate2 format …")
    converter = TransformersConverter(
        model_name_or_path=raw_path,
        copy_files=["tokenizer.json", "preprocessor_config.json"],
    )
    converter.convert(
        output_dir=ct2_path,
        quantization="float16" if _has_gpu() else "float32",
    )
    print(f"Converted model saved to {ct2_path}")
    return ct2_path


# ── SenseVoice ───────────────────────────────────────────────────────

def resolve_sensevoice_model():
    """Download SenseVoice ONNX model from HF mirror, cache to models/sensevoice/.

    Returns
    -------
    tuple[str, str]
        ``(model.onnx path, tokens.txt path)``
    """
    from urllib.request import urlretrieve

    mdir = model_dir("sensevoice")

    model_file = os.path.join(mdir, "model.onnx")
    tokens_file = os.path.join(mdir, "tokens.txt")

    if os.path.isfile(model_file) and os.path.isfile(tokens_file):
        return model_file, tokens_file

    base = (
        "https://hf-mirror.com/csukuangfj/"
        "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main"
    )

    print("Downloading SenseVoice ONNX model …")
    urlretrieve(f"{base}/model.onnx", model_file)
    urlretrieve(f"{base}/tokens.txt", tokens_file)

    print(f"SenseVoice model cached at {mdir}")
    return model_file, tokens_file
```

- [ ] **Step 2: 修改 stt_whisper.py — 从 models 导入**

```python
"""faster-whisper STT backend."""

from .models import resolve_whisper_model
from .stt_engine import STTEngine


class WhisperSTT(STTEngine):
    """faster-whisper based STT engine.

    Parameters
    ----------
    model_size : str
        ``tiny`` / ``base`` / ``small`` / ``medium``.
    model_source : str
        ``modelscope`` / ``huggingface`` / ``local``.
    model_path : str or None
        Path when ``model_source=local``, otherwise unused.
    """

    def __init__(self, model_size="small", model_source="modelscope",
                 model_path=None, device="cpu", compute_type="int8"):
        from faster_whisper import WhisperModel

        model_dir = resolve_whisper_model(model_size, model_source, model_path)
        self._model = WhisperModel(model_dir, device=device,
                                   compute_type=compute_type)

    def transcribe(self, audio):
        if audio is None or len(audio) == 0:
            return ""
        segments, _ = self._model.transcribe(
            audio, language="zh",
            beam_size=5, vad_filter=True,
        )
        return "".join(seg.text for seg in segments).strip()
```

- [ ] **Step 3: 修改 stt_sensevoice.py — 从 models 导入**

```python
"""SenseVoice STT backend via sherpa-onnx."""

from .models import resolve_sensevoice_model
from .stt_engine import STTEngine


class SenseVoiceSTT(STTEngine):
    """SenseVoice STT engine via sherpa-onnx.

    Downloads the ONNX model from HF mirror on first use.
    """

    def __init__(self, language="zh"):
        import sherpa_onnx

        model_path, tokens_path = resolve_sensevoice_model()
        self._model = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=2,
            sample_rate=16000,
            language=language,
            use_itn=True,
        )
        self._stream = self._model.create_stream()

    def transcribe(self, audio):
        if audio is None or len(audio) == 0:
            return ""
        self._stream.accept_waveform(16000, audio)
        self._model.decode_stream(self._stream)
        text = self._stream.result.text
        self._stream = self._model.create_stream()
        return text.strip()
```

- [ ] **Step 4: 验证导入正常**

```bash
source .venv/bin/activate
python -c "from voice.models import resolve_whisper_model, resolve_sensevoice_model, model_dir; print('models OK')"
python -c "from voice.stt_whisper import WhisperSTT; print('whisper STT OK')"
python -c "from voice.stt_sensevoice import SenseVoiceSTT; print('sensevoice STT OK')"
python -c "from voice.main import main; print('main OK')"
```

Expected: all four print "OK".

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "refactor: extract shared model management into voice/models.py

- voice/models.py: resolve_whisper_model(), resolve_sensevoice_model(),
  model_dir(), _has_gpu()
- stt_whisper.py: removed duplicated _resolve_whisper_model,
  _prepare_whisper_modelscope, _has_gpu → import from models
- stt_sensevoice.py: removed local _resolve_sensevoice_model → import from models

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: TTS Barge-in — 改 voice/tts.py

**Files:**
- Modify: `voice/tts.py`

- [ ] **Step 1: 修改 speak() 方法，支持 interrupt_check 回调**

替换 `voice/tts.py:33-52` 的 `speak()` 和 `stop()` 方法：

```python
    def speak(self, text, interrupt_check=None):
        """Speak *text* and block until done.

        Parameters
        ----------
        text : str
            Text to speak aloud.
        interrupt_check : callable or None
            Optional zero-arg callback called each iteration. If it returns
            True, speech is stopped immediately and the method returns False.

        Returns
        -------
        bool
            True if utterance completed naturally, False if interrupted.
        """
        if not text or not text.strip():
            return True

        utterance = AVFoundation.AVSpeechUtterance.speechUtteranceWithString_(
            text
        )
        voice = AVFoundation.AVSpeechSynthesisVoice.voiceWithLanguage_("zh-CN")
        utterance.setVoice_(voice)

        delegate = self.synthesizer.delegate()
        delegate.finished = False

        self.synthesizer.speakUtterance_(utterance)

        while not delegate.finished:
            if interrupt_check and interrupt_check():
                self.stop()
                return False
            NSRunLoop.currentRunLoop().runUntilDate_(
                NSDate.dateWithTimeIntervalSinceNow_(0.1)
            )
        return True

    def stop(self):
        self.synthesizer.stopSpeakingAtBoundary_(
            AVFoundation.AVSpeechBoundaryImmediate
        )
```

Note: `stop()` already exists at line 54-57 — replace the entire method body as shown above (no change needed, just confirming it's correct).

- [ ] **Step 2: 验证 tts 模块导入正常**

```bash
source .venv/bin/activate
python -c "from voice.tts import TTSEngine; print('tts OK')"
```

Expected: prints "tts OK".

- [ ] **Step 3: 提交**

```bash
git add -A
git commit -m "feat: add interrupt_check callback to TTSEngine.speak()

speak() now accepts an optional interrupt_check() callback called each
iteration of the run loop. If it returns True, speech stops immediately
and the method returns False (interrupted) instead of True (completed).

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: TTS Barge-in — 改 voice/av_recorder.py

**Files:**
- Modify: `voice/av_recorder.py`

- [ ] **Step 1: __init__ 加新增字段**

在 `voice/av_recorder.py:27`（`max_record_sec` 初始化行后）添加：

```python
        # Barge-in (TTS 打断)
        self._tts_active = False
        self._tts_vad_threshold = 0.65   # TTS 期间 VAD 阈值
        self._speech_start_time = None   # 当前语音起始时间（用于打断确认）
```

- [ ] **Step 2: 添加 set_tts_active(), check_interrupt(), consume_utterance() 方法**

在 `voice/av_recorder.py` 中 `drain_ring_buffer()` 方法（第148行）之后添加：

```python
    # ── Barge-in support ─────────────────────────────────

    def set_tts_active(self, active, threshold=0.65):
        """Set TTS playback state and VAD threshold during TTS.

        When active, VAD uses a higher threshold to reduce echo residuals
        from triggering false interrupts.
        """
        self._tts_active = active
        self._tts_vad_threshold = threshold
        if not active:
            with self._lock:
                self._speaking = False
                self._speech_start_time = None

    def check_interrupt(self, min_duration=0.3):
        """Check if user has been speaking continuously for *min_duration*.

        Returns True only when VAD has detected sustained speech exceeding
        the duration threshold — prevents transient noise from triggering
        an interrupt.
        """
        with self._lock:
            if self._speaking and self._speech_start_time is not None:
                elapsed = time.monotonic() - self._speech_start_time
                return elapsed >= min_duration
            return False

    def consume_utterance(self):
        """Atomically retrieve a completed utterance without clearing VAD state.

        Use this after an interrupt to grab the audio the user spoke during
        TTS playback.
        """
        with self._lock:
            if self._utterance_result is not None:
                audio = self._utterance_result
                self._utterance_result = None
                self._utterance_ready.clear()
                return audio
            return None
```

- [ ] **Step 3: 修改 _on_tap() 中的 VAD 阈值和语音起始时间**

在 `voice/av_recorder.py` 的 `_on_tap()` 方法中，改两处：

**3a.** 把硬编码 VAD 阈值 `0.5` 改为动态值（第242行）：

```python
                vad_threshold = self._tts_vad_threshold if self._tts_active else 0.5
                if prob > vad_threshold:
```

**3b.** 在 `self._speaking = True` 处加 `self._speech_start_time = now`（第246行后）：

```python
                        self._speaking = True
                        self._speech_start_time = now  # 记录语音起始时间
                        self._buffer = []
```

- [ ] **Step 4: 验证模块导入正常**

```bash
source .venv/bin/activate
python -c "from voice.av_recorder import AVRecorder; print('av_recorder OK')"
```

Expected: prints "av_recorder OK".

- [ ] **Step 5: 提交**

```bash
git add -A
git commit -m "feat: add barge-in support to AVRecorder

- set_tts_active(): enables higher VAD threshold during TTS playback
- check_interrupt(): returns True when sustained speech > min_duration
  (default 300ms) is detected — third layer of interrupt protection
- consume_utterance(): grabs VAD-completed audio without resetting state
- _on_tap(): uses dynamic VAD threshold (0.65 if TTS active, 0.5 otherwise)
- _speech_start_time tracks when user started speaking for duration gate

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: TTS Barge-in — 改 voice/main.py

**Files:**
- Modify: `voice/main.py`

- [ ] **Step 1: 修改 _speak_and_recover() — 返回打断音频**

替换 `voice/main.py:130-132`：

```python
    def _speak_and_recover(recorder, tts, text):
        """Speak TTS reply with barge-in support.

        Returns audio bytes (numpy float32) if user interrupted TTS,
        None if TTS completed normally or text was empty.
        """
        if not text or not text.strip():
            return None

        recorder.set_tts_active(True, config.get("bargein_threshold", 0.65))

        def check_barge_in():
            return recorder.check_interrupt(
                min_duration=config.get("bargein_duration", 0.3)
            )

        completed = tts.speak(text, interrupt_check=check_barge_in)
        recorder.set_tts_active(False)

        if not completed:
            logger.info("TTS 被用户打断")
            # 等待 VAD 完成当前语句的缓冲（最多 3 秒）
            audio = recorder.read_utterance(idle_timeout=3.0)
            return audio

        return None
```

- [ ] **Step 2: 添加 _process_interruption() 方法**

在 `_speak_and_recover()` 定义之后（约第165行）添加：

```python
    def _process_interruption(recorder, stt, hermes, tts, interrupted_audio):
        """Handle audio captured during TTS barge-in.
        
        If the user spoke during TTS playback and we have their audio,
        transcribe it, send to Hermes, and read the reply.
        """
        if interrupted_audio is None:
            return
        # 最多等 3 秒让用户把话说完
        audio = recorder.read_utterance(idle_timeout=3.0)
        if audio is None:
            return

        text = _to_simplified(stt.transcribe(audio))
        if not text or not text.strip():
            return

        logger.info("打断→ %s", text)
        try:
            reply = hermes.send(text)
            logger.info("API → %s", reply)
            _speak_and_recover(recorder, tts, reply)
        except ConnectionError:
            logger.warning("打断处理: Hermes API 不可达")
```

- [ ] **Step 3: 修改 LISTENING 状态的 TTS 调用（第189行）**

替换 `voice/main.py:189` 以及后续几行：

```python
                logger.info("朗读 → %s", reply)
                interrupted = _speak_and_recover(recorder, tts, reply)
                state = "AWAKE"
                recorder.start()
                logger.info("状态机 → 进入跟随时窗 (%.0f秒)",
                            session_timeout)
                _process_interruption(recorder, stt, hermes, tts, interrupted)
```

- [ ] **Step 4: 修改 AWAKE 状态的 TTS 调用（第215行）**

替换 `voice/main.py:213-215`：

```python
                logger.info("朗读 → %s", reply)
                interrupted = _speak_and_recover(recorder, tts, reply)
                _process_interruption(recorder, stt, hermes, tts, interrupted)
```

- [ ] **Step 5: 修改 ConnectionError 中的 _speak_and_recover（第219行）**

替换 `voice/main.py:219`：

```python
            _speak_and_recover(recorder, tts, "请先启动 Hermes 服务")
```

→ 保持原样即可（ConnectionError 时的 TTS 不需要打断检测——还没到正常 AWAKE 流程）。

- [ ] **Step 6: 验证 main.py 导入正常**

```bash
source .venv/bin/activate
python -c "from voice.main import main; print('main OK')"
```

Expected: prints "main OK".

- [ ] **Step 7: 提交**

```bash
git add -A
git commit -m "feat: integrate TTS barge-in into state machine

_speak_and_recover() now returns interrupted audio when user speaks
during TTS playback. _process_interruption() handles the follow-up:
transcribe → Hermes API → TTS reply.

Both LISTENING and AWAKE states use the same barge-in flow.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: 容错增强 — 麦克风权限 + 配置验证 + 模型校验

**Files:**
- Modify: `voice/models.py` — 添加 `validate_model_files()`
- Modify: `voice/main.py` — 添加 `check_mic_permission()`, `validate_config()`
- Modify: `voice/wake_word_engine.py` — 添加模型校验
- Modify: `config.yaml` — 新增 bargein 配置项

- [ ] **Step 1: 在 voice/models.py 末尾添加模型文件校验**

```python

# ── Validation ───────────────────────────────────────────────────────

def validate_model_files(model_dir_path, required_files):
    """Check that *required_files* exist in *model_dir_path* and are non-empty.

    Parameters
    ----------
    model_dir_path : str
        Directory containing model files.
    required_files : list of str
        Filenames (basenames) to check.

    Raises
    ------
    FileNotFoundError
        With a message listing all missing or empty files.
    """
    missing = []
    for fname in required_files:
        fpath = os.path.join(model_dir_path, fname)
        if not os.path.isfile(fpath) or os.path.getsize(fpath) == 0:
            missing.append(fname)
    if missing:
        raise FileNotFoundError(
            f"Model files missing or empty in {model_dir_path}: {missing}"
        )
```

- [ ] **Step 2: 在 voice/main.py 中添加麦克风权限检查**

在 `voice/main.py` 的 `import` 部分添加：

```python
import AVFoundation
```

在 `play_beep()` 函数之后、`main()` 之前（约第83行），添加：

```python
def check_mic_permission():
    """Check macOS microphone permission. Raise if denied."""
    import AVFoundation

    status = AVFoundation.AVCaptureDevice.authorizationStatusForMediaType_(
        AVFoundation.AVMediaTypeAudio
    )
    if status == AVFoundation.AVAuthorizationStatusDenied:
        logger.critical(
            "麦克风权限被拒绝。请在 系统设置 → 隐私与安全性 → 麦克风 中允许本应用"
        )
        sys.exit(1)
    elif status == AVFoundation.AVAuthorizationStatusNotDetermined:
        logger.info("请求麦克风权限 …")
        # AVFoundation will prompt on first access, no explicit call needed.
        # The later recorder.start() triggers the system permission dialog.
```

- [ ] **Step 3: 在 voice/main.py 中添加配置验证**

```python
def validate_config(cfg):
    """Validate required config keys and their types."""
    required = {
        "samplerate": (int, float),
        "silence_timeout": (int, float),
        "hermes_url": str,
        "hermes_api_key": str,
    }
    for key, expected_type in required.items():
        if key not in cfg:
            raise ValueError(f"配置缺少必要字段: {key}")
        if not isinstance(cfg[key], expected_type):
            raise TypeError(
                f"配置字段 {key} 类型错误: "
                f"期望 {expected_type.__name__}, 实际 {type(cfg[key]).__name__}"
            )
```

- [ ] **Step 4: 在 main() 中调用权限检查和配置验证**

修改 `voice/main.py` 的 `main()` 函数开头：

在 `setup_logging()` 和 `config = load_config()` 之后（约第110行），添加：

```python
    check_mic_permission()
    validate_config(config)
```

- [ ] **Step 5: 在 WakeWordEngine.__init__() 中添加模型校验**

修改 `voice/wake_word_engine.py`，在 `__init__()` 中找到 encoder/decoder/joiner 解析后添加校验。

在 `WakeWordEngine.__init__()` 中（约第54行），找到：

```python
        self._spotter = sherpa_onnx.KeywordSpotter(
```

在这之前添加：

```python
        from .models import validate_model_files
        validate_model_files(model_dir, ["tokens.txt", os.path.basename(encoder),
                                          os.path.basename(decoder),
                                          os.path.basename(joiner)])
```

- [ ] **Step 6: 在 config.yaml 中添加 bargein 配置项**

```yaml

# ── TTS 打断（Barge-in） ──────────────────────────
bargein_threshold: 0.65       # TTS 期间 VAD 阈值（0.0-1.0，越高越严格）
bargein_duration: 0.3         # 打断确认时长（秒，默认 300ms）
```

- [ ] **Step 7: 验证启动检查**

```bash
source .venv/bin/activate
python -c "
from voice.main import check_mic_permission, validate_config
import yaml
with open('config.yaml') as f:
    cfg = yaml.safe_load(f)
validate_config(cfg)
print('配置验证通过')
check_mic_permission()
print('权限检查通过')
"
```

- [ ] **Step 8: 提交**

```bash
git add -A
git commit -m "feat: add startup resilience checks

- Mic permission check with helpful error message
- Config validation (required keys + types)
- Model file validation for KWS, whisper, and sensevoice
- config.yaml: add bargein_threshold and bargein_duration

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### 最终验证

- [ ] **Step 1: 完整启动测试**

```bash
source .venv/bin/activate
python -m voice.main
```

Expected: daemon starts normally, shows log output, waits for "小九" wake word.

- [ ] **Step 2: 触发唤醒并对话**

Speak "小九" then a short command. Verify:
- KWS detects wake word
- STT transcribes
- Hermes API responds
- TTS plays the reply
- You can interrupt TTS by speaking again

- [ ] **Step 3: 验证配置错误时的行为**

```bash
# 临时改一个错误的配置项来测试验证逻辑
sed -i '' 's/samplerate: 16000/samplerate: "not-a-number"/' config.yaml
python -m voice.main
# 应报错退出
git checkout config.yaml  # 恢复
```

- [ ] **Step 4: 最终提交**

```bash
git add -A
git commit -m "chore: finalize phase 1 refactor

- Dead code cleanup
- DRY model resolution
- TTS barge-in
- Error resilience

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```
