# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the voice assistant
python -m voice.main

# One-shot start (auto-starts Hermes Gateway if needed)
bash scripts/start.sh

# Install dependencies
bash scripts/install.sh

# Activate venv
source .venv/bin/activate
```

No build system, no test framework, no linter configured yet.

## Architecture

Hermes Voice is a macOS wake-word-activated voice assistant daemon. The user says "小九" (Xiaojiu) to wake it, then speaks a command which is transcribed, sent to a local Hermes API server, and the reply is read aloud via macOS TTS.

### State Machine

Two states, managed in `voice/main.py`:

- **LISTENING** (default) — VAD + faster-whisper tiny + text match for "小九". When detected, strips the wake-word prefix and submits text to Hermes.
- **AWAKE** — 30-second follow-up window. Any speech is transcribed and submitted directly (no wake-word filtering). Each TTS reply resets the 30s timer. Timeout → back to LISTENING.

### Module Layout (`voice/`)

| File | Role |
|------|------|
| `main.py` | Daemon entry point, signal handlers, state machine loop, config loading |
| `recorder.py` | `AudioRecorder` — sounddevice InputStream + silero-vad for speech detection and audio buffering |
| `wake_word.py` | `WakeWordDetector` — faster-whisper wrapper; transcribes audio, detects/strips "小九" with homophone fallback for "九" (酒/就/久/舅/救/旧) |
| `stt.py` | `process_transcription()` — state-dependent text routing (LISTENING strips wake word, AWAKE passes through) |
| `hermes_client.py` | `HermesClient` — httpx-based HTTP client for Hermes API Server (`/v1/chat/completions`), maintains multi-turn message context |
| `tts.py` | `TTSEngine` — macOS AVSpeechSynthesizer via pyobjc, blocks until utterance finishes |

### Key Design Details

- **Wake word + STT reuse**: faster-whisper runs once per utterance. In LISTENING state the same transcription is used for both wake-word detection and command extraction (no second inference pass).
- **VAD flow**: `silero-vad` runs in the audio callback thread. When speech starts, audio is buffered; when silence exceeds `silence_timeout` (1.5s), the full utterance is delivered to the main thread via a threading.Event.
- **TTS echo avoidance**: `_speak_and_recover()` in main.py stops the audio stream before TTS, then restarts it with a 300ms drain delay to flush residual mic buffer.
- **ConnectionError fallback**: if Hermes API is unreachable, says "请先启动 Hermes 服务" and returns to LISTENING.
- **Model sources**: wake_word.py supports `huggingface` (default, lets faster-whisper handle download), `modelscope` (downloads from ModelScope, converts to CTranslate2), and `local` (pre-downloaded CTranslate2 directory).
- **Multi-turn**: HermesClient accumulates messages[] across turns. `clear_context()` resets it on state timeout.

### Configuration

`config.yaml` — session timeout, sample rate, VAD silence threshold, whisper model size, model source, Hermes API URL/key. API key can also come from `HERMES_API_KEY` env var.

### Dependencies

Python 3.12+, macOS 13+ (AVSpeechSynthesizer), Homebrew (portaudio), Hermes Gateway (running on localhost:8642).

## Coding Guidelines

Apply the [Karpathy LLM Coding Guidelines](https://x.com/karpathy/status/2015883857489522876):

1. **Think Before Coding** — State assumptions explicitly before implementing. Surface ambiguities and tradeoffs. Push back on overcomplicated approaches.
2. **Simplicity First** — Minimum code that solves the problem. No speculative features, abstractions, or configurability. If 200 lines can be 50, rewrite it.
3. **Surgical Changes** — Touch only what the task requires. Don't improve adjacent code, refactor what isn't broken, or change style. Clean up orphans your changes create; leave pre-existing dead code alone.
4. **Goal-Driven Execution** — Define verifiable success criteria before starting. For multi-step tasks, state plan as `step → verify: check`. Loop until criteria are met.
