import objc
from Foundation import NSObject, NSRunLoop, NSDate
import AVFoundation


class SpeechDelegate(NSObject):
    """Delegate that signals when AVSpeechSynthesizer finishes speaking."""

    def init(self):
        self = objc.super(SpeechDelegate, self).init()
        self.finished = False
        return self

    def speechSynthesizer_didFinishSpeechUtterance_(self, synthesizer, utterance):
        self.finished = True


class TTSEngine:
    """macOS AVSpeechSynthesizer wrapper.

    Uses pyobjc to call the native macOS speech synthesizer.
    ``speak()`` blocks until the utterance finishes.
    """

    _shared_delegate = None

    def __init__(self):
        self.synthesizer = AVFoundation.AVSpeechSynthesizer.alloc().init()
        if TTSEngine._shared_delegate is None:
            TTSEngine._shared_delegate = SpeechDelegate.alloc().init()
        self.synthesizer.setDelegate_(TTSEngine._shared_delegate)

    def speak(self, text):
        """Speak *text* and block until done."""
        if not text or not text.strip():
            return

        utterance = AVFoundation.AVSpeechUtterance.speechUtteranceWithString_(
            text
        )
        voice = AVFoundation.AVSpeechSynthesisVoice.voiceWithLanguage_("zh-CN")
        utterance.setVoice_(voice)

        delegate = self.synthesizer.delegate()
        delegate.finished = False

        self.synthesizer.speakUtterance_(utterance)

        while not delegate.finished:
            NSRunLoop.currentRunLoop().runUntilDate_(
                NSDate.dateWithTimeIntervalSinceNow_(0.1)
            )

    def stop(self):
        self.synthesizer.stopSpeakingAtBoundary_(
            AVFoundation.AVSpeechBoundaryImmediate
        )
