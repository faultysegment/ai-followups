import whisperx
import torch
import threading
import queue
import time

class AudioTranscriber:
    def __init__(self, language='auto'):
        """
        Initialize transcriber with language option
        language: 'auto', 'en', or 'ru'
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.whisper_model = whisperx.load_model("base", self.device, compute_type="float32")
        self.transcription_queue = queue.Queue()
        self.running = False
        self.full_transcript = []
        self.language = language

    def start(self):
        """Start the transcription process"""
        self.running = True
        self.full_transcript = []
        self.transcription_thread = threading.Thread(target=self._process_queue)
        self.transcription_thread.start()

    def stop(self):
        """Stop the transcription process"""
        self.running = False
        if hasattr(self, 'transcription_thread'):
            self.transcription_thread.join()
        return " ".join(self.full_transcript)

    def add_audio(self, audio_file):
        """Add audio file to transcription queue"""
        self.transcription_queue.put(audio_file)

    def _process_queue(self):
        """Process audio files in the queue"""
        while self.running:
            try:
                audio_file = self.transcription_queue.get(timeout=1)
                if audio_file:
                    transcript = self.transcribe(audio_file)
                    if transcript:
                        self.full_transcript.append(transcript)
                        print("\nTranscription:")
                        print("-" * 50)
                        print(transcript)
                        print("-" * 50)
            except queue.Empty:
                continue

    def transcribe(self, audio_file):
        """Transcribe audio file using WhisperX"""
        try:
            # Set language if not auto
            if self.language != 'auto':
                result = self.whisper_model.transcribe(audio_file, language=self.language)
            else:
                result = self.whisper_model.transcribe(audio_file)

            # Extract text from segments
            if isinstance(result, dict) and "segments" in result:
                transcript = " ".join([segment["text"] for segment in result["segments"]])
            else:
                transcript = result.text if hasattr(result, 'text') else str(result)

            return transcript

        except Exception as e:
            print(f"Transcription error: {e}")
            return None
