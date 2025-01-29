import soundcard as sc
import numpy as np
import soundfile as sf
import threading
import queue
from datetime import datetime
import os

class AudioRecorder:
    def __init__(self, sample_rate=16000, chunk_duration=5):
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration  # Duration of each chunk in seconds
        self.chunk_size = int(sample_rate * chunk_duration)
        self.mic = sc.default_microphone()
        self.speaker = sc.get_microphone(include_loopback=True, id=sc.default_speaker().name)
        self.recording = False
        self.mic_queue = queue.Queue()
        self.speaker_queue = queue.Queue()
        self.audio_chunks = queue.Queue()

    def start_recording(self):
        """Start continuous recording"""
        self.recording = True
        self.recording_thread = threading.Thread(target=self._continuous_record)
        self.recording_thread.start()

    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        if hasattr(self, 'recording_thread'):
            self.recording_thread.join()

    def _continuous_record(self):
        """Continuously record audio in chunks"""
        mic_thread = threading.Thread(target=self._record_stream_continuous,
                                    args=(self.mic, self.mic_queue))
        speaker_thread = threading.Thread(target=self._record_stream_continuous,
                                        args=(self.speaker, self.speaker_queue))

        mic_thread.start()
        speaker_thread.start()

        while self.recording:
            try:
                # Get chunks from both streams
                mic_chunk = self.mic_queue.get(timeout=1)
                speaker_chunk = self.speaker_queue.get(timeout=1)

                # Ensure chunks have the same length
                min_length = min(len(mic_chunk), len(speaker_chunk))
                mic_chunk = mic_chunk[:min_length]
                speaker_chunk = speaker_chunk[:min_length]

                # Convert to stereo if needed
                if mic_chunk.shape[1] > 2:
                    mic_chunk = mic_chunk[:, :2]
                if speaker_chunk.shape[1] > 2:
                    speaker_chunk = speaker_chunk[:, :2]

                # Merge the streams
                merged_chunk = mic_chunk + speaker_chunk

                # Normalize
                max_val = np.max(np.abs(merged_chunk))
                if max_val > 0:
                    merged_chunk = np.int16(merged_chunk / max_val * 32767)
                else:
                    merged_chunk = np.int16(merged_chunk)

                # Add to audio chunks queue
                self.audio_chunks.put(merged_chunk)

            except queue.Empty:
                continue

        mic_thread.join()
        speaker_thread.join()

    def _record_stream_continuous(self, stream, data_queue):
        """Continuously record from a stream in chunks"""
        try:
            with stream.recorder(samplerate=self.sample_rate) as recorder:
                while self.recording:
                    data = recorder.record(numframes=self.chunk_size)
                    data_queue.put(data)
        except Exception as e:
            print(f"Recording error: {e}")

    def get_next_chunk(self):
        """Get the next available audio chunk"""
        try:
            return self.audio_chunks.get_nowait()
        except queue.Empty:
            return None

    def save_chunk(self, chunk, output_dir="recordings"):
        """Save an audio chunk to file"""
        if chunk is None:
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(output_dir, exist_ok=True)
        chunk_file = os.path.join(output_dir, f"chunk_{timestamp}.wav")
        sf.write(chunk_file, chunk, self.sample_rate)
        return chunk_file
