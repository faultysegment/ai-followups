
from recorder import AudioRecorder
from transcriber import AudioTranscriber
from summarizer import TextSummarizer
import time

def main():
    print("Real-time Audio Recording and Transcription Tool")
    print("=" * 50)

    # Language selection
    while True:
        print("\nSelect language:")
        print("1. English")
        print("2. Russian")
        print("3. Auto-detect")

        lang_choice = input("\nSelect option (1-3): ").strip()

        if lang_choice == "1":
            language = "en"
            break
        elif lang_choice == "2":
            language = "ru"
            break
        elif lang_choice == "3":
            language = "auto"
            break
        else:
            print("Invalid choice. Please try again.")

    # Initialize components with selected language
    recorder = AudioRecorder(chunk_duration=60)
    transcriber = AudioTranscriber(language=language)

    # Use a multilingual model that supports both English and Russian
    summarizer = TextSummarizer(
        model_repo="TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        model_file="mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        language=language
    )

    while True:
        print("\nOptions:")
        print("1. Start recording and transcription")
        print("2. Exit")

        choice = input("\nSelect an option: ").strip()

        if choice == "1":
            try:
                print("\nStarting recording and transcription...")
                print("Press Ctrl+C to stop")

                recorder.start_recording()
                transcriber.start()

                try:
                    while True:
                        chunk = recorder.get_next_chunk()
                        if chunk is not None:
                            chunk_file = recorder.save_chunk(chunk)
                            if chunk_file:
                                transcriber.add_audio(chunk_file)

                        time.sleep(0.1)

                except KeyboardInterrupt:
                    print("\nStopping recording and transcription...")
                    recorder.stop_recording()
                    full_transcript = transcriber.stop()

                    if full_transcript:
                        print("\nComplete Transcript:")
                        print("=" * 50)
                        print(full_transcript)
                        print("=" * 50)

                        print("\nGenerating summary...")
                        summary = summarizer.summarize(full_transcript)
                        if summary:
                            print("\nSummary:")
                            print("-" * 50)
                            print(summary)
                            print("-" * 50)

                            results = [{
                                'file': 'recording_session',
                                'transcript': full_transcript,
                                'summary': summary
                            }]
                            summarizer.save_results(results, "recordings")

                    print("Recording and transcription stopped")

            except Exception as e:
                print(f"An error occurred: {e}")

        elif choice == "2":
            print("Goodbye!")
            break

        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
