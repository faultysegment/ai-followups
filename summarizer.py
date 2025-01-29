from datetime import datetime
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama

class TextSummarizer:
    def __init__(self,
                 model_repo="TheBloke/Llama-2-7B-Chat-GGUF",
                 model_file="llama-2-7b-chat.Q4_K_M.gguf",
                 language='en'):
        self.model_path = self._ensure_model_exists(model_repo, model_file)
        self.language = language
        self.llm = Llama(
            model_path=self.model_path,
            n_ctx=2048,
            n_threads=4
        )

    def _ensure_model_exists(self, repo_id, filename):
        """Download the model if it doesn't exist"""
        try:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "ai-followups")
            os.makedirs(cache_dir, exist_ok=True)

            print(f"Checking for model in {cache_dir}...")

            model_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                cache_dir=cache_dir
            )

            print(f"Model path: {model_path}")
            return model_path

        except Exception as e:
            print(f"Error downloading model: {e}")
            raise

    def summarize(self, text):
        """Summarize text using Llama model"""
        try:
            # Different prompts for different languages
            if self.language == 'ru':
                prompt = f"""Пожалуйста, предоставьте краткое содержание следующего текста:

{text}

Краткое содержание:"""
            else:
                prompt = f"""Please provide a concise summary of the following text:

{text}

Summary:"""

            response = self.llm(
                prompt,
                max_tokens=500,
                temperature=0.7,
                top_p=0.95,
                stop=["Summary:", "Краткое содержание:", "\n\n"],
                echo=False
            )

            return response['choices'][0]['text'].strip()

        except Exception as e:
            print(f"Summarization error: {e}")
            return None

    def save_results(self, results, output_dir):
        """Save transcription and summary results to text file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"transcription_summary_{timestamp}.txt")

        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(f"File: {result['file']}\n")
                f.write("\nTranscription:\n")
                f.write("-" * 50 + "\n")
                f.write(result['transcript'] + "\n")
                f.write("\nSummary:\n")
                f.write("-" * 50 + "\n")
                f.write(result['summary'] + "\n")
                f.write("\n" + "=" * 50 + "\n\n")

        print(f"\nResults saved to: {output_file}")
