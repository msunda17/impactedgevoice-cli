"""
download_models.py — Download the recommended model suite for ImpactEdgeVoice.

Recommended suite:
  * Llama 3.2 1B Q4  — Simple voice Q&A (fast, ~0.8 GB)
  * Llama 3.2 3B Q4  — Complex tasks, document summarization (~2.0 GB)
  * Whisper small.en — ASR (downloads automatically on first run via faster-whisper)

Run: python download_models.py
"""

import subprocess
import os
import sys

MODELS_DIR = "models"

MODELS = {
    # Tier 1: Simple voice Q&A — loaded for every voice session
    "llama-3.2-1b": {
        "repo": "bartowski/Llama-3.2-1B-Instruct-GGUF",
        "file": "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
        "desc": "Fast voice Q&A (1B params, ~0.8 GB)",
    },

    # Tier 2: Complex reasoning, documents, summarization
    "llama-3.2-3b": {
        "repo": "bartowski/Llama-3.2-3B-Instruct-GGUF",
        "file": "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        "desc": "Document Q&A, complex reasoning (3B params, ~2.0 GB)",
    },

    # Optional: Best ASR for long-form content
    "whisper-small": {
        "repo": "Systran/faster-whisper-small.en",
        "file": None,  # faster-whisper handles download automatically
        "desc": "ASR model — downloads automatically on first run",
    },
}


def download_gguf(repo: str, filename: str) -> bool:
    """Download a GGUF model from HuggingFace using hf CLI."""
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"

    print(f"\nDownloading {filename} from {repo}...")
    print(f"URL: {url}")

    try:
        result = subprocess.run(
            ["hf", "download", repo, filename, "--local-dir", MODELS_DIR],
            capture_output=True,
            text=True,
            check=True,
        )
        print(f"✓ Downloaded successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Download failed: {e}")
        print(f"stderr: {e.stderr}")
        return False
    except FileNotFoundError:
        print("✗ hf CLI not found. Install with: pip install huggingface-hub")
        return False


def check_model_exists(filename: str) -> bool:
    """Check if model already exists locally."""
    path = os.path.join(MODELS_DIR, filename)
    return os.path.exists(path)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("=" * 60)
    print("ImpactEdgeVoice Model Download Tool")
    print("=" * 60)
    print("\nRecommended model suite for adaptive routing:")

    for key, info in MODELS.items():
        status = ""
        if info["file"] and check_model_exists(info["file"]):
            status = " [ALREADY EXISTS]"
        print(f"  • {key}: {info['desc']}{status}")

    print("\n" + "=" * 60)

    # Download each model
    for key, info in MODELS.items():
        if info["file"] is None:
            # Whisper model — faster-whisper handles download automatically
            print(f"\n{info['desc']} will download on first use.")
            continue

        if check_model_exists(info["file"]):
            print(f"\n✓ {info['file']} already exists, skipping.")
            continue

        print(f"\nDownloading {key}...")
        success = download_gguf(info["repo"], info["file"])

        if not success:
            print(f"\nFailed to download {key}. You can:")
            print(f"  1. Manually download from https://huggingface.co/{info['repo']}")
            print(f"  2. Place the file in ./{MODELS_DIR}/")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("Run: python -m impactedgevoice")
    print("=" * 60)


if __name__ == "__main__":
    main()
