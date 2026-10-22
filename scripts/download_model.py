"""Download Hugging Face weights into ./models/<repo_name>/.

Usage: python -m scripts.download_model Qwen/Qwen3-1.7B
Set HF_TOKEN for gated models.
"""

import os
import sys

from huggingface_hub import snapshot_download

# safetensors + tokenizer + config only; skips .bin / GGUF / ONNX duplicates.
ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "tokenizer*"]


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: python -m scripts.download_model <repo_id>")
    repo_id = sys.argv[1]
    dest = f"models/{repo_id.split('/')[-1]}"
    snapshot_download(
        repo_id,
        local_dir=dest,
        allow_patterns=ALLOW_PATTERNS,
        token=os.environ.get("HF_TOKEN"),
    )
    print(f"Downloaded {repo_id} to {dest}")


if __name__ == "__main__":
    main()
