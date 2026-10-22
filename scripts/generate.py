"""Stream a completion from the configured model: python -m scripts.generate "Say hi" """

import argparse
import time

from agent_bot.config import load_config
from agent_bot.inference import create_backend


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt")
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model_path:
        cfg.model.path = args.model_path
    backend = create_backend(cfg.model)

    start = time.perf_counter()
    for chunk in backend.generate(
        [{"role": "user", "content": args.prompt}],
        [],
        enable_thinking=True,
        max_new_tokens=cfg.generation.max_new_tokens,
        temperature=cfg.generation.temperature,
    ):
        print(chunk, end="", flush=True)
    elapsed = time.perf_counter() - start
    n = backend.last_token_count
    print(f"\n\n[{n} tokens in {elapsed:.1f}s = {n / elapsed:.2f} tok/s]")


if __name__ == "__main__":
    main()
