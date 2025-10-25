import logging
from pathlib import Path

PROMPTS_FILEPATH = Path("src/prompts.yaml")


def get_logger(LOG_LEVEL="INFO"):
    LOG_PATH = Path("logs.log")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    log = logging.Logger("agentic_search")
    log.setLevel(LOG_LEVEL)

    file_handler = logging.FileHandler(LOG_PATH)
    file_handler.setLevel(LOG_LEVEL)
    file_handler.setFormatter(formatter)

    log.addHandler(file_handler)

    return log


log = get_logger("DEBUG")

MODEL_COMBOS = {
    "linux": {
        "embed_model": "Qwen/Qwen3-Embedding-0.6B",
        "gen_model": "Qwen/Qwen3-4B-AWQ",
        # 'gen_model': "Qwen/Qwen3-0.6B-GPTQ-Int8"
        # 'gen_model': "Qwen/Qwen3-1.7B-GPTQ-Int8"
    },
    # feel free to replace with any ??B-MLX-?bit versions from Qwen3 Collection at:
    # https://huggingface.co/collections/Qwen/qwen3-67dd247413f0e2e4f653967f
    "mac": {
        "embed_model": "Qwen/Qwen3-Embedding-0.6B",
        "gen_model": "Qwen/Qwen3-4B-MLX-4bit",
    },
    "mac_mid": {
        "embed_model": "Qwen/Qwen3-Embedding-0.6B",
        "gen_model": "Qwen/Qwen3-4B-MLX-6bit",
    },
    "mac_high": {
        "embed_model": "Qwen/Qwen3-Embedding-0.6B",
        "gen_model": "Qwen/Qwen3-4B-MLX-8bit",
    },
    # HF-low is same as `linux-local`
    "HF-mid": {
        "embed_model": "Qwen/Qwen3-Embedding-0.6B",
        "gen_model": "Qwen/Qwen3-8B-AWQ",
    },
    "HF-high": {
        "embed_model": "Qwen/Qwen3-Embedding-4B",
        "gen_model": "Qwen/Qwen3-14B-AWQ",
    },
}
