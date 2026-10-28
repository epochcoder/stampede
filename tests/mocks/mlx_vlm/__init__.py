"""Stand-in for the mlx-vlm 0.7.x API used by stampede: load, generate (returns an object with
.text), prompt_utils.apply_chat_template, utils.load_config.

Env vars (answers of the "model"):
  MOCK_LINE    answer to the stamp-line question (default "'87 6 14": an adversarial model that
               "reads" the same date on everything)
  MOCK_FULL    answer to the full-photo question
  MOCK_ORIENT  answer to the rotation question (tile number, default "1")"""
import os
from dataclasses import dataclass

LOG = []
ANSWERS = {"line": os.environ.get("MOCK_LINE", "'87 6 14"), "full": os.environ.get("MOCK_FULL", "'87 6 14"),
           "orient": os.environ.get("MOCK_ORIENT", "1")}


@dataclass
class GenerationResult:
    text: str = ""


def load(path_or_hf_repo, adapter_path=None, lazy=False, revision=None, strict=True, **kw):
    LOG.append(("load", path_or_hf_repo))
    return "MODEL", "PROCESSOR"


def generate(model, processor, prompt, image=None, audio=None, video=None, verbose=False, **kwargs):
    assert model == "MODEL" and processor == "PROCESSOR"
    assert isinstance(image, list) and all(os.path.isfile(p) for p in image), image
    assert kwargs.get("temperature") == 0.0 and kwargs.get("max_tokens")
    kind = "orient" if "numbered tiles" in prompt else "full" if "scanned photo print" in prompt else "line"
    LOG.append(("generate", kind))
    return GenerationResult(ANSWERS[kind])
