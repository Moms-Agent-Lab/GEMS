"""Model workflow definitions for ComfyClaw benchmarks."""

from .longcat import LONGCAT_CONFIG
from .qwen import QWEN_CONFIG
from .dreamshaper import DREAMSHAPER_CONFIG

MODELS = {
    "longcat": LONGCAT_CONFIG,
    "qwen": QWEN_CONFIG,
    "dreamshaper": DREAMSHAPER_CONFIG,
}
