"""Benchmark prompt loaders for ComfyClaw experiments."""

from .geneval2 import GENEVAL2_CONFIG
from .dpg_bench import DPG_BENCH_CONFIG
from .oneig_en import ONEIG_EN_CONFIG
from .oneig_zh import ONEIG_ZH_CONFIG
from .wise import WISE_CONFIG

BENCHMARKS = {
    "geneval2": GENEVAL2_CONFIG,
    "dpg-bench": DPG_BENCH_CONFIG,
    "oneig-en": ONEIG_EN_CONFIG,
    "oneig-zh": ONEIG_ZH_CONFIG,
    "wise": WISE_CONFIG,
}
