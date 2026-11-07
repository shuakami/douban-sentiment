"""Shared data locations for the experiment pipeline.

The 911K-review corpus and intermediate artefacts live outside git.
Override the root with the DOUBAN_DATA_ROOT environment variable:

    export DOUBAN_DATA_ROOT=/path/to/data_root   # contains pseudo/ features/ clean/ models/ results/
"""
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("DOUBAN_DATA_ROOT", r"E:/GB77_data"))
