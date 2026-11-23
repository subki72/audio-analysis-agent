"""Utilities for discovering FFmpeg and FFprobe binaries across env, PATH, and Conda."""

from __future__ import annotations

import glob
import logging
import os
import shutil
import sys

logger = logging.getLogger(__name__)

_CONDA_ROOT_NAMES = ("miniconda3", "anaconda3", "Miniconda3", "Anaconda3")


def _find_in_conda(binary_name: str) -> str | None:
    """Search common Conda environment directories for ``binary_name``."""
    home = os.path.expanduser("~")
    conda_paths = [
        os.path.join(home, root, "envs", "voicescript", "Library", "bin")
        for root in _CONDA_ROOT_NAMES
    ]
    conda_prefix = os.getenv("CONDA_PREFIX", "")
    if conda_prefix:
        conda_paths.append(os.path.join(conda_prefix, "Library", "bin"))

    for directory in conda_paths:
        candidate = os.path.join(directory, binary_name)
        if os.path.isfile(candidate):
            return candidate

    for root in _CONDA_ROOT_NAMES:
        pattern = os.path.join(home, root, "envs", "*", "Library", "bin", binary_name)
        matches = glob.glob(pattern)
        if matches:
            return matches[0]

    return None


def _resolve_binary(name: str, env_var: str) -> str:
    """Resolve an executable from env override, system PATH, or Conda environments."""
    override = os.getenv(env_var)
    if override and os.path.isfile(override):
        logger.debug("Resolved %s from %s: %s", name, env_var, override)
        return override

    path = shutil.which(name)
    if path:
        return path

    exe_name = f"{name}.exe" if sys.platform == "win32" else name
    conda_path = _find_in_conda(exe_name)
    if conda_path:
        logger.info("Resolved %s from Conda environment: %s", name, conda_path)
        return conda_path

    logger.error("%s binary not found via %s, PATH, or Conda", name, env_var)
    raise FileNotFoundError(
        f"{name} tidak ditemukan. Pastikan {name} terinstal dan ada di PATH, "
        "atau aktifkan conda environment 'voicescript' terlebih dahulu:\n"
        "  conda activate voicescript"
    )


def find_ffmpeg() -> str:
    """Return the resolved path to the ``ffmpeg`` binary.

    Raises:
        FileNotFoundError: If ``ffmpeg`` cannot be located.
    """
    return _resolve_binary("ffmpeg", "FFMPEG_PATH")


def find_ffprobe() -> str:
    """Return the resolved path to the ``ffprobe`` binary.

    Raises:
        FileNotFoundError: If ``ffprobe`` cannot be located.
    """
    return _resolve_binary("ffprobe", "FFPROBE_PATH")
