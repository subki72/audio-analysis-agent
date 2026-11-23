"""VoiceScript CLI entrypoint for audio analysis, batch processing, and transcription."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import click
from dotenv import load_dotenv

from src.agent.orchestrator import AudioAnalysisAgent
from src.audio.processor import AudioProcessor
from src.config import SUPPORTED_AUDIO_EXTENSIONS, AppConfig, configure_logging
from src.health import check_system_health
from src.tools.silence import configure as configure_silence
from src.tools.volume import configure as configure_volume
from src.transcription.transcriber import Transcriber
from src.utils.helpers import load_config

load_dotenv()

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "settings.yaml")
SUPPORTED_EXTENSIONS = set(SUPPORTED_AUDIO_EXTENSIONS)


def _prepare_analysis_settings(verbose: bool) -> dict[str, Any]:
    """Load YAML configuration, initialize logging, and configure FFmpeg tool thresholds."""
    settings = load_config(CONFIG_PATH)
    config = AppConfig.from_env(settings, load_env_file=False)
    configure_logging("INFO" if verbose else config.log_level)

    analysis_cfg = settings.get("analysis", {})
    configure_silence(analysis_cfg)
    configure_volume(analysis_cfg)
    return settings


def _save_json_report(report: dict[str, Any] | list[dict[str, Any]], output_path: str) -> None:
    """Write a JSON analysis report to disk with UTF-8 encoding."""
    with open(output_path, "w", encoding="utf-8") as report_file:
        json.dump(report, report_file, ensure_ascii=False, indent=2)
    logger.info("Saved report to %s", output_path)


def _display_report_summary(report: dict[str, Any], report_path: str) -> None:
    """Render human-readable summary, actions, and formatted JSON report to stdout."""
    click.echo()
    click.echo(click.style("=== Laporan Analisis ===", fg="yellow", bold=True))

    insights = report.get("llm_insights", {})
    summary = insights.get("summary", "Tidak ada summary.")
    actions = insights.get("recommended_actions", [])

    click.echo(f"\n[Summary]\n   {summary}")
    if actions:
        click.echo("\n[Rekomendasi]")
        for index, action in enumerate(actions, 1):
            click.echo(f"   {index}. {action}")

    click.echo()
    click.echo(click.style(f"[+] Laporan lengkap disimpan ke: {report_path}", fg="green"))
    click.echo()
    click.echo(click.style("=== JSON Report ===", fg="yellow"))
    click.echo(json.dumps(report, ensure_ascii=False, indent=2))


def _collect_audio_files(directory: str) -> list[str]:
    """Collect sorted supported audio file paths from a target directory."""
    audio_files: list[str] = []
    for fname in sorted(os.listdir(directory)):
        ext = os.path.splitext(fname)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            audio_files.append(os.path.join(directory, fname))
    return audio_files


def _process_batch_file(
    agent: AudioAnalysisAgent,
    audio_path: str,
    directory: str,
    index: int,
    total: int,
    verbose: bool,
) -> dict[str, Any]:
    """Analyze a single file within a batch run and persist its individual report."""
    basename = os.path.basename(audio_path)
    click.echo(click.style(f"--- [{index}/{total}] {basename} ---", fg="cyan"))
    report = agent.analyze(audio_path, verbose=verbose)

    if "error" in report:
        click.echo(click.style(f"  [-] Error: {report['error']}", fg="red"))
        return report

    name_without_ext = os.path.splitext(basename)[0]
    individual_path = os.path.join(directory, f"{name_without_ext}_report.json")
    _save_json_report(report, individual_path)

    summary = report.get("llm_insights", {}).get("summary", "N/A")
    click.echo(f"  [+] {summary[:100]}...")
    click.echo(click.style(f"  [+] Disimpan: {individual_path}", fg="green"))
    return report


@click.group()
def cli() -> None:
    """VoiceScript - Audio Analysis Agent & Transcription Tool (Groq API)."""


@cli.command()
def health() -> None:
    """Memeriksa status kesiapan sistem (FFmpeg, FFprobe, dan konfigurasi)."""
    payload = check_system_health()
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["status"] != "healthy":
        sys.exit(1)


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option(
    "--output-format",
    "-f",
    default=None,
    type=click.Choice(["txt", "srt", "vtt", "json"]),
    help="Format output transkripsi.",
)
@click.option("--language", "-l", default="id", help="Kode bahasa audio (contoh: id, en).")
def transcribe(audio_path: str, output_format: str | None, language: str) -> None:
    """Mentranskripsi file audio ke format teks/script."""
    settings = load_config(CONFIG_PATH)
    resolved_format = output_format or settings.get("output", {}).get("default_format", "txt")

    click.echo("[*] Menggunakan provider: GROQ")
    click.echo(f"[*] Model transkripsi: {settings.get('api', {}).get('model', 'whisper-large-v3')}")
    click.echo(f"[*] File audio: {audio_path}")
    click.echo(f"[*] Bahasa target: {language} | Format output: {resolved_format}")

    try:
        processor = AudioProcessor(settings)
        transcriber = Transcriber(settings)
        meta = processor.get_metadata(audio_path)
        click.echo(f"[*] Ukuran audio: {meta['size_bytes'] / (1024 * 1024):.2f} MB")

        max_size_mb = settings.get("audio", {}).get("max_size_mb", 25)
        audio_files = processor.split_audio(audio_path, max_size_mb=max_size_mb)
        segments = _transcribe_segments(transcriber, audio_files, language, resolved_format)

        name_without_ext = os.path.splitext(os.path.basename(audio_path))[0]
        out_dir = settings.get("output", {}).get("dir", "data/transcripts")
        out_path = os.path.join(out_dir, f"{name_without_ext}.{resolved_format}")
        transcriber.save_output("\n\n".join(segments), out_path)
        click.echo(click.style(f"[+] Transkripsi selesai! Output disimpan ke: {out_path}", fg="green"))
    except Exception as exc:
        logger.error("Transcription command failed: %s", exc)
        click.echo(click.style(f"[-] Terjadi kesalahan: {exc}", fg="red"), err=True)
        sys.exit(1)


def _transcribe_segments(
    transcriber: Transcriber,
    audio_files: list[str],
    language: str,
    response_format: str,
) -> list[str]:
    """Transcribe one or more split audio segments sequentially via Groq Whisper."""
    transcripts: list[str] = []
    total = len(audio_files)
    for index, segment_path in enumerate(audio_files, 1):
        status_msg = (
            f"[*] Mentranskripsi bagian {index}/{total}..."
            if total > 1
            else "[*] Mengirim audio ke API..."
        )
        click.echo(status_msg)
        transcripts.append(
            transcriber.transcribe_file(
                file_path=segment_path,
                language=language,
                response_format=response_format,
            )
        )
    return transcripts


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
def info(audio_path: str) -> None:
    """Menampilkan detail dan informasi metadata dari file audio."""
    settings = load_config(CONFIG_PATH)
    processor = AudioProcessor(settings)

    try:
        meta = processor.get_metadata(audio_path)
        size_mb = meta["size_bytes"] / (1024 * 1024)
        click.echo(f"File Path        : {meta['path']}")
        click.echo(f"Format           : {meta['format'].upper()}")
        click.echo(f"Ukuran File      : {size_mb:.2f} MB ({meta['size_bytes']} bytes)")
        click.echo(f"Durasi Estimasi  : {meta['duration_seconds']} detik")
    except Exception as exc:
        logger.error("Failed to read audio metadata for %s: %s", audio_path, exc)
        click.echo(click.style(f"[-] Gagal membaca file: {exc}", fg="red"), err=True)


@cli.command()
@click.argument("audio_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Tampilkan langkah-langkah agent secara detail.")
@click.option("--no-cache", is_flag=True, help="Abaikan cache, paksa analisis ulang.")
def analyze(audio_path: str, verbose: bool, no_cache: bool) -> None:
    """Menganalisis kualitas file audio menggunakan Agent (ffmpeg + LLM).

    Output: ``<filename>_report.json``
    """
    settings = _prepare_analysis_settings(verbose)
    click.echo(click.style("+==========================================+", fg="cyan"))
    click.echo(click.style("|   Audio Analysis Agent (ffmpeg + LLM)    |", fg="cyan"))
    click.echo(click.style("+==========================================+", fg="cyan"))
    click.echo(f"[*] File target: {audio_path}")
    click.echo(f"[*] Model LLM : {os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}")
    click.echo()

    try:
        agent = AudioAnalysisAgent(settings, use_cache=not no_cache)
        click.echo("[*] Memulai loop ReAct Agent...")
        report = agent.analyze(audio_path, verbose=verbose)
        if "error" in report:
            click.echo(click.style(f"[-] Error: {report['error']}", fg="red"), err=True)
            sys.exit(1)

        name_without_ext = os.path.splitext(os.path.basename(audio_path))[0]
        report_dir = os.path.dirname(audio_path) or "."
        report_path = os.path.join(report_dir, f"{name_without_ext}_report.json")
        _save_json_report(report, report_path)
        _display_report_summary(report, report_path)
    except Exception as exc:
        logger.error("Analyze command failed for %s: %s", audio_path, exc)
        click.echo(click.style(f"[-] Terjadi kesalahan: {exc}", fg="red"), err=True)
        sys.exit(1)


@cli.command(name="analyze-batch")
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--verbose", "-v", is_flag=True, help="Tampilkan langkah-langkah agent secara detail.")
@click.option("--no-cache", is_flag=True, help="Abaikan cache, paksa analisis ulang.")
def analyze_batch(directory: str, verbose: bool, no_cache: bool) -> None:
    """Menganalisis semua file audio dalam satu direktori (Batch Processing)."""
    settings = _prepare_analysis_settings(verbose)
    click.echo(click.style("+==========================================+", fg="cyan"))
    click.echo(click.style("|   Batch Audio Analysis (ffmpeg + LLM)    |", fg="cyan"))
    click.echo(click.style("+==========================================+", fg="cyan"))
    click.echo(f"[*] Direktori target: {directory}\n")

    audio_files = _collect_audio_files(directory)
    if not audio_files:
        click.echo(click.style(f"[-] Tidak ditemukan file audio di: {directory}", fg="red"))
        sys.exit(1)

    click.echo(f"[*] Ditemukan {len(audio_files)} file audio:")
    for audio_file in audio_files:
        click.echo(f"    - {os.path.basename(audio_file)}")
    click.echo()

    try:
        agent = AudioAnalysisAgent(settings, use_cache=not no_cache)
        batch_results = [
            _process_batch_file(agent, path, directory, idx, len(audio_files), verbose)
            for idx, path in enumerate(audio_files, 1)
        ]
        master_path = os.path.join(directory, "batch_insights_report.json")
        _save_json_report(batch_results, master_path)
        click.echo(click.style("===========================================", fg="yellow"))
        click.echo(
            click.style(f"[+] Batch selesai! {len(audio_files)} file dianalisis.", fg="green", bold=True)
        )
        click.echo(click.style(f"[+] Master report: {master_path}", fg="green"))
    except Exception as exc:
        logger.error("Batch analysis failed for %s: %s", directory, exc)
        click.echo(click.style(f"[-] Terjadi kesalahan: {exc}", fg="red"), err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
