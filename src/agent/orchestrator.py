"""Audio Analysis Agent — ReAct loop with Groq tool calling.

ReAct (Reason + Act) lifecycle:
1. Validate target audio path and check SHA-256 disk cache.
2. Iteratively invoke Groq tool calling across registered FFmpeg tools.
3. Parse and deterministically normalize the final JSON quality report.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import re
from typing import Any

from groq import Groq

from src.config import AppConfig, validate_audio_file_path
from src.health import SimpleRateLimiter
from src.tools import TOOL_REGISTRY, TOOL_SCHEMAS

logger = logging.getLogger(__name__)

SILENCE_ISSUE_THRESHOLD = 0.1
LOW_VOLUME_ISSUE_THRESHOLD_DB = -35.0
MAX_REPORT_PREVIEW_CHARS = 200
MAX_HIGHLIGHTED_SILENCE_SEGMENTS = 3

SYSTEM_PROMPT = """Kamu adalah Audio Analysis Agent yang ahli dalam analisis kualitas audio.
Tugasmu adalah menganalisis file audio secara menyeluruh dan memberikan penilaian profesional.

Kamu memiliki akses ke 4 tools:
1. get_audio_metadata — Mengambil metadata teknis (format, durasi, sample rate, channels, codec, bitrate).
2. detect_silence — Mendeteksi segmen keheningan beserta posisi waktunya dalam audio.
3. detect_clipping — Mengukur level volume dan mendeteksi distorsi/clipping.
4. detect_noise — Menganalisis noise level dan dynamic range audio.

LANGKAH KERJA:
1. Panggil keempat tools tersebut untuk mengumpulkan data teknis secara lengkap.
2. Analisis data yang kamu terima — gunakan judgment profesionalmu untuk menilai
   apakah kondisi audio tersebut bermasalah, mengacu pada standar industri rekaman audio.
3. Hasilkan laporan akhir dalam format JSON berikut (tanpa markdown code fence):

{
  "file_name": "<nama file>",
  "duration_seconds": <durasi dalam detik>,
  "audio_quality": {
    "silence_ratio": <rasio 0.0–1.0>,
    "clipping_detected": <true/false>,
    "avg_volume_db": <angka dB>
  },
  "issues": [
    "<deskripsi masalah spesifik dalam Bahasa Inggris, sertakan angka dan timestamp jika relevan>"
  ],
  "llm_insights": {
    "summary": "<analisis kondisi kualitas audio secara keseluruhan dalam Bahasa Indonesia>",
    "recommended_actions": ["<aksi konkret 1>", "<aksi konkret 2>"]
  }
}

PANDUAN ANALISIS:
- Nilai setiap metrik berdasarkan konteks penggunaan audio (rekaman deposisi hukum).
- Untuk field "issues": tulis deskripsi spesifik yang mencantumkan nilai aktual dan,
  jika ada data segmen, sertakan rentang waktu (contoh: "silence detected at 12.5s–45.2s").
- Jika audio tidak memiliki masalah signifikan, kembalikan array "issues" kosong: [].
- Tulis "summary" dan "recommended_actions" dalam Bahasa Indonesia yang profesional.
- Semua angka di "audio_quality" HARUS berasal dari data tool — jangan estimasi.
"""


def _strip_markdown_fences(content: str) -> str:
    """Remove surrounding Markdown code fences from an LLM JSON response."""
    cleaned = content.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    return cleaned.strip()


def _extract_json_object(cleaned: str) -> dict[str, Any] | None:
    """Parse a JSON object directly or from the largest braced block in text."""
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError as direct_err:
        logger.debug("Direct JSON decode failed (%s); attempting regex extraction.", direct_err)

    json_match = re.search(r"\{[\s\S]*\}", cleaned)
    if not json_match:
        logger.warning("No JSON object found in LLM response payload.")
        return None

    try:
        parsed = json.loads(json_match.group())
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError as fallback_err:
        logger.warning("Failed to parse extracted JSON block from LLM response: %s", fallback_err)
        return None


def _build_parse_fallback_report(raw_content: str) -> dict[str, Any]:
    """Build a schema-compliant fallback report when LLM output cannot be parsed."""
    return {
        "file_name": "unknown",
        "duration_seconds": 0,
        "audio_quality": {
            "silence_ratio": 0,
            "clipping_detected": False,
            "avg_volume_db": 0,
        },
        "issues": [],
        "llm_insights": {
            "summary": raw_content,
            "recommended_actions": ["Gagal mem-parsing output JSON dari LLM."],
            "_raw_response": raw_content,
        },
    }


def _resolve_audio_quality(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize quality metrics across current ``audio_quality`` and legacy ``audio_metrics``."""
    aq = raw.get("audio_quality", {})
    am = raw.get("audio_metrics", {})
    return {
        "silence_ratio": aq.get("silence_ratio", am.get("silence_ratio", 0)),
        "clipping_detected": aq.get(
            "clipping_detected",
            am.get("volume_clipping", am.get("clipping_detected", False)),
        ),
        "avg_volume_db": aq.get(
            "avg_volume_db",
            am.get("avg_volume_db", am.get("mean_volume_db", 0)),
        ),
    }


def _format_longest_silence_segments(silence_segments: list[dict[str, Any]]) -> str:
    """Format up to three longest silent segments as a human-readable summary string."""
    longest = sorted(
        silence_segments,
        key=lambda seg: seg.get("duration", 0),
        reverse=True,
    )[:MAX_HIGHLIGHTED_SILENCE_SEGMENTS]
    formatted = [
        f"{seg['start']:.1f}s\u2013{seg['end']:.1f}s ({seg['duration']:.1f}s)"
        for seg in longest
        if all(key in seg for key in ("start", "end", "duration"))
    ]
    return ", ".join(formatted)


def _build_fallback_issues(
    raw: dict[str, Any],
    tool_results: dict[str, Any],
    quality: dict[str, Any],
) -> list[str]:
    """Synthesize deterministic audio issues when the LLM omits the ``issues`` list."""
    issues: list[str] = []
    silence_ratio = quality["silence_ratio"]
    if silence_ratio and silence_ratio > SILENCE_ISSUE_THRESHOLD:
        pct = round(silence_ratio * 100, 1)
        base_issue = f"High silence ratio ({pct}%) - audio contains significant silent segments"
        silence_segments = (
            raw.get("silence_segments")
            or tool_results.get("detect_silence", {}).get("silence_segments", [])
        )
        segment_summary = _format_longest_silence_segments(silence_segments)
        if segment_summary:
            base_issue += f". Longest silent segments: {segment_summary}"
        issues.append(base_issue)

    if quality["clipping_detected"]:
        am = raw.get("audio_metrics", {})
        aq = raw.get("audio_quality", {})
        max_vol = am.get("max_volume_db", aq.get("max_volume_db", "N/A"))
        issues.append(f"Audio clipping detected (max volume: {max_vol} dB)")

    avg_vol = quality["avg_volume_db"]
    if avg_vol and avg_vol < LOW_VOLUME_ISSUE_THRESHOLD_DB:
        issues.append(f"Unusually low average volume ({avg_vol} dB)")

    return issues


class AudioAnalysisAgent:
    """ReAct agent that orchestrates FFmpeg tools via Groq tool calling."""

    def __init__(
        self,
        settings: dict[str, Any] | None = None,
        use_cache: bool = True,
        *,
        config: AppConfig | None = None,
        client: Any | None = None,
        rate_limiter: SimpleRateLimiter | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self.settings = settings or {}
        self.config = config if config is not None else AppConfig.from_env(self.settings)
        self.config.validate(require_api_key=client is None)

        self.api_key = self.config.groq_api_key
        self.model = self.config.groq_model
        self.api_timeout = self.config.api_timeout_seconds
        self.max_iterations = self.config.max_iterations
        self.noise_threshold = self.config.noise_threshold
        self.min_silence_duration = self.config.min_silence_duration
        self.clipping_threshold_db = self.config.clipping_threshold_db
        self.low_volume_threshold_db = self.config.low_volume_threshold_db

        self.client = client if client is not None else self._build_client()
        self.rate_limiter = rate_limiter or SimpleRateLimiter(
            max_calls=self.config.rate_limit_calls_per_minute
        )
        self.use_cache = use_cache
        self._cache_dir = cache_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            ".cache",
        )
        self._current_tool_results: dict[str, Any] = {}

        # Hashed host identifier for Groq abuse-prevention tracking without PII.
        raw_id = f"voicescript-{platform.node()}"
        self.user_id = hashlib.sha256(raw_id.encode()).hexdigest()[:16]

    def _build_client(self) -> Groq:
        """Instantiate the Groq SDK client with explicit request timeout."""
        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "timeout": self.api_timeout,
        }
        if self.config.groq_base_url:
            kwargs["base_url"] = self.config.groq_base_url
        return Groq(**kwargs)

    def _compute_cache_path(self, file_path: str) -> str | None:
        """Compute deterministic SHA-256 cache file path for an input audio file."""
        if not self.use_cache:
            return None
        try:
            os.makedirs(self._cache_dir, exist_ok=True)
            with open(file_path, "rb") as audio_file:
                file_hash = hashlib.sha256(audio_file.read()).hexdigest()
            return os.path.join(self._cache_dir, f"{file_hash}_report.json")
        except OSError as exc:
            logger.warning("Unable to prepare analysis cache for %s: %s", file_path, exc)
            return None

    def _load_cached_report(self, cache_path: str | None, verbose: bool) -> dict[str, Any] | None:
        """Load and return a cached JSON report when available and readable."""
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            with open(cache_path, "r", encoding="utf-8") as cache_file:
                cached = json.load(cache_file)
            log_fn = logger.info if verbose else logger.debug
            log_fn("[Cache] Hit! Loaded stored analysis report from %s", cache_path)
            return cached
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Failed to read cached report at %s: %s", cache_path, exc)
            return None

    def _save_cached_report(self, cache_path: str | None, result: dict[str, Any], verbose: bool) -> None:
        """Persist a completed analysis report to the local SHA-256 cache."""
        if not cache_path or "error" in result:
            return
        try:
            with open(cache_path, "w", encoding="utf-8") as cache_file:
                json.dump(result, cache_file, ensure_ascii=False, indent=2)
            log_fn = logger.info if verbose else logger.debug
            log_fn("[Cache] Stored analysis report at %s", cache_path)
        except OSError as exc:
            logger.warning("Failed to write analysis cache at %s: %s", cache_path, exc)

    def _build_initial_messages(self, file_path: str) -> list[dict[str, Any]]:
        """Construct the opening system and user prompt messages for the ReAct loop."""
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Analisis file audio di path berikut: {file_path}\n"
                    "Ekstrak metadata, deteksi masalah kualitas (silence, clipping, low volume), "
                    "dan hasilkan laporan JSON akhir."
                ),
            },
        ]

    def _request_llm_completion(self, messages: list[dict[str, Any]]) -> Any:
        """Apply rate limiting and request the next completion turn from Groq."""
        self.rate_limiter.acquire()
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=TOOL_SCHEMAS,
            tool_choice="auto",
            temperature=0.1,
            max_tokens=4096,
            user=self.user_id,
        )

    def _inject_tool_defaults(self, fn_name: str, fn_args: dict[str, Any]) -> dict[str, Any]:
        """Fill configured analysis thresholds when omitted by the LLM tool call."""
        enriched = dict(fn_args)
        if fn_name == "detect_silence":
            enriched.setdefault("noise_threshold", self.noise_threshold)
            enriched.setdefault("min_duration", self.min_silence_duration)
        elif fn_name == "detect_clipping":
            enriched.setdefault("clipping_threshold_db", self.clipping_threshold_db)
            enriched.setdefault("low_volume_threshold_db", self.low_volume_threshold_db)
        return enriched

    def _execute_single_tool(self, tool_call: Any, verbose: bool) -> dict[str, Any]:
        """Parse arguments, execute one registered FFmpeg tool, and return tool message."""
        fn_name = tool_call.function.name
        try:
            raw_args = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as exc:
            logger.warning("Invalid JSON arguments for tool %s: %s", fn_name, exc)
            tool_result: dict[str, Any] = {"error": f"Argumen tool tidak valid: {exc}"}
        else:
            fn_args = self._inject_tool_defaults(fn_name, raw_args)
            log_fn = logger.info if verbose else logger.debug
            log_fn("[Tool Call] %s(%s)", fn_name, json.dumps(fn_args, ensure_ascii=False))
            if fn_name in TOOL_REGISTRY:
                tool_result = TOOL_REGISTRY[fn_name](**fn_args)
                self._current_tool_results[fn_name] = tool_result
            else:
                logger.warning("Unknown tool requested by LLM: %s", fn_name)
                tool_result = {"error": f"Tool tidak dikenal: {fn_name}"}

        result_str = json.dumps(tool_result, ensure_ascii=False, indent=2)
        if verbose:
            preview = (
                f"{result_str[:MAX_REPORT_PREVIEW_CHARS]}..."
                if len(result_str) > MAX_REPORT_PREVIEW_CHARS
                else result_str
            )
            logger.info("[Result] %s", preview)

        return {"role": "tool", "tool_call_id": tool_call.id, "content": result_str}

    def _handle_tool_calls(
        self,
        assistant_message: Any,
        messages: list[dict[str, Any]],
        verbose: bool,
    ) -> None:
        """Append assistant tool call message and all corresponding tool execution results."""
        messages.append(
            {
                "role": "assistant",
                "content": assistant_message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in assistant_message.tool_calls
                ],
            }
        )
        for tool_call in assistant_message.tool_calls:
            messages.append(self._execute_single_tool(tool_call, verbose))

    def analyze(self, file_path: str, verbose: bool = False) -> dict[str, Any]:
        """Run the ReAct loop to analyze a single audio file.

        Args:
            file_path: Path to the target audio file.
            verbose: Emit step-by-step agent progress at INFO log level when True.

        Returns:
            Normalized JSON analysis report dictionary, or ``{"error": ...}``.
        """
        try:
            validated_path = str(validate_audio_file_path(file_path, must_exist=True))
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("Audio analysis rejected invalid path '%s': %s", file_path, exc)
            return {"error": str(exc)}

        cache_path = self._compute_cache_path(validated_path)
        cached_report = self._load_cached_report(cache_path, verbose)
        if cached_report is not None:
            return cached_report

        self._current_tool_results = {}
        messages = self._build_initial_messages(validated_path)
        step_log = logger.info if verbose else logger.debug

        for iteration in range(self.max_iterations):
            step_log("[Agent] Iteration %d/%d", iteration + 1, self.max_iterations)
            try:
                response = self._request_llm_completion(messages)
            except Exception as exc:
                logger.error("[Agent] Groq API call failed on iteration %d: %s", iteration + 1, exc)
                return {"error": f"Gagal memanggil API: {exc}"}

            assistant_message = response.choices[0].message
            if assistant_message.tool_calls:
                self._handle_tool_calls(assistant_message, messages, verbose)
                continue

            step_log("[Agent] Synthesizing final JSON report")
            result = self._parse_final_report(assistant_message.content or "")
            self._save_cached_report(cache_path, result, verbose)
            return result

        logger.error("Agent reached max_iterations=%d without final report", self.max_iterations)
        return {"error": "Agent mencapai batas iterasi maksimum tanpa menghasilkan laporan."}

    def _parse_final_report(self, content: str) -> dict[str, Any]:
        """Parse raw LLM completion text into a normalized assessment dictionary."""
        cleaned = _strip_markdown_fences(content)
        parsed = _extract_json_object(cleaned)
        if parsed is None:
            return _build_parse_fallback_report(content)
        return self._normalize_schema(parsed, self._current_tool_results)

    def _normalize_schema(
        self,
        raw: dict[str, Any],
        tool_results: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Deterministically normalize LLM output into the target assessment schema.

        Supports both legacy (``file_metadata``/``audio_metrics``) and current
        (``file_name``/``audio_quality``/``issues``) payloads.

        Args:
            raw: Parsed JSON dictionary produced by the LLM.
            tool_results: Raw FFmpeg tool outputs captured during the ReAct loop.

        Returns:
            Normalized report dictionary.
        """
        resolved_tools = tool_results or {}
        file_meta = raw.get("file_metadata", {})

        file_name = raw.get("file_name") or file_meta.get(
            "filename", file_meta.get("file_name", "unknown")
        )
        duration = raw.get("duration_seconds")
        if duration is None:
            duration = file_meta.get("duration_seconds", 0)

        quality = _resolve_audio_quality(raw)
        issues = raw.get("issues")
        if issues is None:
            issues = _build_fallback_issues(raw, resolved_tools, quality)

        insights = raw.get("llm_insights", {})
        return {
            "file_name": file_name,
            "duration_seconds": duration,
            "audio_quality": quality,
            "issues": issues,
            "llm_insights": {
                "summary": insights.get("summary", ""),
                "recommended_actions": insights.get("recommended_actions", []),
            },
        }
