# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in `audio-analysis-agent`:

1. **Do NOT open a public GitHub issue** for undisclosed vulnerabilities.
2. Submit a private report via GitHub Security Advisories (`Security -> Advisories -> Report a vulnerability`).
3. Include affected file paths, line numbers, reproduction steps, and estimated severity.
4. Initial acknowledgement will be provided within **48 hours**.

## Security Controls

- **Secret Hygiene**: `.env`, `.env.*`, and private key files are excluded via `.gitignore`; zero active credentials are stored in source control, and `GROQ_API_KEY` is never logged.
- **Input Validation & Subprocess Safety**: All FFmpeg/FFprobe calls invoke list-based `subprocess.run(..., shell=False)` with explicit timeouts, and file paths are validated against empty or null-byte payloads via `validate_audio_file_path()`.
- **Error Boundary**: Centralized `AudioAnalysisErrorHandler` maps internal exceptions to safe payloads without leaking stack traces to MCP clients.
- **Automated Vulnerability Management**: Weekly Dependabot updates (`semver-major` ignored for build stability), Semgrep SAST (`p/python`), and Trivy container vulnerability scanning in GitHub Actions CI.
- **Container Hardening**: Multi-stage Docker build running as non-root `appuser` (UID `10001`) with a built-in FFmpeg/config `HEALTHCHECK`.
