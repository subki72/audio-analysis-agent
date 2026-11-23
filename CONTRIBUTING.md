# Contributing to Audio Analysis Agent

Thank you for contributing to `audio-analysis-agent`. Please follow these engineering standards to keep the codebase reliable, secure, and maintainable.

## Development Setup

1. Clone the repository and create a Python 3.10+ virtual environment (or Conda environment via `environment.yml`).
2. Ensure `ffmpeg` and `ffprobe` are installed and discoverable on `PATH` (or set `FFMPEG_PATH` / `FFPROBE_PATH`).
3. Install pinned dependencies:

   ```bash
   make install
   ```

4. Copy `.env.example` to `.env` and supply your local credentials (never commit `.env`):

   ```bash
   cp .env.example .env
   ```

5. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Coding Standards

- Follow PEP 8 with a 120-character maximum line length (`make lint` must exit `0`).
- Keep functions under 50 lines with control-flow nesting depth `< 3` and explicit type annotations.
- Never silently swallow exceptions (`except: pass`); log actionable context with `logging.getLogger(__name__)`.
- Validate all external file paths and configuration values upfront using guard clauses.
- Never commit API keys, tokens, or real credentials — use placeholder strings like `test_token_placeholder` in unit tests.

## Commit & Pull Request Guidelines

- Use [Conventional Commits](https://www.conventionalcommits.org/): `feat(...)`, `fix(...)`, `refactor(...)`, `test(...)`, `ci(...)`, `ops(...)`, `docs(...)`.
- Add or update unit tests covering happy paths, edge cases, and error paths. All tests must pass completely offline without live Groq API calls:

  ```bash
  make lint
  make test-cov
  ```

- Update `CHANGELOG.md` and `README.md` whenever changing public CLI options, MCP tools, or configuration keys.
