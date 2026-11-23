.PHONY: install lint format test test-cov health run-mcp docker-build docker-up clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m flake8 --max-line-length=120 .

format:
	ruff format src tests mcp_server.py

test:
	$(PYTHON) -m pytest tests/ -v

test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=src --cov-report=term-missing

health:
	$(PYTHON) -m src.main health

run-mcp:
	$(PYTHON) mcp_server.py

docker-build:
	docker build -t audio-analysis-agent:latest .

docker-up:
	docker compose up --build

clean:
	$(PYTHON) -c "import pathlib, shutil; [shutil.rmtree(p, ignore_errors=True) for p in list(pathlib.Path('.').rglob('__pycache__')) + [pathlib.Path(d) for d in ('.pytest_cache', '.ruff_cache', '.mypy_cache', 'htmlcov', 'dist', 'build')]]"
