.PHONY: lint fix format type-check security test test-cov clean install all \
	a1-smoke a1-run a1-run-prod a1-validate a1-judge a1-anchor a1-distill \
	a1-augment a1-assemble

lint:
	ruff check .
	ruff format --check .

fix:
	ruff check . --fix
	ruff format .

format:
	ruff format .

type-check:
	ty check src/ || mypy src/ --ignore-missing-imports

security:
	# chromadb CVE-2026-45830/31/33 have no patched release yet (<= 1.5.9 all
	# affected); embedded-only usage makes them tolerable. Drop when fixed.
	pip-audit --ignore-vuln CVE-2026-28684 --ignore-vuln CVE-2026-3219 --ignore-vuln CVE-2026-45830 --ignore-vuln CVE-2026-45831 --ignore-vuln CVE-2026-45833
	bandit -r src/ -c pyproject.toml -ll || bandit -r src/ -ll

test:
	pytest --cov --cov-report=term-missing --cov-fail-under=80 -q

test-cov:
	pytest --cov --cov-report=term-missing --cov-report=html

clean:
	rm -rf build/ dist/ *.egg-info htmlcov/ .coverage .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

install:
	uv tool install -e .

all: lint type-check security test

# ---------------------------------------------------------------------------
# Phase A1 — synthetic tool-calling training data pipeline
# ---------------------------------------------------------------------------

PY ?= python
A1_DATA ?= experiments/phase_a1/data
A1_OUT ?= $(A1_DATA)/phase_a1_smoke.jsonl
A1_PROD ?= $(A1_DATA)/phase_a1_full.jsonl
A1_ANCHOR ?= $(A1_DATA)/anchor_opus_50.jsonl
A1_VERDICTS ?= $(A1_DATA)/judge_verdicts.jsonl
A1_DISTILL ?= $(A1_DATA)/phase_a1_swesmith_distilled.jsonl
A1_SWESMITH_SRC ?= ../ml-lab/experiments/2026-04-godspeed-coder/data/phase2_swesmith.jsonl
A1_AUGMENT ?= $(A1_DATA)/phase_a1_augmented.jsonl
A1_FINAL ?= $(A1_DATA)/phase_a1_final.jsonl

a1-smoke:
	$(PY) -m experiments.phase_a1.orchestrate --limit 3 --reset --output $(A1_OUT)

# Free-tier basic run: 6,200 specs, validate-gated only. ~15-25h wall, $0.
# Resumable across midnight UTC quota resets.
a1-run:
	$(PY) -m experiments.phase_a1.orchestrate \
		--total 6200 --limit 6200 --concurrency 2 \
		--output $(A1_PROD)

# Production-quality run: validate + judge gates + 1 anchor blueprint
# few-shot per spec. Drops 15-30% of generated samples but the surviving
# corpus is the highest-quality we can produce on free-tier.
# Adds ~1 judge call per spec (~1s on Z.ai flash). Slower wall-time.
a1-run-prod:
	$(PY) -m experiments.phase_a1.orchestrate \
		--total 6200 --limit 6200 --concurrency 2 \
		--judge --blueprint-few-shots 1 \
		--resume \
		--anchor $(A1_ANCHOR) \
		--output $(A1_PROD)

a1-validate:
	$(PY) -m experiments.phase_a1.validate --input $(A1_OUT) --min-coverage 0

a1-judge:
	$(PY) -m experiments.phase_a1.judge --input $(A1_OUT) --output $(A1_VERDICTS) --anchor $(A1_ANCHOR)

a1-anchor:
	$(PY) -m experiments.phase_a1.anchor_opus --output $(A1_ANCHOR)

a1-distill:
	$(PY) -m experiments.phase_a1.swesmith_distill \
		--input $(A1_SWESMITH_SRC) --output $(A1_DISTILL) \
		--target 1500 --shell-cap 800 --k 30 --seed 42

a1-augment:
	$(PY) -m experiments.phase_a1.augment --output $(A1_AUGMENT) --total 200 --seed 42

# Final assembly: merge anchor + augment + synthetic + distill, validate,
# dedup, shuffle. Run AFTER a1-run / a1-run-prod completes.
a1-assemble:
	$(PY) -m experiments.phase_a1.assemble --output $(A1_FINAL) --seed 42

# Capped assembly: distill source limited to 80 records per primary tool.
# Defends against single-source domination of tool coverage (RESEARCH_LOG F1).
# Smaller corpus, much healthier per-tool / per-category distribution.
A1_DISTILL_CAP ?= 80
a1-assemble-prod:
	$(PY) -m experiments.phase_a1.assemble \
		--output $(A1_FINAL) --seed 42 \
		--distill-per-tool-cap $(A1_DISTILL_CAP)
