.PHONY: test models mac-tools servers-mac dev

SHELL := /bin/bash
MODELS ?= models
VENV := .venv/bin
CONFIG ?= config/mac.toml
STT_MODEL ?= $(MODELS)/ggml-small.bin
LLM_MODEL ?= $(MODELS)/gemma-3-4b-it-Q4_K_M.gguf
WHISPER := whisper-server -m $(STT_MODEL) --port 8081
LLAMA := llama-server -m $(LLM_MODEL) --port 8080 -c 8192 -ngl 99 -np 2 --cache-reuse 256
LOGS ?= /tmp/mirror-dev

test:
	SDL_VIDEODRIVER=dummy $(VENV)/pytest -q

models:
	PYTHON=$(VENV)/python deploy/fetch_models.sh $(MODELS)

mac-tools:
	@command -v whisper-server >/dev/null && command -v llama-server >/dev/null \
		|| brew install whisper-cpp llama.cpp

# Both servers in the foreground; Ctrl-C stops them.
servers-mac: mac-tools
	@pids=""; trap 'kill $$pids 2>/dev/null; wait' EXIT; trap 'exit 130' INT TERM; \
	$(WHISPER) & pids="$$pids $$!"; \
	$(LLAMA) & pids="$$pids $$!"; \
	wait

# Servers in the background (logs in $(LOGS)), wait for both /health endpoints,
# then the face window and the voice loop. Ctrl-C (or voice exiting) stops everything.
dev: mac-tools
	@mkdir -p $(LOGS); pids=""; \
	trap 'kill $$pids 2>/dev/null; wait' EXIT; trap 'exit 130' INT TERM; \
	$(WHISPER) > $(LOGS)/whisper.log 2>&1 & pids="$$pids $$!"; \
	$(LLAMA) > $(LOGS)/llama.log 2>&1 & pids="$$pids $$!"; \
	for url in http://127.0.0.1:8081 http://127.0.0.1:8080; do \
		echo "waiting for $$url ..."; \
		for i in $$(seq 1 240); do \
			curl -sf -o /dev/null $$url/health && break; \
			for p in $$pids; do \
				kill -0 $$p 2>/dev/null || { echo "a server died; see $(LOGS)/*.log"; exit 1; }; \
			done; \
			sleep 0.5; \
		done; \
		curl -sf -o /dev/null $$url/health || { echo "$$url not ready; see $(LOGS)/*.log"; exit 1; }; \
	done; \
	$(VENV)/mirror-face --config $(CONFIG) & pids="$$pids $$!"; \
	$(VENV)/mirror-voice --config $(CONFIG)
