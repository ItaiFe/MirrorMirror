.PHONY: test servers-mac dev

test:
	SDL_VIDEODRIVER=dummy .venv/bin/pytest -q

servers-mac:
	@echo "implemented in Task 11"

dev:
	@echo "implemented in Task 11"
