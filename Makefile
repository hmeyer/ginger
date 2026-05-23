CARGO ?= cargo

.PHONY: all build check fmt fmt-check clippy audit lint install-hooks deploy

all: build

build:
	$(CARGO) build --bin ginger

check:
	$(CARGO) check

fmt:
	$(CARGO) fmt

fmt-check:
	$(CARGO) fmt --check

clippy:
	$(CARGO) clippy -- -D warnings

audit:
	$(CARGO) audit

# What CI and the pre-commit hook both run
lint: fmt-check clippy

install-hooks:
	cp scripts/pre-commit .git/hooks/pre-commit
	chmod +x .git/hooks/pre-commit
	@echo "pre-commit hook installed"

# Push to GitHub and kick the Pi-side burst pull (see scripts/deploy.sh).
# Extra args go through to `git push`: `make deploy ARGS="--force-with-lease"`.
deploy:
	./scripts/deploy.sh $(ARGS)
