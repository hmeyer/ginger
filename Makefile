CARGO ?= cargo

.PHONY: all build check fmt fmt-check clippy audit lint install-hooks

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
