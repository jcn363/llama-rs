---
name: workflow-automation
description: "Use when automating repetitive development tasks, setting up multi-step build processes, standardizing team onboarding, configuring Git hooks, writing deployment scripts, or setting up CI/CD pipelines - covers npm scripts, Makefiles, Husky pre-commit hooks, custom shell scripts for setup/deploy, and GitHub Actions workflow templates."
---

# Workflow Automation

## Overview

Automate development tasks with npm scripts, Makefiles, Git hooks, and shell scripts — covering setup, testing, linting, deployment, and CI/CD pipelines with idempotency and error handling.

## When to Use

- **Repetitive tasks**: Running the same commands every time
- **Complex builds**: Multi-step build processes
- **Team onboarding**: Consistent development environment setup

## Core Workflow

### Step 1: npm Scripts

Centralize common commands in `package.json`:

```json
{
  "scripts": {
    "dev": "nodemon src/index.ts",
    "build": "tsc && vite build",
    "test": "jest --coverage",
    "test:watch": "jest --watch",
    "lint": "eslint src --ext .ts,.tsx",
    "lint:fix": "eslint src --ext .ts,.tsx --fix",
    "format": "prettier --write \"src/**/*.{ts,tsx,json}\"",
    "type-check": "tsc --noEmit",
    "pre-commit": "lint-staged",
    "prepare": "husky install",
    "clean": "rm -rf dist node_modules",
    "reset": "npm run clean && npm install",
    "docker:build": "docker build -t myapp .",
    "docker:run": "docker run -p 3000:3000 myapp"
  }
}
```

### Step 2: Makefile

Platform-agnostic interface wrapping npm scripts with help:

```makefile
.PHONY: help install dev build test clean docker

.DEFAULT_GOAL := help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	npm install

dev: ## Start development server
	npm run dev

build: ## Build for production
	npm run build

test: ## Run all tests
	npm test

lint: ## Run linter
	npm run lint

lint-fix: ## Fix linting issues
	npm run lint:fix

clean: ## Clean build artifacts
	rm -rf dist coverage

docker-build: ## Build Docker image
	docker build -t myapp:latest .

docker-run: ## Run Docker container
	docker run -d -p 3000:3000 --name myapp myapp:latest

deploy: build ## Deploy to production (build first)
	@echo "Deploying to production..."
	./scripts/deploy.sh production

ci: lint test build ## Run CI pipeline locally
	@echo "✅ CI pipeline passed!"
```

**Usage:**
```bash
make help        # Show all commands
make dev         # Start development
make ci          # Run full CI locally
```

### Step 3: Husky + lint-staged (Git Hooks)

**`package.json` (lint-staged config):**

```json
{
  "lint-staged": {
    "*.{ts,tsx}": ["eslint --fix", "prettier --write"],
    "*.{json,md}": ["prettier --write"]
  }
}
```

**`.husky/pre-commit`:**

```bash
#!/usr/bin/env sh
. "$(dirname -- "$0")/_/husky.sh"

echo "Running pre-commit checks..."

# Lint staged files
npx lint-staged

# Type check
npm run type-check

# Run tests related to changed files
npm test -- --onlyChanged

echo "✅ Pre-commit checks passed!"
```

### Step 4: Custom Shell Scripts

**`scripts/dev-setup.sh` — idempotent environment setup:**

```bash
#!/bin/bash
set -e

echo "🚀 Setting up development environment..."

# Check prerequisites
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed"
    exit 1
fi

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Copy environment file (idempotent: skip if exists)
if [ ! -f .env ]; then
    echo "📄 Creating .env file..."
    cp .env.example .env
    echo "⚠️ Please update .env with your configuration"
fi

# Start Docker services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for database
echo "⏳ Waiting for database..."
./scripts/wait-for-it.sh localhost:5432 --timeout=30

# Run migrations
echo "🗄️ Running database migrations..."
npm run migrate

# Seed data (optional, with confirmation)
read -p "Seed database with sample data? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    npm run seed
fi

echo "✅ Development environment ready!"
echo "Run 'make dev' to start the development server"
```

**`scripts/deploy.sh` — staged deployment:**

```bash
#!/bin/bash
set -e

ENV=$1

if [ -z "$ENV" ]; then
    echo "Usage: ./deploy.sh [staging|production]"
    exit 1
fi

echo "🚀 Deploying to $ENV..."

# Build
echo "📦 Building application..."
npm run build

# Run tests
echo "🧪 Running tests..."
npm test

# Deploy based on environment
if [ "$ENV" == "production" ]; then
    echo "🌍 Deploying to production..."
    ssh production "cd /app && git pull && npm install && npm run build && pm2 restart all"
elif [ "$ENV" == "staging" ]; then
    echo "🧪 Deploying to staging..."
    ssh staging "cd /app && git pull && npm install && npm run build && pm2 restart all"
fi

echo "✅ Deployment to $ENV completed!"
```

### Step 5: GitHub Actions CI/CD

**`.github/workflows/ci.yml`:**

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'

      - name: Install dependencies
        run: npm ci

      - name: Run linter
        run: npm run lint

      - name: Type check
        run: npm run type-check

      - name: Run tests
        run: npm test -- --coverage

      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Output Format: Directory Structure

```
project/
├── scripts/
│   ├── dev-setup.sh       # Environment setup
│   ├── deploy.sh           # Deployment
│   ├── test.sh             # Test runner
│   └── cleanup.sh          # Cleanup
├── Makefile                # Platform-agnostic interface
├── package.json            # npm scripts + lint-staged config
└── .husky/
    ├── pre-commit          # Lint + type-check + test
    └── pre-push            # Full CI check
```

## Constraints

### Required (MUST)
- **Idempotency**: Scripts must be safe to run multiple times — check preconditions, skip completed steps
- **Error handling**: Clear messages on failure; use `set -e` for early exit; exit with meaningful codes
- **Documentation**: Comments explaining how to use each script; `make help` as default goal

### Prohibited (MUST NOT)
- **Hardcoded secrets**: Never include passwords, API keys, or tokens in scripts — use environment variables or secrets managers
- **Destructive commands without confirmation**: `rm -rf`, `DROP TABLE`, etc. must prompt or accept explicit confirmation flags

## Best Practices

- **Use Make**: Platform-agnostic entry point that wraps npm scripts and shell scripts
- **Git Hooks**: Automate quality gates at commit/push time (lint → type-check → test)
- **CI/CD**: Run the same pipeline in CI (GitHub Actions) as developers run locally with `make ci`
- **Fail fast**: Check prerequisites at the top of setup scripts before doing work
- **Idempotent setup**: `dev-setup.sh` should be safe to run on an already-configured machine

## Common Mistakes

| Mistake | Correction |
|---------|-----------|
| Hardcoded secrets in scripts | Use `$SECRET` env vars or CI secrets |
| `rm -rf` without confirmation | Add a `-f` flag check or `read -p` prompt |
| Chaining commands with `;` instead of `&&` | Use `&&` so failures stop the chain |
| No `set -e` in shell scripts | Add `set -e` at the top to fail on first error |
| Skipping prerequisites check | Check for `node`, `docker`, `python` before proceeding |
| Makefile without help target | Add `.DEFAULT_GOAL := help` with grep/awk help output |

## Related Skills

- **code-review**: Review automation scripts for correctness and security
- **security-best-practices**: Secret management and injection prevention in scripts

## References

- [npm scripts documentation](https://docs.npmjs.com/cli/v10/using-npm/scripts)
- [Make tutorial (GNU)](https://www.gnu.org/software/make/manual/)
- [Husky](https://typicode.github.io/husky/)
- [GitHub Actions workflow syntax](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
