# Octocode Shared Documentation

This file consolidates the common installation and setup instructions for the Octocode MCP server. The content originates from several duplicated sections across:
- `docs/others/OCTOCODE_SETUP.md`
- `docs/others/OCTOCODE_INSTALLATION_SUMMARY.md`
- `docs/others/OCTOCODE_QUICK_START.md`

## Overview

Octocode provides an AI‑powered code‑search and analysis MCP server. It integrates with OpenCode agents to enable natural‑language queries across repositories, issue tracking, and local files.

## Installation

```bash
npm install -g octocode-mcp@latest   # or use npx for a one‑off run
```

## Configuration (OpenCode)

```json
"octocode": {
  "type": "stdio",
  "command": "npx",
  "args": ["octocode-mcp@latest"],
  "env": {"ENABLE_LOCAL": "true"},
  "enabled": true
}
```

## Authentication

- Set `GITHUB_TOKEN` (required for private repos).
- Optional `GITLAB_TOKEN`, `BITBUCKET_TOKEN`.

```bash
export GITHUB_TOKEN=ghp_your_token_here
```

## Quick Start

1. Start OpenCode.
2. Verify MCP server: `> mcp list`.
3. Test a search: `> Search GitHub for "Rust async"`.

For full details see the original setup guides.
