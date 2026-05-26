# OpenCode Setup Index

This directory contains comprehensive setup documentation for your OpenCode environment.

## 📚 Documentation Files

### 1. oh-my-opencode-slim (Agent Orchestration)
- **OH_MY_OPENCODE_SLIM_SETUP.md** - Complete setup guide with all details
- **OH_MY_OPENCODE_SLIM_QUICK_START.md** - Quick reference for common tasks

**What it is:** Multi-agent orchestration system that automatically delegates tasks to specialized agents (Explorer, Oracle, Librarian, Designer, Fixer, Council)

**Key Features:**
- 7 specialist agents
- Automatic task delegation
- 15 authenticated providers
- 587 available models
- 2 presets (OpenCode and OpenRouter)

### 2. rust-mcp-filesystem (Filesystem MCP Server)
- **RUST_MCP_FILESYSTEM_SETUP.md** - Comprehensive setup guide
- **RUST_MCP_FILESYSTEM_QUICK_START.md** - Quick reference
- **RUST_MCP_FILESYSTEM_INSTALLATION_SUMMARY.md** - Installation details

**What it is:** Blazingly fast, asynchronous MCP server for filesystem operations written in Rust

**Key Features:**
- <100ms startup time
- No external dependencies
- Read-only by default (safe)
- Advanced glob pattern matching
- ZIP archive support
- SHA256 hashing
- MIME type detection

## 🚀 Quick Start

### Start OpenCode
```bash
opencode
```

### Verify Setup
```
> ping all agents          # Check agent orchestration
> mcp list                 # Check MCP servers
```

### Use Agents
```
> @explorer Find all .rs files
> @oracle Should we refactor this?
> @librarian Show me Next.js patterns
```

### Use Filesystem Tools
```
> List all Rust files in the project
> Find files matching *.rs pattern
> Search for TODO comments
```

## 📋 Configuration Files

### OpenCode Main Config
**Location:** `~/.config/opencode/opencode.json`

Contains:
- Plugin configuration (oh-my-opencode-slim)
- MCP server definitions (including rust-mcp-filesystem)
- Permission settings
- Model preferences

### oh-my-opencode-slim Config
**Location:** `~/.config/opencode/oh-my-opencode-slim.json`

Contains:
- Active preset (opencode or openrouter)
- Agent model assignments
- Skill and MCP permissions per agent

## 🎯 Your Setup

### Installed Components
✅ oh-my-opencode-slim v2 (agent orchestration)
✅ rust-mcp-filesystem v0.4.2 (filesystem MCP)
✅ 15 authenticated providers
✅ 587 available models

### Security Settings
✅ Read-only filesystem access (no write by default)
✅ Limited to specified directories
✅ Type-safe Rust implementation
✅ All tools enabled (safe for read operations)

### Performance
⚡ Agent startup: <100ms
⚡ Filesystem operations: Instant
⚡ Async I/O: Native Rust
⚡ Memory: Minimal

## 🔧 Common Tasks

### Change Active Preset
```
/preset openrouter    # Switch to OpenRouter models
/preset opencode      # Switch back to OpenCode models
```

### Enable Write Access
Edit `~/.config/opencode/opencode.json`:
```json
"env": {
  "ALLOW_WRITE": "true"
}
```

### Add More Directories
Edit `~/.config/opencode/opencode.json`:
```json
"args": [
  "/home/user/Desktop/llama-rs",
  "/home/user",
  "/tmp"
]
```

### Pin Session Goal
```
/goal Implement user authentication
```

### Run Bounded Work
```
/subtask Analyze test coverage
```

## 📚 Resources

### oh-my-opencode-slim
- GitHub: https://github.com/alvinunreal/oh-my-opencode-slim
- Docs: https://github.com/alvinunreal/oh-my-opencode-slim/blob/master/docs/configuration.md

### rust-mcp-filesystem
- GitHub: https://github.com/rust-mcp-stack/rust-mcp-filesystem
- Docs: https://rust-mcp-stack.github.io/rust-mcp-filesystem
- Docker: https://hub.docker.com/mcp/server/rust-mcp-filesystem

## 🆘 Troubleshooting

### Agents Not Responding
```bash
opencode auth list          # Check authentication
opencode models --refresh   # Refresh available models
```

### MCP Not Loading
```bash
# Check binary
~/.rust-mcp-stack/bin/rust-mcp-filesystem --version

# Check config
grep rust-mcp-filesystem ~/.config/opencode/opencode.json
```

### Permission Issues
```bash
chmod +x ~/.rust-mcp-stack/bin/rust-mcp-filesystem
```

## 📝 Next Steps

1. **Start OpenCode:** `opencode`
2. **Verify setup:** `ping all agents` and `mcp list`
3. **Try a task:** Ask the orchestrator something complex
4. **Explore agents:** Use `@explorer`, `@oracle`, `@librarian`
5. **Customize:** Adjust models and settings in config files

## 📊 System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      OpenCode                               │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  oh-my-opencode-slim (Agent Orchestration)           │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • Orchestrator (Master delegator)                   │  │
│  │  • Explorer (Codebase scout)                         │  │
│  │  • Oracle (Strategic advisor)                        │  │
│  │  • Librarian (Knowledge weaver)                      │  │
│  │  • Designer (UI/UX guardian)                         │  │
│  │  • Fixer (Fast builder)                              │  │
│  │  • Council (Multi-model consensus)                   │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MCP Servers                                         │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • rust-mcp-filesystem (Blazingly fast)              │  │
│  │  • filesystem (JavaScript version)                   │  │
│  │  • codegraph (Code analysis)                         │  │
│  │  • octocode (GitHub integration)                     │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Providers (15 authenticated)                        │  │
│  ├──────────────────────────────────────────────────────┤  │
│  │  • OpenCode (Active)                                 │  │
│  │  • OpenRouter                                        │  │
│  │  • DeepSeek                                          │  │
│  │  • GitHub Copilot                                    │  │
│  │  • And more...                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

**Setup Date:** 2026-05-26  
**Status:** ✅ Ready to use  
**Last Updated:** 2026-05-26
