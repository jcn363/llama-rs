# oh-my-opencode-slim Quick Reference

## ✅ Installation Complete

Your oh-my-opencode-slim agent orchestration system is now installed and ready to use.

### What You Have

- **7 Specialist Agents** ready to delegate work
- **15 Authenticated Providers** (OpenCode, OpenRouter, DeepSeek, etc.)
- **587 Available Models** across all providers
- **4 Built-in Skills** (agent-browser, simplify, codemap, clonedeps)
- **2 Presets** (opencode active, openrouter available)

---

## 🚀 Getting Started

### 1. Start OpenCode
```bash
opencode
```

### 2. Verify Agents Are Online
```
ping all agents
```

You should see all 7 agents respond with their status.

### 3. Try Automatic Delegation
Ask the orchestrator a complex task and watch it delegate:
```
Help me refactor this function for better performance
```

The orchestrator will automatically route to specialists:
- **Explorer** for codebase analysis
- **Oracle** for architecture advice
- **Fixer** for implementation

---

## 👥 Your Agent Team

| Agent | Best For | Trigger |
|-------|----------|---------|
| **Orchestrator** | Main coordinator, complex tasks | Default (no prefix) |
| **Explorer** | Finding files, patterns, codebase structure | `@explorer` |
| **Oracle** | Architecture, debugging, code review | `@oracle` |
| **Librarian** | Documentation, API research, examples | `@librarian` |
| **Designer** | UI/UX, frontend, visual polish | `@designer` |
| **Fixer** | Fast implementation, tests, scoped work | `@fixer` |
| **Council** | Multi-model consensus on hard decisions | `@council` |

---

## 💡 Common Workflows

### Explore a Codebase
```
@explorer Find all database-related files and summarize the schema
```

### Get Architecture Advice
```
@oracle Should we refactor this monolith into microservices?
```

### Research a Library
```
@librarian Show me the latest Next.js App Router patterns
```

### Implement a Feature
```
@fixer Implement the login form based on this spec
```

### Get Multiple Perspectives
```
@council Compare these two API design approaches
```

---

## ⚙️ Configuration

### View Current Config
```bash
cat ~/.config/opencode/oh-my-opencode-slim.json
```

### Switch Presets at Runtime
```
/preset openrouter    # Switch to OpenRouter models
/preset opencode      # Switch back to OpenCode models
```

### Customize Agent Models
Edit `~/.config/opencode/oh-my-opencode-slim.json`:

```json
{
  "preset": "opencode",
  "presets": {
    "opencode": {
      "orchestrator": {
        "model": "opencode/claude-opus-4-7"  // Change this
      }
    }
  }
}
```

### Disable Specific Agents
```json
{
  "disabled_agents": ["observer"]  // Disable observer
}
```

---

## 🎯 Advanced Features

### Pin a Session Goal
```
/goal Implement user authentication with JWT
```
All agents will stay aligned with this objective.

### Run Bounded Work
```
/subtask Analyze the test coverage and report gaps
```
Returns a structured summary without cluttering main context.

### Auto-Continue Sessions
```
/auto-continue on
```
Orchestrator automatically continues through todos with cooldowns.

### Session Management
```
/session list          # List recent sessions
/session resume <id>   # Resume a previous session
```

---

## 📊 Model Recommendations

### For Best Quality (Higher Cost)
```json
"orchestrator": { "model": "opencode/claude-opus-4-7" }
"oracle": { "model": "opencode/claude-opus-4-7" }
```

### For Balanced Performance
```json
"orchestrator": { "model": "opencode/gpt-5.5" }
"oracle": { "model": "opencode/gpt-5.5" }
```

### For Speed & Cost (Lower Quality)
```json
"orchestrator": { "model": "opencode/gpt-5.4-mini" }
"explorer": { "model": "opencode/gpt-5.4-mini" }
```

---

## 🔍 Troubleshooting

### Agents Not Responding
```bash
opencode auth list          # Check authentication
opencode models --refresh   # Refresh available models
```

### Reset Configuration
```bash
bunx oh-my-opencode-slim@latest install --reset
```

### Check Plugin Status
```bash
grep "oh-my-opencode-slim" ~/.config/opencode/opencode.json
```

### View Logs
```bash
opencode --log-level DEBUG
```

---

## 📚 Documentation

- **Full Config Reference:** https://github.com/alvinunreal/oh-my-opencode-slim/blob/master/docs/configuration.md
- **Installation Guide:** https://github.com/alvinunreal/oh-my-opencode-slim/blob/master/docs/installation.md
- **Council Usage:** https://github.com/alvinunreal/oh-my-opencode-slim/blob/master/docs/council.md
- **Custom Agents:** https://github.com/alvinunreal/oh-my-opencode-slim/blob/master/docs/configuration.md#custom-agents
- **Presets:** https://github.com/alvinunreal/oh-my-opencode-slim/blob/master/docs/authors-preset.md

---

## 🎓 Example Interactions

### Example 1: Explore & Refactor
```
User: Help me understand the authentication flow

Orchestrator: I'll scout the codebase first
→ @explorer finds auth-related files
→ @oracle reviews the architecture
→ Orchestrator summarizes findings
```

### Example 2: Implement with Review
```
User: Add dark mode support to the UI

Orchestrator: I'll design and implement this
→ @designer creates the UI strategy
→ @fixer implements the changes
→ @oracle reviews for best practices
```

### Example 3: Research & Implement
```
User: Integrate Stripe payments

Orchestrator: I'll research and implement
→ @librarian fetches latest Stripe docs
→ @fixer implements the integration
→ @oracle reviews security
```

---

## 🚀 Next Steps

1. **Start OpenCode:** `opencode`
2. **Verify agents:** `ping all agents`
3. **Try a task:** Ask the orchestrator something complex
4. **Explore presets:** Try `@council` for multi-model consensus
5. **Customize:** Adjust models in the config for your workflow

---

**Ready to go!** Your agent team is standing by. 🎯
