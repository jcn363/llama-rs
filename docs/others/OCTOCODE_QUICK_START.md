# Octocode Quick Start Guide

## 🚀 Quick Start

### 1. Start OpenCode
```bash
opencode
```

### 2. Verify Octocode is Loaded
```
> mcp list
```
You should see `octocode` in the list of MCP servers.

### 3. Set GitHub Authentication (Required for private repos)
```bash
export GITHUB_TOKEN=ghp_your_token_here
```

### 4. Test Octocode Tools
```
> Search GitHub for "Rust machine learning"
> Find all files matching src/**/*.rs in llama-rs
> Get pull request details for #42
> Search for "TODO" comments in the codebase
```

## 🎯 Common Use Cases

### GitHub Repository Search
```
> Search GitHub for React hooks patterns
> Find repositories similar to llama-rs
> Search for "Rust async patterns" on GitHub
```

### Local Code Analysis
```
> Search for "TODO" comments in the project
> Find all test files matching *test*.rs
> Browse the src directory structure
```

### Pull Request Analysis
```
> Get details for PR #123
> Review changes in pull request #45
> Find open issues related to performance
```

### LSP Navigation
```
> Go to definition of function `main`
> Find all references to `struct Config`
> Show call hierarchy for `fn process`
```

## 🤖 Agent Integration

### Explorer Agent
```
> @explorer Search GitHub for Rust async patterns
> @explorer Find similar implementations of this algorithm
> @explorer Browse llama-rs repository structure
```

### Librarian Agent
```
> @librarian Find documentation for async Rust
> @librarian Search GitHub for best practices
> @librarian Get package source code for serde
```

### Oracle Agent
```
> @oracle Analyze architecture of similar projects
> @oracle Find established patterns for this problem
> @oracle Research implementation approaches
```

### Fixer Agent
```
> @fixer Read reference implementation from GitHub
> @fixer Find examples of error handling patterns
> @fixer Get documentation for this function
```

## 🔧 Authentication Setup

### GitHub Token Required
1. Create GitHub Personal Access Token:
   - Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Scopes needed: `repo`, `read:org`, `read:user`

2. Set token:
   ```bash
   export GITHUB_TOKEN=ghp_your_token_here
   ```

### GitLab (Optional)
```bash
export GITLAB_TOKEN=your_gitlab_token
```

### Bitbucket (Optional)
```bash
export BITBUCKET_TOKEN=your_bitbucket_token
```

## 📋 Available Tools

### GitHub Tools
- `github_search_repositories` - Search repositories
- `github_search_code` - Search code in repositories
- `github_get_file` - Read repository files
- `github_get_pull_request` - Get PR details
- `github_get_issue` - Get issue details
- `github_get_user` - Get user information
- `github_get_repository` - Get repository information

### Local Tools
- `local_search_code` - Search local codebase
- `local_browse_directory` - Browse directories
- `local_find_files` - Find files by pattern
- `local_read_file` - Read local files

### LSP Tools
- `lsp_go_to_definition` - Navigate to definitions
- `lsp_find_references` - Find symbol references
- `lsp_call_hierarchy` - Analyze call relationships
- `lsp_hover` - Get symbol documentation

### Package Tools
- `npm_resolve_package` - Resolve npm packages
- `pypi_resolve_package` - Resolve PyPI packages

## 🚨 Troubleshooting

### "GitHub token not found"
```bash
# Check if token is set
echo $GITHUB_TOKEN

# Set token if missing
export GITHUB_TOKEN=ghp_your_token_here
```

### "MCP server not loading"
```bash
# Test octocode directly
npx octocode-mcp@latest --version

# Check OpenCode config
grep octocode ~/.config/opencode/opencode.json
```

### "Permission denied"
```bash
# Check GitHub token scopes
gh auth status

# Ensure token has `repo` scope for private repos
```

## 💡 Pro Tips

### Efficient Searching
- Use specific keywords for better results
- Combine search terms for precision
- Use quotes for exact matches

### Local vs Remote
- Use local tools for fast file operations
- Use GitHub tools for broader research
- Leverage LSP for accurate code navigation

### Agent Collaboration
- Use multiple agents for comprehensive analysis
- Cross-reference information between sources
- Build on agent findings for deeper insights

---

**Ready to use!** 🎉

Start by running `opencode` and try searching GitHub or analyzing your local codebase with the integrated agents.