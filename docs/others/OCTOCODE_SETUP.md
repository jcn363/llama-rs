# Octocode Installation & Configuration Guide

## Overview

Octocode is a powerful MCP server that provides GitHub, GitLab & Bitbucket integration with AI-powered code research capabilities. It enables semantic code search, context generation, and real-time code analysis across public and private repositories.

## What is Octocode?

**Octocode** is an MCP server that empowers your AI assistant with the skills of a Senior Staff Engineer:

- **GitHub, GitLab & Bitbucket**: Search repositories, find usage patterns, read implementations, explore PRs/MRs
- **Local Tools**: Search code (ripgrep), browse directories, find files in your local codebase
- **LSP Intelligence**: Go to Definition, Find References, Call Hierarchy — compiler-level understanding
- **Package Discovery**: Resolve npm/PyPI packages to their source repos

## Installation Status

✅ **Installation Complete**
- Version: 14.2.0
- Status: Working and configured
- Location: npx-based (no local installation required)
- Integration: Enabled in OpenCode

## Configuration

### OpenCode Configuration
**File:** `~/.config/opencode/opencode.json`

```json
"octocode": {
  "type": "stdio",
  "command": "npx",
  "args": [
    "octocode-mcp@latest"
  ],
  "env": {
    "ENABLE_LOCAL": "true"
  },
  "enabled": true
}
```

### Key Settings
- **Type**: `stdio` (standard input/output)
- **Command**: `npx` (Node package runner)
- **Args**: `octocode-mcp@latest` (latest version)
- **Environment**: `ENABLE_LOCAL: "true"` (enable local tools)
- **Status**: `enabled: true` (active)

## Available Tools

### GitHub, GitLab & Bitbucket Tools
- **Repository Search**: Find repositories by name, topic, language
- **Code Search**: Search across repositories with natural language
- **File Operations**: Read files, get file metadata
- **Pull Request Analysis**: Review PRs, get diff information
- **Issue Tracking**: Search and analyze issues
- **User Operations**: Get user information, repositories

### Local Tools
- **Code Search**: Search local codebase with ripgrep
- **File Browser**: Browse directory structures
- **File Finder**: Find files by pattern
- **Local File Operations**: Read local files

### LSP Tools
- **Go to Definition**: Navigate to symbol definitions
- **Find References**: Find all references to a symbol
- **Call Hierarchy**: Analyze function call relationships
- **Hover Information**: Get documentation for symbols

### Package Discovery
- **npm/PyPI Resolution**: Resolve packages to source repositories
- **Package Information**: Get package metadata and documentation

## Authentication

### GitHub Authentication Required
Octocode requires GitHub authentication for access to private repositories and GitHub API.

#### Setup GitHub Token
1. Create a GitHub Personal Access Token:
   - Go to GitHub → Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Generate new token with appropriate scopes:
     - `repo` (Full control of private repositories)
     - `read:org` (Read org and team membership)
     - `read:user` (Read user and email)

2. Set environment variable:
   ```bash
   export GITHUB_TOKEN=ghp_your_token_here
   ```

#### Alternative: GitHub CLI
```bash
gh auth login
gh auth status
```

### GitLab Authentication
```bash
export GITLAB_TOKEN=your_gitlab_token
```

### Bitbucket Authentication
```bash
export BITBUCKET_TOKEN=your_bitbucket_token
```

## Usage Examples

### In OpenCode
```
> Search GitHub for React hooks patterns
> Find all files matching src/**/*.ts in llama-rs
> Get pull request details for #42
> Search for "TODO" comments in the codebase
> Go to definition of function `main`
```

### Common Commands
```bash
# Test octocode installation
npx octocode-mcp@latest --version

# Check MCP server status in OpenCode
> mcp list

# Verify octocode tools are available
> Use octocode to search GitHub for "Rust machine learning"
```

## Integration with oh-my-opencode-slim Agents

Octocode tools are automatically available to all oh-my-opencode-slim agents:

### Explorer Agent
- **Codebase Scouting**: Search local and remote repositories
- **Pattern Discovery**: Find implementation patterns across projects
- **File Navigation**: Browse code structures with LSP support

### Librarian Agent  
- **Documentation Search**: Find READMEs and docs in repositories
- **Code Examples**: Search for implementation examples
- **Package Research**: Resolve packages to source code

### Oracle Agent
- **Architecture Analysis**: Study codebase structure across repos
- **Best Practices**: Find established patterns and conventions
- **Research Support**: Gather evidence from multiple sources

### Fixer Agent
- **Implementation Context**: Read remote code for reference
- **Dependency Resolution**: Find package sources and documentation
- **Code Comparison**: Compare implementations across projects

### Designer Agent
- **UI Pattern Research**: Find design system implementations
- **Component Examples**: Search for similar UI components
- **Style Guides**: Find design documentation

### Council Agent
- **Multi-Source Analysis**: Cross-reference information from multiple repos
- **Consensus Building**: Compare implementations and approaches
- **Evidence Gathering**: Collect data for decision making

## Security Considerations

### Token Security
- Never commit tokens to version control
- Use environment variables or secure credential managers
- Regularly rotate tokens
- Use minimal required scopes

### Access Control
- Octocode respects repository permissions
- Private repositories require authentication
- Local tools are limited to configured directories

### Network Security
- All API calls go through official GitHub/GitLab/Bitbucket APIs
- No third-party proxy services
- Encrypted communication via HTTPS

## Troubleshooting

### Common Issues

#### "GitHub token not found"
```bash
# Check if token is set
echo $GITHUB_TOKEN

# Set token if missing
export GITHUB_TOKEN=ghp_your_token_here
```

#### "MCP server not loading"
```bash
# Test octocode directly
npx octocode-mcp@latest --version

# Check OpenCode config
grep octocode ~/.config/opencode/opencode.json
```

#### "Permission denied for repository"
```bash
# Check GitHub token scopes
gh auth status

# Ensure token has `repo` scope for private repos
```

#### "Local tools not working"
```bash
# Verify ENABLE_LOCAL is set
grep ENABLE_LOCAL ~/.config/opencode/opencode.json

# Check directory permissions
ls -la /home/user/Desktop/llama-rs
```

### Debug Commands
```bash
# Test octocode standalone
npx octocode-mcp@latest --help

# Check MCP server in OpenCode
> mcp list

# Test specific tool
> Use octocode to search GitHub for "Rust async"
```

## Performance Tips

### Caching
- Octocode caches responses to improve performance
- Local tool results are cached for faster repeated searches
- LSP data is cached to reduce compilation overhead

### Optimization
- Use specific search queries to reduce API calls
- Leverage local tools for faster file operations
- Use LSP tools for accurate code navigation

### Resource Management
- Monitor memory usage with large repositories
- Use pagination for large result sets
- Clean up unused sessions regularly

## Advanced Configuration

### Environment Variables
```json
{
  "env": {
    "ENABLE_LOCAL": "true",
    "GITHUB_TOKEN": "your_token",
    "GITLAB_TOKEN": "your_token",
    "BITBUCKET_TOKEN": "your_token",
    "MAX_RESULTS": "50",
    "ENABLE_LSP": "true"
  }
}
```

### Custom Arguments
```json
{
  "args": [
    "octocode-mcp@latest",
    "--verbose",
    "--timeout=30000"
  ]
}
```

### Multiple Instances
```json
{
  "mcp": {
    "octocode-github": {
      "type": "stdio",
      "command": "npx",
      "args": ["octocode-mcp@latest"],
      "env": {
        "ENABLE_LOCAL": "false",
        "GITHUB_TOKEN": "github_token"
      }
    },
    "octocode-local": {
      "type": "stdio", 
      "command": "npx",
      "args": ["octocode-mcp@latest"],
      "env": {
        "ENABLE_LOCAL": "true"
      }
    }
  }
}
```

## Learning Resources

### Documentation
- **Official Docs**: https://github.com/bgauryy/octocode-mcp/tree/main/docs
- **Configuration Guide**: https://github.com/bgauryy/octocode-mcp/blob/main/docs/configuration/CONFIGURATION_REFERENCE.md
- **Tool References**: https://github.com/bgauryy/octocode-mcp/blob/main/docs/dev/reference/GITHUB_GITLAB_TOOLS_REFERENCE.md

### Skills and Workflows
- **Research Skills**: https://github.com/bgauryy/octocode-mcp/tree/main/skills
- **Workflows**: https://github.com/bgauryy/octocode-mcp/tree/main/docs/dev/workflows
- **Agent Guidance**: https://github.com/bgauryy/octocode-mcp/blob/main/AGENTS.md

### Community
- **GitHub Discussions**: https://github.com/bgauryy/octocode-mcp/discussions
- **Issues**: https://github.com/bgauryy/octocode-mcp/issues
- **Discord**: https://discord.gg/octocode

## Next Steps

1. **Verify Setup**:
   ```bash
   opencode
   > mcp list
   ```

2. **Test Authentication**:
   ```bash
   export GITHUB_TOKEN=your_token
   npx octocode-mcp@latest --version
   ```

3. **Explore Tools**:
   - Search GitHub for relevant repositories
   - Use local code search
   - Test LSP navigation

4. **Integrate with Agents**:
   - Ask @explorer to search GitHub patterns
   - Use @librarian to find documentation
   - Leverage @oracle for architectural analysis

---

**Installation Date**: 2026-05-26  
**Status**: ✅ Complete and Ready  
**Version**: 14.2.0  
**Integration**: ✅ OpenCode + oh-my-opencode-slim