# rust-mcp-filesystem Quick Reference

## ✅ Installation Status

**Status:** ✅ Installed and configured  
**Version:** 0.4.2  
**Binary Location:** `~/.rust-mcp-stack/bin/rust-mcp-filesystem`  
**Config Location:** `~/.config/opencode/opencode.json`

---

## 🎯 Quick Start

### 1. Start OpenCode
```bash
opencode
```

### 2. Verify MCP is Loaded
```
> mcp list
```

You should see `rust-mcp-filesystem` in the list.

### 3. Use Filesystem Tools
```
> List all Rust files in the project
> Find files matching *.rs pattern
> Search for TODO comments
```

---

## 📋 Available Tools

| Tool | Purpose | Example |
|------|---------|---------|
| `read_file` | Read file contents | Read src/main.rs |
| `list_directory` | List directory contents | List the src directory |
| `get_file_info` | Get file metadata | Show file size and date |
| `search_files` | Search file contents | Find all TODO comments |
| `find_files` | Find files by pattern | Find all *.rs files |
| `get_file_hash` | Calculate SHA256 hash | Hash a file |
| `get_file_mime_type` | Detect file type | Check MIME type |
| `create_zip` | Create ZIP archive | Archive the src directory |
| `extract_zip` | Extract ZIP file | Extract archive.zip |

---

## ⚙️ Configuration

### Current Settings
```json
{
  "command": "/home/user/.rust-mcp-stack/bin/rust-mcp-filesystem",
  "args": [
    "/home/user/Desktop/llama-rs",
    "/home/user"
  ],
  "env": {
    "ALLOW_WRITE": "false",
    "ENABLE_ROOTS": "false"
  }
}
```

### Allowed Directories
- `/home/user/Desktop/llama-rs` (project root)
- `/home/user` (home directory)

### Security
- ✅ Read-only mode (no write access)
- ✅ Limited to specified directories
- ✅ All tools enabled

---

## 🔧 Common Tasks

### List Files
```
Use rust-mcp-filesystem to list all files in /home/user/Desktop/llama-rs
```

### Search for Pattern
```
Find all files matching src/**/*.rs
```

### Search File Contents
```
Search for "TODO" in all files
```

### Get File Info
```
Show metadata for Cargo.toml
```

### Create Archive
```
Create a ZIP archive of the src directory
```

---

## 🔐 Security Settings

### Current (Safe)
- Write: **Disabled**
- Dynamic Roots: **Disabled**
- Tools: **All enabled** (safe for read-only)

### To Enable Write Access
Edit `~/.config/opencode/opencode.json`:
```json
"env": {
  "ALLOW_WRITE": "true"
}
```

### To Disable Specific Tools
```json
"env": {
  "DISABLE_TOOLS": "create_zip,extract_zip"
}
```

---

## 📊 Performance

- **Startup:** <100ms
- **File Read:** Instant
- **Directory Scan:** Very fast (async)
- **Search:** Optimized with parallel processing
- **Memory:** Minimal footprint

---

## 🆚 vs JavaScript Version

| Feature | Rust | JS |
|---------|------|-----|
| Speed | ⚡⚡⚡ | ⚡⚡ |
| Dependencies | None | Node.js |
| Startup | <100ms | 1-2s |
| Hash Support | ✅ | ❌ |
| MIME Detection | ✅ | ❌ |

---

## 🚀 Integration with Agents

All oh-my-opencode-slim agents can use filesystem tools:

- **Explorer** - Scout codebase
- **Librarian** - Search documentation
- **Fixer** - Read implementation context
- **Oracle** - Analyze code structure

---

## 🛠️ Troubleshooting

### MCP Not Loading
```bash
# Check binary
~/.rust-mcp-stack/bin/rust-mcp-filesystem --version

# Check config
cat ~/.config/opencode/opencode.json | grep rust-mcp-filesystem
```

### Permission Issues
```bash
chmod +x ~/.rust-mcp-stack/bin/rust-mcp-filesystem
```

### Directory Not Found
- Verify paths in config
- Check directory exists
- Verify permissions

---

## 📚 Resources

- **GitHub:** https://github.com/rust-mcp-stack/rust-mcp-filesystem
- **Docs:** https://rust-mcp-stack.github.io/rust-mcp-filesystem
- **Docker:** https://hub.docker.com/mcp/server/rust-mcp-filesystem

---

## 🔄 Update

To update to the latest version:
```bash
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/rust-mcp-stack/rust-mcp-filesystem/releases/download/v0.4.2/rust-mcp-filesystem-installer.sh | sh
```

---

**Ready to use!** Start OpenCode and begin using filesystem tools. 🚀
