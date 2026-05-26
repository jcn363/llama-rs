# rust-mcp-filesystem Installation & Configuration

## ✅ Installation Complete

**Status:** Successfully installed and configured in OpenCode

### What Was Installed

- **Binary:** `rust-mcp-filesystem v0.4.2`
- **Location:** `~/.rust-mcp-stack/bin/rust-mcp-filesystem`
- **Type:** Standalone Rust binary (no dependencies required)
- **Size:** Lightweight, single executable

### Configuration in OpenCode

The MCP server has been added to your OpenCode configuration at:
```
~/.config/opencode/opencode.json
```

**Configuration Details:**
```json
"rust-mcp-filesystem": {
  "type": "stdio",
  "command": "/home/user/.rust-mcp-stack/bin/rust-mcp-filesystem",
  "args": [
    "/home/user/Desktop/llama-rs",
    "/home/user"
  ],
  "env": {
    "ALLOW_WRITE": "false",
    "ENABLE_ROOTS": "false"
  },
  "enabled": true
}
```

---

## 🎯 Features

### ⚡ Performance
- **Blazingly Fast:** Built in Rust with asynchronous I/O
- **Lightweight:** Single binary, no external dependencies
- **Efficient:** Minimal resource footprint

### 🔒 Security
- **Read-Only by Default:** No write access unless explicitly enabled
- **Controlled Access:** Specify allowed directories
- **Safe Operations:** Type-safe Rust implementation

### 🔍 Capabilities

#### File Operations
- `read_file` - Read file contents
- `write_file` - Write to files (when enabled)
- `list_directory` - List directory contents
- `get_file_info` - Get file metadata
- `move_file` - Move/rename files (when enabled)
- `delete_file` - Delete files (when enabled)

#### Search & Pattern Matching
- `search_files` - Search file contents with regex
- `list_allowed_directories` - List accessible directories
- `find_files` - Find files matching glob patterns
  - Supports: `*.rs`, `src/**/*.txt`, `logs/error-???.log`

#### Archive Operations
- `create_zip` - Create ZIP archives from files/directories
- `extract_zip` - Extract ZIP files

#### Utilities
- `get_file_hash` - Calculate SHA256 file hash
- `get_file_mime_type` - Detect MIME type

---

## 📋 Current Configuration

### Allowed Directories
```
/home/user/Desktop/llama-rs    (Primary project directory)
/home/user                      (Home directory)
```

### Security Settings
```
Write Access:     DISABLED (read-only mode)
Dynamic Roots:    DISABLED (fixed directory list)
```

### Available Tools
All tools are enabled by default. To disable specific tools, use the `DISABLE_TOOLS` environment variable.

---

## 🚀 Usage Examples

### In OpenCode Sessions

#### 1. List Files in Project
```
Use the rust-mcp-filesystem to list all Rust files in the project
```

#### 2. Search for Patterns
```
Find all TODO comments in the codebase
```

#### 3. Get File Information
```
Show me the size and modification time of main.rs
```

#### 4. Create Archives
```
Create a ZIP archive of the src directory
```

---

## ⚙️ Configuration Options

### Command-Line Arguments

```bash
rust-mcp-filesystem [OPTIONS] [ALLOWED_DIRECTORIES]...
```

#### Options

| Option | Flag | Description | Default |
|--------|------|-------------|---------|
| Write Mode | `-w, --allow-write` | Enable read/write operations | Disabled |
| Disable Tools | `-d, --disable-tools` | Comma-separated tool names to disable | All enabled |
| Enable Roots | `-t, --enable-roots` | Allow dynamic directory updates from client | Disabled |
| Help | `-h, --help` | Show help message | - |
| Version | `-V, --version` | Show version | - |

#### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ALLOW_WRITE` | Enable write operations | `false` |
| `DISABLE_TOOLS` | Comma-separated tools to disable | (none) |
| `ENABLE_ROOTS` | Enable dynamic roots | `false` |

### Modifying Configuration

To change settings, edit `~/.config/opencode/opencode.json`:

#### Enable Write Access
```json
"env": {
  "ALLOW_WRITE": "true",
  "ENABLE_ROOTS": "false"
}
```

#### Add More Directories
```json
"args": [
  "/home/user/Desktop/llama-rs",
  "/home/user",
  "/tmp",
  "/var/log"
]
```

#### Disable Specific Tools
```json
"env": {
  "ALLOW_WRITE": "false",
  "DISABLE_TOOLS": "write_file,delete_file,move_file"
}
```

---

## 🔧 Advanced Configuration

### Disable Specific Tools

To reduce token usage or limit functionality, disable tools you don't need:

```json
"env": {
  "DISABLE_TOOLS": "create_zip,extract_zip,get_file_hash"
}
```

Available tools to disable:
- `read_file`
- `write_file`
- `list_directory`
- `get_file_info`
- `move_file`
- `delete_file`
- `search_files`
- `list_allowed_directories`
- `find_files`
- `create_zip`
- `extract_zip`
- `get_file_hash`
- `get_file_mime_type`

### Enable Dynamic Roots

Allow MCP clients to dynamically update allowed directories:

```json
"env": {
  "ENABLE_ROOTS": "true"
}
```

When enabled, clients can override the initial directory list.

---

## 📊 Comparison: Rust vs JavaScript

| Aspect | rust-mcp-filesystem | @modelcontextprotocol/server-filesystem |
|--------|-------------------|----------------------------------------|
| **Language** | Rust | JavaScript/Node.js |
| **Performance** | ⚡ Very Fast | Fast |
| **Dependencies** | None (single binary) | Node.js required |
| **Memory** | Minimal | Moderate |
| **Startup Time** | <100ms | ~1-2s |
| **Type Safety** | ✅ Full | Partial |
| **Async I/O** | ✅ Native | ✅ Native |
| **Glob Support** | ✅ Full | ✅ Full |
| **ZIP Support** | ✅ Yes | ✅ Yes |
| **Hash Support** | ✅ SHA256 | ❌ No |
| **MIME Detection** | ✅ Yes | ❌ No |

---

## 🔐 Security Considerations

### Current Setup (Safe)
- ✅ Read-only mode (no write access)
- ✅ Limited to specific directories
- ✅ No dynamic root changes
- ✅ All tools enabled (safe for read operations)

### To Enable Write Access
Only enable if you trust the MCP client:
```json
"env": {
  "ALLOW_WRITE": "true"
}
```

### Best Practices
1. **Keep read-only by default** - Only enable writes when needed
2. **Limit directories** - Only add directories you want to expose
3. **Disable unused tools** - Reduce attack surface
4. **Monitor access** - Check logs for unusual activity

---

## 🛠️ Installation Methods

The binary was installed using the shell script installer. Alternative methods:

### Homebrew
```bash
brew install rust-mcp-stack/tap/rust-mcp-filesystem
```

### Cargo
```bash
cargo install rust-mcp-filesystem --locked
```

### NPM
```bash
npm i -g @rustmcp/rust-mcp-filesystem@latest
```

### Docker
```bash
docker run -it rust-mcp-filesystem:latest
```

### Manual Binary Download
https://github.com/rust-mcp-stack/rust-mcp-filesystem/releases/tag/v0.4.2

---

## 📚 Documentation & Resources

- **GitHub Repository:** https://github.com/rust-mcp-stack/rust-mcp-filesystem
- **Project Documentation:** https://rust-mcp-stack.github.io/rust-mcp-filesystem
- **Docker Hub:** https://hub.docker.com/mcp/server/rust-mcp-filesystem
- **Capabilities Reference:** https://rust-mcp-stack.github.io/rust-mcp-filesystem/#/capabilities
- **License:** MIT

---

## 🔄 Integration with oh-my-opencode-slim

The rust-mcp-filesystem MCP server integrates seamlessly with your oh-my-opencode-slim agent orchestration system:

### Agent Access
- **Explorer** - Uses filesystem tools for codebase scouting
- **Librarian** - Searches documentation files
- **Fixer** - Reads files for implementation context
- **All Agents** - Can access filesystem tools as needed

### Usage in Agents
Agents automatically have access to:
- File reading and searching
- Directory listing and exploration
- Pattern matching with glob syntax
- Archive creation and extraction

---

## 🚀 Next Steps

1. **Start OpenCode:**
   ```bash
   opencode
   ```

2. **Verify MCP is loaded:**
   ```
   > mcp list
   ```

3. **Test filesystem operations:**
   ```
   > Use rust-mcp-filesystem to list all .rs files in the project
   ```

4. **Customize configuration:**
   - Edit `~/.config/opencode/opencode.json`
   - Adjust allowed directories
   - Enable/disable tools as needed
   - Enable write access if required

---

## 🐛 Troubleshooting

### MCP Not Loading
```bash
# Check if binary exists
ls -la ~/.rust-mcp-stack/bin/rust-mcp-filesystem

# Test binary directly
~/.rust-mcp-stack/bin/rust-mcp-filesystem --version
```

### Permission Denied
```bash
# Make binary executable
chmod +x ~/.rust-mcp-stack/bin/rust-mcp-filesystem
```

### Directory Not Accessible
- Verify directory paths in config
- Check directory permissions
- Ensure directories exist

### Tools Not Available
- Check `DISABLE_TOOLS` environment variable
- Verify MCP is properly loaded
- Restart OpenCode

---

## 📝 Configuration File Location

```
~/.config/opencode/opencode.json
```

### Full MCP Section
```json
"mcp": {
  "rust-mcp-filesystem": {
    "type": "stdio",
    "command": "/home/user/.rust-mcp-stack/bin/rust-mcp-filesystem",
    "args": [
      "/home/user/Desktop/llama-rs",
      "/home/user"
    ],
    "env": {
      "ALLOW_WRITE": "false",
      "ENABLE_ROOTS": "false"
    },
    "enabled": true
  }
}
```

---

**Installation Date:** 2026-05-26  
**Version:** 0.4.2  
**Status:** ✅ Ready to use
