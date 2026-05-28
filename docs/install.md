# Install pre-built version of llama.cpp

| Install via | Windows | Mac | Linux |
|-------------|---------|-----|-------|
| Winget      | ✅      |      |      |
| Homebrew    |         | ✅   | ✅   |
| MacPorts    |         | ✅   |      |
| Nix         |         | ✅   | ✅   |
| Debian/Ubuntu |       |      | ✅   |

## Winget (Windows)

```sh
winget install llama.cpp
```

The package is automatically updated with new `llama.cpp` releases. More info: https://github.com/ggml-org/llama.cpp/issues/8188

## Homebrew (Mac and Linux)

```sh
brew install llama.cpp
```

The formula is automatically updated with new `llama.cpp` releases. More info: https://github.com/ggml-org/llama.cpp/discussions/7668

## MacPorts (Mac)

```sh
sudo port install llama.cpp
```

See also: https://ports.macports.org/port/llama.cpp/details/

## Nix (Mac and Linux)

```sh
nix profile install nixpkgs#llama-cpp
```

For flake enabled installs.

Or

```sh
nix-env --file '<nixpkgs>' --install --attr llama-cpp
```

For non-flake enabled installs.

This expression is automatically updated within the [nixpkgs repo](https://github.com/NixOS/nixpkgs/blob/nixos-24.05/pkgs/by-name/ll/llama-cpp/package.nix#L164).

## Debian/Ubuntu (.deb package)

```sh
sudo dpkg -i llama-rs_0.1.0_amd64.deb
```

Or build from source:

```sh
cargo build --release --workspace
sudo cp target/release/llama-cli target/release/llama-server target/release/llama-ui /usr/local/bin/
```

The .deb package includes:
- `llama-cli` — Command-line interface for interactive text generation
- `llama-server` — HTTP server with `/completion` and `/health` endpoints
- `llama-ui` — Desktop GUI for interactive LLM inference

To uninstall:

```sh
sudo dpkg -r llama-rs
```
