# Uncensored Persistence System

A type-safe persistence system for uncensored agents with proper error handling, validation, and type safety.

## Features

- Type-safe session management
- Proper error handling with custom error types
- Input validation to prevent unsafe operations
- Session caching for performance
- Comprehensive testing support

## Usage

```bash
# Save current state
uncensored-persistence save --name my-session

# Load a saved state
uncensored-persistence load --name my-session

# List available sessions
uncensored-persistence list

# Validate a session name
uncensored-persistence validate --name test_name
```

## Building

```bash
cargo build --release
```

## Testing

```bash
cargo test
```

## License

MIT