// src/main.rs
use clap::{Parser, Subcommand};
use std::path::PathBuf;
use uncensored_persistence::PersistenceManager;

#[derive(Parser)]
#[command(
    name = "uncensored-persistence",
    version,
    about = "Type-safe persistence for uncensored agents"
)]
struct Cli {
    /// Base directory for storing sessions
    #[arg(short, long, default_value = ".uncensored")]
    base_dir: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Save the current state to a named session
    Save {
        /// Name of the session to save
        name: String,
    },

    /// Load a previously saved session
    Load {
        /// Name of the session to load
        name: String,
    },

    /// List all available sessions
    List,

    /// Validate a session name for safety
    Validate {
        /// Name to validate
        name: String,
    },
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    // Create persistence manager
    let mut manager = PersistenceManager::new(cli.base_dir)?;

    match cli.command {
        Commands::Save { name } => {
            // Use manager to create placeholder session
            manager.save(&name)?;
            println!("Session '{}' saved successfully", name);
        }
        Commands::Load { name } => {
            let state = manager.load(&name)?;
            println!("Loaded session '{}':", name);
            println!("  Session ID: {}", state.session_id);
            println!("  Created: {}", state.created_at);
            println!("  Updated: {}", state.updated_at);
        }
        Commands::List => {
            let sessions = manager.list()?;
            if sessions.is_empty() {
                println!("No sessions found");
            } else {
                println!("Available sessions:");
                for session in sessions {
                    println!("  - {}", session);
                }
            }
        }
        Commands::Validate { name } => {
            let valid = uncensored_persistence::validate_session_name(&name);
            if valid {
                println!("Session name '{}' is valid", name);
            } else {
                println!("Session name '{}' is INVALID", name);
            }
        }
    }

    Ok(())
}
