//! Sandbox process management for llama-server.
//!
//! Manages spawning, health-checking, port allocation, crash detection,
//! and graceful shutdown of `llama-server` processes.

#![deny(missing_docs)]

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Errors from sandbox client operations.
#[derive(Debug, thiserror::Error)]
pub enum SandboxError {
    /// Failed to spawn the llama-server process.
    #[error("Failed to spawn llama-server: {0}")]
    Spawn(String),
    /// Health check request failed.
    #[error("Health check failed: {0}")]
    Health(String),
    /// Server did not become ready within the timeout.
    #[error("Timeout waiting for server: {0}")]
    Timeout(String),
    /// The llama-server binary could not be found.
    #[error("llama-server binary not found: {0}")]
    BinaryNotFound(String),
    /// Wraps [`std::io::Error`] via `From` conversion.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// Wraps [`reqwest::Error`] via `From` conversion.
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
}

/// Resource limits for a sandboxed server.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceLimits {
    /// Memory limit in MB (0 = no limit).
    pub memory_mb: u64,
    /// CPU quota percentage (0 = no limit).
    pub cpu_percent: u8,
}

/// Runtime state of a sandboxed llama-server.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SandboxStatus {
    /// Server process has been spawned but not yet confirmed healthy.
    Starting,
    /// Server is running and `/health` returns OK.
    Ready,
    /// Server has been intentionally stopped/shut down.
    Stopped,
    /// Server has crashed or become unresponsive, with a description.
    Crashed(String),
}

/// Client for managing a llama-server sandbox process.
#[derive(Debug)]
pub struct SandboxClient {
    /// Port the server is listening on.
    pub port: u16,
    /// Current status.
    pub status: SandboxStatus,
    /// Path to the llama-server binary.
    binary_path: PathBuf,
    /// Model path to serve.
    model_path: PathBuf,
    /// Backend to use.
    backend: String,
    /// Resource limits.
    limits: ResourceLimits,
    /// Handle to the child process.
    child: Option<Child>,
    /// HTTP client for health checks.
    client: reqwest::Client,
    /// Tag for log filtering.
    tag: String,
}

impl SandboxClient {
    /// Create a new sandbox client configuration (does not spawn).
    pub fn new(
        binary_path: PathBuf,
        model_path: PathBuf,
        backend: impl Into<String>,
        tag: impl Into<String>,
    ) -> Self {
        Self {
            port: 0,
            status: SandboxStatus::Stopped,
            binary_path,
            model_path,
            backend: backend.into(),
            limits: ResourceLimits::default(),
            child: None,
            client: reqwest::Client::new(),
            tag: tag.into(),
        }
    }

    /// Set resource limits.
    pub fn with_limits(mut self, limits: ResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Resolve the llama-server binary path.
    /// Searches next to the current executable first, then PATH.
    pub fn resolve_binary() -> Result<PathBuf, SandboxError> {
        // Check next to current executable
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(parent) = exe_path.parent() {
                let candidate = parent.join("llama-server");
                if candidate.exists() {
                    return Ok(candidate);
                }
                #[cfg(windows)]
                {
                    let candidate = parent.join("llama-server.exe");
                    if candidate.exists() {
                        return Ok(candidate);
                    }
                }
            }
        }

        // Fall back to PATH
        if let Ok(path) = std::env::var("PATH") {
            for dir in std::env::split_paths(&path) {
                let candidate = dir.join("llama-server");
                if candidate.exists() {
                    return Ok(candidate);
                }
                #[cfg(windows)]
                {
                    let candidate = dir.join("llama-server.exe");
                    if candidate.exists() {
                        return Ok(candidate);
                    }
                }
            }
        }

        Err(SandboxError::BinaryNotFound(
            "llama-server not found next to binary or in PATH".into(),
        ))
    }

    /// Spawn the llama-server process.
    pub fn spawn(&mut self) -> Result<(), SandboxError> {
        if !self.binary_path.exists() {
            return Err(SandboxError::BinaryNotFound(format!(
                "Binary not found: {}",
                self.binary_path.display()
            )));
        }

        // Find a free port by binding to port 0
        let listener = std::net::TcpListener::bind("127.0.0.1:0")?;
        self.port = listener.local_addr()?.port();
        drop(listener); // Release so llama-server can bind

        let mut cmd = Command::new(&self.binary_path);
        cmd.arg("-m")
            .arg(&self.model_path)
            .arg("--port")
            .arg(self.port.to_string())
            .arg("--host")
            .arg("127.0.0.1")
            .arg("--backend")
            .arg(&self.backend)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        // Apply resource limits via systemd-run if available
        if (self.limits.memory_mb > 0 || self.limits.cpu_percent > 0) && cfg!(target_os = "linux") {
            // Try to detect systemd-run availability
            let has_systemd_run = Command::new("which")
                .arg("systemd-run")
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .ok()
                .map(|s| s.success())
                .unwrap_or(false);

            if has_systemd_run {
                let mut sd_cmd = Command::new("systemd-run");
                sd_cmd.arg("--scope");
                sd_cmd.arg("--user");
                sd_cmd.arg("-q");
                if self.limits.memory_mb > 0 {
                    let mem_arg = format!("MemoryMax={}M", self.limits.memory_mb);
                    sd_cmd.arg("-p");
                    sd_cmd.arg(&mem_arg);
                }
                if self.limits.cpu_percent > 0 {
                    let cpu_arg = format!("CPUQuota={}%", self.limits.cpu_percent);
                    sd_cmd.arg("-p");
                    sd_cmd.arg(&cpu_arg);
                }
                sd_cmd.arg(&self.binary_path);
                sd_cmd.arg("-m").arg(&self.model_path);
                sd_cmd.arg("--port").arg(self.port.to_string());
                sd_cmd.arg("--host").arg("127.0.0.1");
                sd_cmd.arg("--backend").arg(&self.backend);
                sd_cmd.stdout(Stdio::piped()).stderr(Stdio::piped());
                cmd = sd_cmd;
            } else {
                tracing::warn!(
                    tag = %self.tag,
                    "systemd-run not available, skipping resource limits"
                );
            }
        }

        let child = cmd
            .spawn()
            .map_err(|e| SandboxError::Spawn(format!("Failed to start llama-server: {e}")))?;

        self.child = Some(child);
        self.status = SandboxStatus::Starting;
        tracing::info!(tag = %self.tag, port = %self.port, "Sandbox spawned");

        Ok(())
    }

    /// Health check: poll `/health` until ready or timeout.
    pub async fn wait_for_ready(&self, timeout: Duration) -> Result<(), SandboxError> {
        let start = std::time::Instant::now();
        loop {
            if start.elapsed() > timeout {
                return Err(SandboxError::Timeout(format!(
                    "Server not ready after {}ms",
                    timeout.as_millis()
                )));
            }

            match self.health_check().await {
                Ok(true) => return Ok(()),
                _ => tokio::time::sleep(Duration::from_millis(200)).await,
            }
        }
    }

    /// Perform a health check against `/health`.
    pub async fn health_check(&self) -> Result<bool, SandboxError> {
        let url = format!("http://127.0.0.1:{}/health", self.port);
        match self.client.get(&url).send().await {
            Ok(resp) => Ok(resp.status().is_success()),
            Err(_) => Ok(false),
        }
    }

    /// Send a completion request to the sandbox.
    pub async fn complete(
        &self,
        prompt: &str,
        max_tokens: usize,
        temperature: f32,
        top_k: usize,
        top_p: f32,
        stream: bool,
    ) -> Result<String, SandboxError> {
        let url = format!("http://127.0.0.1:{}/completion", self.port);
        #[derive(Serialize)]
        struct Request<'a> {
            prompt: &'a str,
            max_tokens: usize,
            stream: bool,
            temperature: f32,
            top_k: usize,
            top_p: f32,
        }
        let req = Request {
            prompt,
            max_tokens,
            stream,
            temperature,
            top_k,
            top_p,
        };
        let resp = self.client.post(&url).json(&req).send().await?;
        let body: serde_json::Value = resp.json().await?;
        Ok(body["content"].as_str().unwrap_or("").to_string())
    }

    /// Gracefully stop the sandbox process.
    pub fn stop(&mut self) {
        if let Some(mut child) = self.child.take() {
            // Try SIGTERM first, wait, then SIGKILL
            #[cfg(unix)]
            {
                // Send SIGTERM
                let pid = nix::unistd::Pid::from_raw(child.id() as i32);
                let _ = nix::sys::signal::kill(pid, nix::sys::signal::SIGTERM);
                std::thread::sleep(Duration::from_secs(5));
                if child.try_wait().ok().flatten().is_none() {
                    let _ = nix::sys::signal::kill(pid, nix::sys::signal::SIGKILL);
                }
            }
            #[cfg(not(unix))]
            {
                let _ = child.kill();
            }
            let _ = child.wait();
            tracing::info!(tag = %self.tag, "Sandbox stopped");
        }
        self.status = SandboxStatus::Stopped;
    }

    /// Check if the sandbox is still alive and update status.
    pub async fn probe(&mut self) -> SandboxStatus {
        match self.health_check().await {
            Ok(true) => {
                self.status = SandboxStatus::Ready;
                SandboxStatus::Ready
            }
            _ => {
                self.status = SandboxStatus::Crashed("Health check failed".into());
                self.status.clone()
            }
        }
    }
}

impl Drop for SandboxClient {
    fn drop(&mut self) {
        self.stop();
    }
}

impl From<SandboxError> for error::Error {
    fn from(err: SandboxError) -> Self {
        match err {
            SandboxError::Spawn(s) => error::Error::Other(s),
            SandboxError::Health(s) => error::Error::Network(s),
            SandboxError::Timeout(s) => error::Error::Other(s),
            SandboxError::BinaryNotFound(s) => error::Error::Other(s),
            SandboxError::Io(e) => error::Error::Io(e),
            SandboxError::Http(e) => error::Error::Other(e.to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_binary_not_found() {
        // In test environment, llama-server won't be in PATH
        let result = SandboxClient::resolve_binary();
        // Either succeeds (if found) or fails with BinaryNotFound
        match result {
            Ok(path) => assert!(path.exists()),
            Err(e) => assert!(matches!(e, SandboxError::BinaryNotFound(_))),
        }
    }

    #[test]
    fn test_resource_limits_default() {
        let limits = ResourceLimits::default();
        assert_eq!(limits.memory_mb, 0);
        assert_eq!(limits.cpu_percent, 0);
    }

    #[test]
    fn test_sandboxclient_new_and_with_limits() {
        let binary = PathBuf::from("/does/not/exist/llama-server");
        let model = PathBuf::from("/tmp/model.gguf");
        let client = SandboxClient::new(binary.clone(), model.clone(), "cpu", "tag1");
        assert!(matches!(client.status, SandboxStatus::Stopped));
        let limits = ResourceLimits {
            memory_mb: 512,
            cpu_percent: 50,
        };
        let client2 = client.with_limits(limits.clone());
        assert_eq!(client2.limits.memory_mb, 512);
        assert_eq!(client2.limits.cpu_percent, 50);
    }
}
