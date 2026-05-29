//! Chat template rendering using Jinja (minijinja).
//!
//! Shared across `llama-cli`, `llama-server`, and `llama-ui`.
//! Uses the existing `minijinja` dependency in the `common` crate.

#![deny(missing_docs)]

use minijinja::Environment;

/// Errors from chat template rendering.
#[derive(Debug, thiserror::Error)]
pub enum ChatTemplateError {
    /// The template could not be rendered (e.g., missing variable).
    #[error("Template rendering failed: {0}")]
    RenderFailed(String),
    /// The named template could not be found in the environment.
    #[error("Template not found: {0}")]
    TemplateNotFound(String),
}

/// Render a chat template with system message, user prompt, and model response.
///
/// `template` is a Jinja string like:
/// ```jinja
/// <|im_start|>system\n{{ system }}<|im_end|>\n<|im_start|>user\n{{ prompt }}<|im_end|>\n<|im_start|>assistant\n
/// ```
///
/// Returns the rendered string ready to send to `/completion`.
///
/// # Errors
/// Returns `ChatTemplateError` if the template cannot be rendered or is invalid.
pub fn render_chat_template(
    system: &str,
    prompt: &str,
    template: &str,
) -> Result<String, ChatTemplateError> {
    let mut env = Environment::new();

    // Register the template string
    env.add_template("chat", template)
        .map_err(|e| ChatTemplateError::TemplateNotFound(e.to_string()))?;

    let tmpl = env
        .get_template("chat")
        .map_err(|e| ChatTemplateError::TemplateNotFound(e.to_string()))?;

    let result = tmpl
        .render(minijinja::context! {
            system => system,
            prompt => prompt,
            response => "",
            add_generation_prompt => true,
        })
        .map_err(|e| ChatTemplateError::RenderFailed(e.to_string()))?;

    Ok(result)
}

/// Get a predefined chat template for well-known architectures.
#[must_use]
pub fn get_builtin_template(architecture: &str) -> &'static str {
    match architecture {
        "chatml" | "phi3" | "phi3.5" | "qwen2" => {
            "{% if system %}<|im_start|>system\n{{ system }}<|im_end|>\n{% endif %}<|im_start|>user\n{{ prompt }}<|im_end|>\n<|im_start|>assistant\n{% if response %}{{ response }}<|im_end|>\n{% endif %}"
        }
        "llama" | "mistral" | "mixtral" => {
            "{{ system }}\n[INST] {{ prompt }} [/INST]{% if response %} {{ response }}{% endif %}"
        }
        "gemma" | "gemma2" => {
            "{{ system }}\n<start_of_turn>user\n{{ prompt }}<end_of_turn>\n<start_of_turn>model\n{% if response %}{{ response }}<end_of_turn>\n{% endif %}"
        }
        "stablelm" => {
            "{{ system }}\n<|user|>\n{{ prompt }}<|end|>\n<|assistant|>\n{% if response %}{{ response }}<|end|>\n{% endif %}"
        }
        _ => {
            // Fallback: plain concatenation with newlines
            "{{ system }}\n{{ prompt }}\n{% if response %}{{ response }}{% endif %}"
        }
    }
}

/// Render a chat prompt using the built-in template for the given architecture.
/// If `template_override` is provided, it takes precedence.
///
/// # Errors
/// Returns `ChatTemplateError` if the template cannot be rendered or is invalid.
///
/// # Examples
/// ```
/// use common::chat_templates::render_with_architecture;
///
/// // Use the built‑in "llama" template
/// let out = render_with_architecture("sys", "hi", "llama", None).unwrap();
/// assert!(out.contains("[INST] hi [/INST]"));
///
/// // Override with a custom template
/// let custom = "CUSTOM {{ prompt }}";
/// let out2 = render_with_architecture("", "test", "llama", Some(custom)).unwrap();
/// assert_eq!(out2, "CUSTOM test");
/// ```
pub fn render_with_architecture(
    system: &str,
    prompt: &str,
    architecture: &str,
    template_override: Option<&str>,
) -> Result<String, ChatTemplateError> {
    let template = template_override.unwrap_or_else(|| get_builtin_template(architecture));
    render_chat_template(system, prompt, template)
}

impl From<ChatTemplateError> for error::Error {
    fn from(err: ChatTemplateError) -> Self {
        match err {
            ChatTemplateError::RenderFailed(s) => error::Error::Template(s),
            ChatTemplateError::TemplateNotFound(s) => error::Error::Template(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chatml_template() {
        let result = render_chat_template(
            "You are a helpful assistant.",
            "What is Rust?",
            get_builtin_template("chatml"),
        )
        .unwrap();
        assert!(result.contains("system"));
        assert!(result.contains("You are a helpful assistant."));
        assert!(result.contains("user"));
        assert!(result.contains("What is Rust?"));
        assert!(result.contains("assistant"));
    }

    #[test]
    fn test_llama_template() {
        let result = render_chat_template("", "Hello!", get_builtin_template("llama")).unwrap();
        assert!(result.contains("[INST]"));
        assert!(result.contains("Hello!"));
        assert!(result.contains("[/INST]"));
    }

    #[test]
    fn test_fallback_template() {
        let result = render_chat_template(
            "System msg",
            "User msg",
            get_builtin_template("unknown-arch"),
        )
        .unwrap();
        assert!(result.contains("System msg"));
        assert!(result.contains("User msg"));
    }

    #[test]
    fn test_render_with_architecture_override() {
        let custom = "CUSTOM {{ prompt }}";
        let result = render_with_architecture("", "hi", "llama", Some(custom)).unwrap();
        assert_eq!(result, "CUSTOM hi");
    }
}
