/// LLM Client — Provider-agnostic chat interface
/// ================================================
/// Supports Ollama (local) and Anthropic (cloud).
/// Configuration via .env file or environment variables.

use serde::{Serialize, Deserialize};

#[derive(Debug, Clone)]
pub enum LlmProvider {
    Ollama { base_url: String, model: String },
    Anthropic { api_key: String, model: String },
}

#[derive(Serialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaResponse {
    message: Option<OllamaMessageContent>,
}

#[derive(Deserialize)]
struct OllamaMessageContent {
    content: String,
}

#[derive(Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    system: String,
    messages: Vec<AnthropicMessage>,
}

#[derive(Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct AnthropicResponse {
    content: Option<Vec<AnthropicContent>>,
    error: Option<AnthropicError>,
}

#[derive(Deserialize)]
struct AnthropicContent {
    text: String,
}

#[derive(Deserialize)]
struct AnthropicError {
    message: String,
}

/// Load LLM provider configuration from environment / .env file
pub fn load_provider() -> Option<LlmProvider> {
    // Try to load .env file (ignore errors if not found)
    load_dotenv();

    let provider = std::env::var("LLM_PROVIDER").unwrap_or_default();
    match provider.to_lowercase().as_str() {
        "ollama" => {
            let base_url = std::env::var("LLM_BASE_URL")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            let model = std::env::var("LLM_MODEL")
                .unwrap_or_else(|_| "llama3.1:8b".to_string());
            Some(LlmProvider::Ollama { base_url, model })
        }
        "anthropic" => {
            let api_key = std::env::var("LLM_API_KEY").ok()?;
            let model = std::env::var("LLM_MODEL")
                .unwrap_or_else(|_| "claude-sonnet-4-5-20250929".to_string());
            Some(LlmProvider::Anthropic { api_key, model })
        }
        _ => None,
    }
}

/// Simple .env loader — reads KEY=VALUE lines from .env file
fn load_dotenv() {
    if let Ok(contents) = std::fs::read_to_string(".env") {
        for line in contents.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') { continue; }
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim();
                // Only set if not already in environment
                if std::env::var(key).is_err() {
                    std::env::set_var(key, value);
                }
            }
        }
    }
}

/// Send a chat message to the configured LLM provider
pub async fn chat(
    client: &reqwest::Client,
    provider: &LlmProvider,
    system_prompt: &str,
    user_message: &str,
) -> Result<String, String> {
    match provider {
        LlmProvider::Ollama { base_url, model } => {
            chat_ollama(client, base_url, model, system_prompt, user_message).await
        }
        LlmProvider::Anthropic { api_key, model } => {
            chat_anthropic(client, api_key, model, system_prompt, user_message).await
        }
    }
}

async fn chat_ollama(
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
    system_prompt: &str,
    user_message: &str,
) -> Result<String, String> {
    let url = format!("{}/api/chat", base_url);

    let request = OllamaRequest {
        model: model.to_string(),
        messages: vec![
            OllamaMessage { role: "system".to_string(), content: system_prompt.to_string() },
            OllamaMessage { role: "user".to_string(), content: user_message.to_string() },
        ],
        stream: false,
    };

    let response = client.post(&url)
        .json(&request)
        .timeout(std::time::Duration::from_secs(120))
        .send()
        .await
        .map_err(|e| format!("Ollama connection failed: {}. Is Ollama running at {}?", e, base_url))?;

    if !response.status().is_success() {
        return Err(format!("Ollama returned HTTP {}", response.status()));
    }

    let body: OllamaResponse = response.json().await
        .map_err(|e| format!("Failed to parse Ollama response: {}", e))?;

    body.message
        .map(|m| m.content)
        .ok_or_else(|| "Empty response from Ollama".to_string())
}

async fn chat_anthropic(
    client: &reqwest::Client,
    api_key: &str,
    model: &str,
    system_prompt: &str,
    user_message: &str,
) -> Result<String, String> {
    let request = AnthropicRequest {
        model: model.to_string(),
        max_tokens: 1024,
        system: system_prompt.to_string(),
        messages: vec![
            AnthropicMessage { role: "user".to_string(), content: user_message.to_string() },
        ],
    };

    let response = client.post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", api_key)
        .header("anthropic-version", "2023-06-01")
        .header("content-type", "application/json")
        .json(&request)
        .timeout(std::time::Duration::from_secs(60))
        .send()
        .await
        .map_err(|e| format!("Anthropic API error: {}", e))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(format!("Anthropic returned HTTP {}: {}", status, body));
    }

    let body: AnthropicResponse = response.json().await
        .map_err(|e| format!("Failed to parse Anthropic response: {}", e))?;

    if let Some(err) = body.error {
        return Err(format!("Anthropic error: {}", err.message));
    }

    body.content
        .and_then(|c| c.into_iter().next())
        .map(|c| c.text)
        .ok_or_else(|| "Empty response from Anthropic".to_string())
}
