# Changelog

## v0.2.0 (2026-06-06)

### Multi-Provider LLM Support
- **New provider abstraction layer** — `providers/` package with `BaseLLMProvider` interface
- **OllamaProvider** — existing local Ollama support, refactored into a class
- **OpenAIProvider** — new provider supporting OpenAI + compatible APIs (Groq, Together, vLLM)
- **Provider registry** — auto-select via `LLM_PROVIDER` and `EMBED_PROVIDER` env vars (can differ)
- **Config** — new env vars: `LLM_PROVIDER`, `EMBED_PROVIDER`, `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBED_MODEL`
- **Backward compatible** — default provider remains Ollama, zero caller changes

### Cross-Vault Entity Resolution
- **`entity_resolver.py`** — new module for importing entities from other vaults
- **`import_entities` MCP tool** — import + merge entities with 3 dedup strategies (exact name, alias overlap, fuzzy similarity)
- Configurable matching thresholds via `dedup_config` parameter

### Documentation
- README, setup docs, and architecture docs updated for multi-provider support
- `.env.example` updated with all new variables

## v0.1.0 (Initial)
- Initial release with Ollama-only support
