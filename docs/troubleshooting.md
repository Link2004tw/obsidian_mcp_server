# Troubleshooting

## 500 Server Error from Ollama

**Symptom:**
```
FAILED: path/to/note.md — 500 Server Error: Internal Server Error for url: http://localhost:11434/api/embeddings
```

**Common causes:**

1. **Broken Unicode in the note** — The `_sanitize()` function handles most cases. If it still fails, the note may have encoding issues at the Obsidian API level.

2. **Ollama model not loaded** — The first embedding call may be slow while the model loads. Subsequent calls should be faster.

3. **Ollama out of memory** — If you're running other models, Ollama may not have enough VRAM. Stop other models with `ollama stop`.

**Fix:** Check `indexer.log` for details. The log includes the word count and error message.

---

## Notes not being indexed

**Symptom:** A note exists in Obsidian but doesn't appear in the indexer output.

**Common causes:**

1. **Too short** — Notes under 20 words are skipped. Check the note content.

2. **Excluded pattern** — The note path may contain `_gsdata_`, `.gsbak`, `.git`, `.excalidraw.md`, etc. Check `EXCLUDE_PATTERNS` in `config.py`.

3. **Not a `.md` file** — Only Markdown files are indexed.

---

## Obsidian REST API connection refused

**Symptom:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Fix:**
1. Make sure Obsidian is open
2. Make sure the Local REST API plugin is enabled
3. Check the port in `.env` matches the plugin settings (default: 27123)
4. Test with curl: `curl -H "Authorization: Bearer YOUR_KEY" http://localhost:27123/vault/`

---

## Ollama timeout / embedding too slow

**Symptom:**
```
ReadTimeoutError: Read timed out. (read timeout=60)
```

**Fix:**
1. The client retries up to 3 times with exponential backoff (2s → 4s → 8s). If it still fails:
2. Check if Ollama is overloaded — run `ollama ps` to see loaded models
3. Stop unused models: `ollama stop <model-name>`
4. The first embedding call may be slow while `nomic-embed-text` loads into memory
5. You can increase `EMBED_TIMEOUT` (default 180s) in `src/obsidian_ai/llm_client.py`
6. Run the indexer again: `python -m obsidian_ai.indexer`

---

## Ollama connection refused

**Symptom:**
```
ConnectionRefusedError: [Errno 111] Connection refused
```

**Fix:**
1. Make sure Ollama is running: `ollama serve`
2. Check the URL in `.env` (default: `http://localhost:11434`)
3. Test with curl: `curl http://localhost:11434/api/tags`

---

## Embedding model not found

**Symptom:**
```
404 Not Found for http://localhost:11434/api/embeddings
```

**Fix:**
1. Pull the model: `ollama pull nomic-embed-text`
2. Verify: `ollama list`
3. Check `OLLAMA_EMBED_MODEL` in `.env` matches the pulled model name

---

## ChromaDB errors

**Symptom:** Errors related to ChromaDB collection or database.

**Fix:**
1. Stop the indexer
2. Delete the `./chroma_db` directory
3. Run the indexer again to rebuild from scratch

---

## Entity extraction fails

**Symptom:** Notes are indexed but have no entities. `entity_cache.json` may contain partial entries.

**Common causes:**

1. **LLM timeout** — Entity extraction needs `qwen3:8b` (or the configured chat model). If the model is slow, the request may time out. The system retries 3× with exponential backoff (1s, 2s, 4s).

2. **Model not pulled** — Pull the chat model: `ollama pull qwen3:8b`

3. **Invalid LLM response** — If the LLM returns malformed JSON, the entity is skipped with a warning log. The note is still indexed without entities.

**Fix:** Check `logs/indexer.log` for entity extraction errors. Use `--skip-entities` to bypass entirely if you don't need entities.

---

## Summary generation fails

**Symptom:** Notes are indexed but have no summary in search results.

**Common causes:**

1. **LLM timeout** — Same retry mechanism as entity extraction (3×, exponential backoff, 1s/2s/4s).

2. **Model not pulled** — Pull the chat model: `ollama pull qwen3:8b`

**Fix:** Check `logs/indexer.log` for summary errors. Use `--skip-summaries` to bypass.

---

## Concurrent indexing timeouts

**Symptom:** Multiple notes fail with timeouts during indexing, especially entity extraction or summary generation.

**Cause:** The indexer uses parallel workers (2-6 threads). Without rate limiting, Ollama can be overwhelmed by concurrent LLM requests.

**Fix:** The system now serializes LLM chat calls (entity extraction + summary generation) to at most 1 concurrent call via `_llm_chat_lock` (semaphore). If you still see timeouts:
1. Reduce `_embed_workers_for_current_machine()` in `indexer.py` (set a lower value)
2. Use `--skip-entities --skip-summaries` to skip LLM-dependent steps

---

## Chat/LLM errors (ask_vault, tag_notes, summarize_topic)

**Symptom:** LLM-powered tools return an error or empty response.

**Common causes:**

1. **Chat model not pulled** — Pull the chat model: `ollama pull qwen3:8b`

2. **LLM response not valid JSON** — The `tag_notes` tool expects JSON. If the LLM returns malformed JSON, the tool tries to extract it with regex.

3. **Context too long** — Notes are truncated to 3000 words each before sending to the LLM. If total context is still too large, Ollama may error. Try reducing `top_k`.

**Fix:** Check `logs/mcp_calls.log` for the full error details.

---

## Checking the error log

All indexing errors are logged to `logs/indexer.log` in the project root:

```bash
cat indexer.log
```

Each entry includes a timestamp, the note path, word count, and error message.

MCP tool errors are logged to `logs/mcp_calls.log`.
