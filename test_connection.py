"""Quick test: verify the .env loads and the Obsidian REST API responds."""

# Test 1: Check config loads the API key
from src.obsidian_ai import config
print(f"[CONFIG] OBSIDIAN_API_KEY length: {len(config.obsidian_api_key)}")
print(f"[CONFIG] OBSIDIAN_API_KEY non-empty: {bool(config.obsidian_api_key)}")
print(f"[CONFIG] Host: {config.obsidian_host}:{config.obsidian_port}")

if not config.obsidian_api_key:
    print("[FAIL] API key is empty — .env not loading correctly!")
    exit(1)

# Test 2: Try listing notes via the Obsidian REST API
from src.obsidian_ai import obsidian_client
try:
    notes = obsidian_client.list_all_notes()
    print(f"[OK] list_all_notes returned {len(notes)} notes!")
    for n in notes[:5]:
        print(f"     - {n}")
    if len(notes) > 5:
        print(f"     ... and {len(notes) - 5} more")
except Exception as e:
    print(f"[FAIL] list_all_notes failed: {e}")
    exit(1)
