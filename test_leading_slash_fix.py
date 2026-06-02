"""Test that leading-slash fixes work correctly against the live Obsidian API."""

import sys
sys.path.insert(0, "src")

from obsidian_ai import obsidian_client

# ── Test 1: list_folder("/") should work (was failing with ////) ──
print("=" * 60)
print("TEST 1: list_folder('/')")
print("=" * 60)
try:
    notes = obsidian_client.list_folder("/")
    print(f"  OK — returned {len(notes)} notes")
    for n in notes[:5]:
        print(f"       - {n}")
    if len(notes) > 5:
        print(f"       ... and {len(notes) - 5} more")
except Exception as e:
    print(f"  FAIL — {e}")

# ── Test 2: list_folder("/Projects") (or some folder) ──
print()
print("=" * 60)
print("TEST 2: list_folder('/Projects')")
print("=" * 60)
try:
    notes = obsidian_client.list_folder("/Projects")
    print(f"  OK — returned {len(notes)} notes")
    for n in notes[:5]:
        print(f"       - {n}")
    if len(notes) > 5:
        print(f"       ... and {len(notes) - 5} more")
except Exception as e:
    print(f"  FAIL — {e}")

# ── Test 3: list_folder("/") root (alternative presentation) ──
print()
print("=" * 60)
print("TEST 3: list_folder('') (empty path = root)")
print("=" * 60)
try:
    notes = obsidian_client.list_folder("")
    print(f"  OK — returned {len(notes)} notes")
    for n in notes[:5]:
        print(f"       - {n}")
    if len(notes) > 5:
        print(f"       ... and {len(notes) - 5} more")
except Exception as e:
    print(f"  FAIL — {e}")

# ── Test 4: get_note("/Notes/Test") ──
print()
print("=" * 60)
print("TEST 4: get_note('/Notes/Test')")
print("=" * 60)
try:
    content = obsidian_client.get_note("/Notes/Test")
    print(f"  OK — returned {len(content)} chars")
except Exception as e:
    print(f"  FAIL — {e}")

# ── Test 5: get_note("Notes/Test") (no leading slash, should still work) ──
print()
print("=" * 60)
print("TEST 5: get_note('Notes/Test') (no leading slash)")
print("=" * 60)
try:
    content = obsidian_client.get_note("Notes/Test")
    print(f"  OK — returned {len(content)} chars")
except Exception as e:
    print(f"  FAIL — {e}")

print()
print("Done.")
