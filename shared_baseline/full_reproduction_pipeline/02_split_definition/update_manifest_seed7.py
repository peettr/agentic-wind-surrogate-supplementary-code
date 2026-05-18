"""Update split_manifest.json: change val/holdout split to seed=7.

Reads current manifest, takes the 102 test cases (val+holdout from seed=1),
re-splits them with random seed=7 (50/50), and writes back.
"""
import json, random, sys
from pathlib import Path

MANIFEST_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "shared" / "data" / "split_manifest.json"

with open(MANIFEST_PATH) as f:
    manifest = json.load(f)

# Get seed=1's test cases (val + holdout = all 102 test)
s1 = manifest["seeds"]["1"]
test_cases = sorted(set(s1["val"]) | set(s1["holdout"]))
print(f"Total test cases: {len(test_cases)}")

# Split with seed=7
random.seed(7)
shuffled = test_cases.copy()
random.shuffle(shuffled)
mid = len(shuffled) // 2
new_val = sorted(shuffled[:mid])
new_holdout = sorted(shuffled[mid:])

print(f"New val: {len(new_val)} cases")
print(f"New holdout: {len(new_holdout)} cases")

# Verify no overlap
assert len(set(new_val) & set(new_holdout)) == 0
assert len(set(new_val) | set(new_holdout)) == len(test_cases)

# Update all seeds to use seed=7 val/holdout split
# (train stays the same per seed, val/holdout are re-split)
manifest["val_holdout_seed"] = 7

for seed_key in manifest["seeds"]:
    sp = manifest["seeds"][seed_key]
    # Keep original train, update val/holdout to seed=7 split
    sp["val"] = new_val
    sp["holdout"] = new_holdout

# Backup original
backup_path = MANIFEST_PATH.with_suffix(".json.seed42_backup")
if not backup_path.exists():
    with open(backup_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Backup written to {backup_path}")

# Write updated manifest
with open(MANIFEST_PATH, "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Updated {MANIFEST_PATH} with seed=7 val/holdout split")
print(f"First 5 val: {new_val[:5]}")
print(f"First 5 holdout: {new_holdout[:5]}")



