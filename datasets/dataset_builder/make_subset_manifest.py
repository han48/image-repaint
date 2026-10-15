"""Sample a balanced 5k subset from train_manifest.json (~555 entries per style)."""
import json, random, pathlib
from collections import defaultdict

random.seed(42)
src = pathlib.Path('datasets/sketches/train_manifest.json')
dst = pathlib.Path('datasets/sketches/train_manifest_5k.json')

with open(src) as f:
    entries = json.load(f)

# Group by style
by_style = defaultdict(list)
for e in entries:
    by_style[e['style']].append(e)

per_style = 5000 // 9  # ~555
subset = []
for style_id, items in sorted(by_style.items()):
    sampled = random.sample(items, min(per_style, len(items)))
    subset.extend(sampled)

# Top up to exactly 5000 if needed
remaining = [e for e in entries if e not in subset]
random.shuffle(remaining)
subset.extend(remaining[:max(0, 5000 - len(subset))])
subset = subset[:5000]
random.shuffle(subset)

with open(dst, 'w') as f:
    json.dump(subset, f)
print(f'Saved {len(subset)} entries to {dst}')
print(f'Style distribution: {dict(sorted((s, sum(1 for e in subset if e["style"]==s)) for s in range(9)))}')
