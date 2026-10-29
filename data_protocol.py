"""Exact assignment splits and fail-closed audits; no model dependencies."""
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from urllib.request import urlopen

SOURCE_URL = 'https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Gift_Cards.jsonl.gz'
LABELS = ('POSITIVE', 'NEUTRAL', 'NEGATIVE')
RANGES = {'train': (1000, 19999), 'val': (20000, 29999), 'test': (30000, 39999)}
FIELDS = ('filtered_index', 'original_row', 'reference_label', 'rating', 'title', 'text')

def label_for(rating):
    assert rating in (1, 2, 3, 4, 5), f'Unexpected rating: {rating!r}'
    return 'POSITIVE' if rating >= 4 else 'NEUTRAL' if rating == 3 else 'NEGATIVE'

def canonical_records(path):
    filtered_index = 0
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        for original_row, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            text = row.get('text') or ''
            assert isinstance(text, str)
            if len(text.strip()) < 3:
                continue
            if filtered_index > 39999:
                break
            title = row.get('title') or ''
            assert isinstance(title, str)
            yield dict(filtered_index=filtered_index, original_row=original_row,
                       rating=float(row['rating']), reference_label=label_for(row['rating']),
                       title=title, text=text)
            filtered_index += 1

def select_splits(records):
    pools = {name: {label: [] for label in LABELS} for name in RANGES}
    for row in records:
        for name, (lo, hi) in RANGES.items():
            if lo <= row['filtered_index'] <= hi:
                pools[name][row['reference_label']].append(row)
    rng = random.Random(43)
    train = []
    for label in LABELS:
        train.extend(rng.sample(pools['train'][label], 300))
    rng.shuffle(train)
    result = {'train': train}
    for name in ('val', 'test'):
        selected = []
        for label in LABELS:
            assert len(pools[name][label]) >= 100
            selected.extend(pools[name][label][:100])
        result[name] = sorted(selected, key=lambda row: row['filtered_index'])
    return result

def fingerprint(rows):
    # Stable across irrelevant metadata and ordering; text content remains exact.
    normalized = [{k: (float(r[k]) if k == 'rating' else r[k]) for k in FIELDS}
                  for r in sorted(rows, key=lambda r: r['filtered_index'])]
    return hashlib.sha256(json.dumps(normalized, ensure_ascii=False,
                                    sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def audit_splits(actual, expected):
    report = {}
    for name, rows in actual.items():
        per_class = 300 if name == 'train' else 100
        assert len(rows) == 3 * per_class, (name, len(rows))
        ids = [r['filtered_index'] for r in rows]
        originals = [r['original_row'] for r in rows]
        assert len(ids) == len(set(ids)), (name, 'duplicate filtered_index')
        assert len(originals) == len(set(originals)), (name, 'duplicate original_row')
        lo, hi = RANGES[name]
        assert all(lo <= i <= hi for i in ids), (name, 'range violation')
        counts = Counter(r['reference_label'] for r in rows)
        assert counts == {label: per_class for label in LABELS}, (name, counts)
        assert set(ids) == {r['filtered_index'] for r in expected[name]}, (name, 'wrong prescribed IDs')
        exp = {r['filtered_index']: r for r in expected[name]}
        for row in rows:
            assert len(row['text'].strip()) >= 3
            assert row['reference_label'] == label_for(row['rating'])
            assert all(row[k] == exp[row['filtered_index']][k] for k in FIELDS), (name, row['filtered_index'], 'content mismatch')
        report[name] = dict(size=len(rows), class_counts=dict(counts),
                            content_sha256=fingerprint(rows), unique_ids=True,
                            prescribed_ids_and_contents_match=True)
    for left, right in (('train','val'), ('train','test'), ('val','test')):
        for key in ('filtered_index', 'original_row'):
            assert not ({r[key] for r in actual[left]} & {r[key] for r in actual[right]})
    report['disjoint'] = True
    return report

def download_source(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        partial = path.with_suffix('.partial')
        with urlopen(SOURCE_URL, timeout=120) as response, partial.open('wb') as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        partial.replace(path)
    return path

def load_jsonl(path):
    with open(path, encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]
