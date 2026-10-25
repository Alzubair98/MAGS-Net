"""Restore the exact saved research reference locally, without overwriting conflicts."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    records = json.loads((ROOT / 'docs/reference_artifacts.json').read_text())
    tasks = []
    for record in records:
        source = ROOT / record['file']
        if sha(source) != record['sha256']:
            raise ValueError(f"Changed reference artifact: {record['file']}")
        relative = source.relative_to(ROOT / 'research_reference')
        target = ROOT / relative
        if target.exists() and (not target.is_file() or sha(target) != record['sha256']):
            raise FileExistsError(f'Refusing to replace a different file: {relative}')
        tasks.append((source, target))
    for source, target in tasks:
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    print(f'Restored/verified {len(tasks)} reference files. No model training was run.')


if __name__ == '__main__':
    main()
