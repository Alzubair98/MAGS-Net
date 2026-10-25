"""Dataset-free static, artifact and patient-fold integrity checks. No training."""
import ast
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    for path in ROOT.rglob('*.py'):
        if not any(p in {'.venv', '.git'} for p in path.parts):
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path.relative_to(ROOT)))
    notebooks = list(ROOT.glob('*.ipynb'))
    assert len(notebooks) == 6
    for path in notebooks:
        book = json.loads(path.read_text(encoding='utf-8'))
        for cell in book['cells']:
            if cell['cell_type'] != 'code':
                assert not cell.get('attachments')
                continue
            assert not cell['outputs'] and cell['execution_count'] is None, path.name
            for node in ast.parse(''.join(cell['source'])).body:
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                    if node.targets[0].id.startswith('RUN_') and isinstance(node.value, ast.Constant):
                        assert node.value.value is False, (path.name, node.targets[0].id)
    records = json.loads((ROOT / 'docs/reference_artifacts.json').read_text())
    for record in records:
        assert sha(ROOT / record['file']) == record['sha256'], record['file']
    manifest = json.loads((ROOT / 'magsnet_demo/checkpoint_manifest.json').read_text())
    study = ROOT / manifest['study_dir']
    experiment = json.loads((study / 'experiment.json').read_text())
    protocol = experiment['protocol']
    assert hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest() == manifest['experiment_hash']
    definitions = ('conv_bn_relu', 'Stem', 'MAGBlock', 'MAGSStage', 'SegmentationDecoder', 'ClassificationHead', 'MAGSNet')
    nodes = [n for n in ast.parse((ROOT / manifest['model_source']).read_text(encoding='utf-8')).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in definitions]
    assert tuple(n.name for n in nodes) == definitions
    fingerprint = {n.name: ast.dump(n, include_attributes=False).replace(', type_params=[]', '') for n in nodes}
    assert hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest() == manifest['model_definition_sha256']
    outer, inner = protocol['outer'], protocol['inner']
    patient_folds = {row['patient_id']: row['outer_fold'] for row in outer}
    assert len(patient_folds) == len(outer) == 233
    prep = ROOT / 'research_reference/results/magsnet_nested_cv/preparation/04ddbd02826f'
    samples = csv_rows(prep / 'sample_index.csv')
    expected = {row['sample_id']: row for row in samples}
    assert len(expected) == len(samples) == 3064
    seen = set()
    for record in manifest['checkpoints']:
        fold = record['outer_fold']
        checkpoint = study / record['relative_path']
        assert sha(checkpoint) == record['sha256']
        development = {p for p, f in patient_folds.items() if f != fold}
        heldout = {p for p, f in patient_folds.items() if f == fold}
        assert not development & heldout
        local_inner = [row for row in inner if row['outer_fold'] == fold]
        assert {r['patient_id'] for r in local_inner} == development
        assert len(local_inner) == len(development)
        for inner_fold in range(1, 6):
            validation = {r['patient_id'] for r in local_inner if r['inner_fold'] == inner_fold}
            train = development - validation
            assert validation and train and not (validation & train or heldout & (validation | train))
        directory = checkpoint.parent.parent
        provenance = json.loads((directory / 'evaluation.json').read_text())
        assert provenance['predictions_hash'] == sha(directory / 'outer_predictions.csv')
        assert provenance['checkpoint_hash'] == record['sha256']
        rows = csv_rows(directory / 'outer_predictions.csv')
        for row in rows:
            sid, patient = row['sample_id'], row['patient_id']
            assert sid not in seen and sid in expected
            assert patient in heldout and expected[sid]['patient_id'] == patient
            assert row['label'] == expected[sid]['label']
            seen.add(sid)
    assert seen == set(expected)
    secrets = re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----')
    for path in ROOT.rglob('*'):
        if path.is_file() and path.suffix in {'.py', '.md', '.json', '.ipynb', '.toml', '.txt', '.cff'} and path.name != 'check_release.py':
            text = path.read_text(encoding='utf-8')
            assert not secrets.search(text), f'Potential secret: {path.relative_to(ROOT)}'
            assert not re.search(r'[A-Za-z]:[\\/]+Users[\\/]', text), f'Local user path: {path.relative_to(ROOT)}'
    print(json.dumps({'status': 'passed', 'clean_notebooks': len(notebooks), 'reference_files': len(records),
                      'outer_models': 5, 'outer_folds': 5, 'inner_partitions': 25,
                      'oof_slices': len(seen), 'unique_patients': len(patient_folds),
                      'duplicate_samples': 0, 'patient_overlap': 0, 'training_run': False}, indent=2))


if __name__ == '__main__':
    main()
