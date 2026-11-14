"""Create and verify the submission's experiment-file integrity manifest."""
import argparse
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
from .common import ROOT, sha

DOCUMENTS = (
    'README.md', 'VALIDATION.md', 'data/README.md', 'docs/METHODS.md',
    'docs/REFERENCES.md', 'docs/AI_USE.md', 'handoff/RESULTS_SUMMARY_KO.md',
)
IGNORED = {'.git', '.venv', '.repro_venv', '__pycache__', '.pytest_cache', 'semantic_test_artifacts',
              '.DS_Store', 'Thumbs.db', 'desktop.ini'}


def submission_files(root):
    return sorted(p for p in Path(root).rglob('*') if p.is_file()
                  and not IGNORED.intersection(p.relative_to(root).parts))


def is_hash_target(relative):
    path = Path(relative)
    if path.suffix.lower() == '.md':
        return False
    return (path.parts[0] in {'src', 'tests', 'configs', 'outputs'}
            or path.as_posix() in {'requirements-lock.txt', 'pytest.ini'})


def hash_targets(root):
    return {p.relative_to(root).as_posix(): p for p in submission_files(root)
            if is_hash_target(p.relative_to(root))}


def create_manifest(root=ROOT):
    return dict(schema=2, description='SHA-256 for executable code, environment and experiment artifacts; Markdown is editable.',
                files={name: sha(path) for name, path in hash_targets(root).items()})


def check_documents(root):
    root = Path(root).resolve()
    missing = [name for name in DOCUMENTS if not (root/name).is_file()]
    assert not missing, dict(missing_documents=missing)
    broken = []; count = 0
    for path in submission_files(root):
        if path.suffix.lower() != '.md':
            continue
        content = re.sub(r'```.*?```', '', path.read_text(encoding='utf-8-sig'), flags=re.S)
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', content):
            target = target.strip().split(' "', 1)[0].strip('<>')
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            destination = (path.parent/unquote(parsed.path)).resolve()
            count += 1
            if not destination.is_relative_to(root) or not destination.exists():
                broken.append((path.relative_to(root).as_posix(), target))
    assert not broken, dict(broken_relative_links=broken)
    return count


def verify_manifest(root=ROOT):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    assert manifest.get('schema') == 2, 'Unsupported manifest schema'
    targets = hash_targets(root)
    recorded = manifest['files']
    assert set(recorded) == set(targets), dict(missing=sorted(set(recorded)-set(targets)),
                                             unlisted=sorted(set(targets)-set(recorded)))
    mismatched = [name for name, path in targets.items() if sha(path) != recorded[name]]
    assert not mismatched, dict(hash_mismatches=mismatched)
    return dict(hashed_files=len(targets), relative_links=check_documents(root))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    check_documents(ROOT)
    (ROOT/'manifest.json').write_text(json.dumps(create_manifest(), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('manifest.json')
