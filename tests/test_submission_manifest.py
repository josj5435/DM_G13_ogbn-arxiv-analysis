import json
import uuid
import pytest
from src.common import ROOT
from src.submission_manifest import DOCUMENTS, create_manifest, verify_manifest


@pytest.fixture
def package():
    root = ROOT/'outputs/semantic_test_artifacts'/uuid.uuid4().hex
    for name in DOCUMENTS:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# Project\n', encoding='utf-8')
    for name in ['outputs/run/models/k.joblib', 'outputs/run/results.csv', 'configs/final.yaml', 'src/main.py']:
        path = root/name; path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('saved artifact\n', encoding='utf-8')
    (root/'manifest.json').write_text(json.dumps(create_manifest(root)), encoding='utf-8')
    return root


def test_editable_documents_keep_experiment_integrity(package):
    for name in DOCUMENTS:
        (package/name).write_text('# Team\nGitHub URL: https://example.org/project\n', encoding='utf-8')
    (package/'README.md').write_text('[Methods](docs/METHODS.md)\n', encoding='utf-8')
    result = verify_manifest(package)
    assert result == {'hashed_files': 4, 'relative_links': 1}
    assert all(not name.endswith('.md') for name in create_manifest(package)['files'])


@pytest.mark.parametrize('name', ['outputs/run/models/k.joblib', 'outputs/run/results.csv'])
def test_core_artifact_changes_are_rejected(package, name):
    (package/name).write_bytes(b'changed')
    with pytest.raises(AssertionError, match='hash_mismatches'):
        verify_manifest(package)


def test_missing_documents_and_broken_links_are_rejected(package):
    original = (package/'docs/METHODS.md').read_text(encoding='utf-8')
    (package/'docs/METHODS.md').unlink()
    with pytest.raises(AssertionError, match='missing_documents'):
        verify_manifest(package)
    (package/'docs/METHODS.md').write_text(original, encoding='utf-8')
    (package/'README.md').write_text('[Missing](docs/absent.md)', encoding='utf-8')
    with pytest.raises(AssertionError, match='broken_relative_links'):
        verify_manifest(package)


def test_manifest_cannot_omit_experiment_files(package):
    manifest = create_manifest(package)
    manifest['files'].pop('outputs/run/results.csv')
    (package/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(AssertionError, match='unlisted'):
        verify_manifest(package)
