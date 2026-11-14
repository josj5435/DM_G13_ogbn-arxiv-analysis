import uuid
from pathlib import Path
import pytest
from src.common import ROOT
from src import reproduce


def test_reproduction_rejects_submission_and_nonempty_directory():
    with pytest.raises(ValueError): reproduce.prepare_workspace(ROOT)
    with pytest.raises(ValueError): reproduce.prepare_workspace(ROOT/'new_run')
    with pytest.raises(FileExistsError): reproduce.prepare_workspace(ROOT.parent)


def test_preparation_copies_inputs_without_saved_results(monkeypatch):
    base=ROOT/'outputs/semantic_test_artifacts'/uuid.uuid4().hex
    source=base/'source'; source.mkdir(parents=True)
    for folder in ['src','tests','configs','docs','data']:
        (source/folder).mkdir()
    for name in ['README.md','requirements-lock.txt','pytest.ini','.gitattributes','.gitignore','data/README.md']:
        (source/name).write_text('fixture',encoding='utf-8')
    for name in ['validation','final']:
        (source/'configs'/f'{name}.yaml').write_text('run_id: untouched\n',encoding='utf-8')
    (source/'outputs').mkdir(); (source/'outputs/protected.bin').write_bytes(b'preserve')
    monkeypatch.setattr(reproduce,'ROOT',source)
    target,run_id=reproduce.prepare_workspace(base/'new')
    assert run_id=='untouched' and not (target/'outputs').exists()
    assert (source/'outputs/protected.bin').read_bytes()==b'preserve'
    assert (target/'configs/final.yaml').read_bytes()==(source/'configs/final.yaml').read_bytes()
