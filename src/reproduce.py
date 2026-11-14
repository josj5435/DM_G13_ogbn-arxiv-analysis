"""Run the documented pipeline in a fresh workspace, preserving submitted results."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
from .common import ROOT, load_config


def pipeline_commands(run_id):
    return [
        ['-m','src.acquire','--root','data'],
        ['-m','src.preprocess'],
        ['-m','pytest','-q','-p','no:cacheprovider'],
        ['-m','src.run','--stage','validation','--config','configs/validation.yaml'],
        ['-m','src.run','--stage','final','--config','configs/final.yaml'],
        ['-m','src.diagnostics','--run-id',run_id],
        ['-m','src.benchmark','--run-id',run_id],
        ['-m','src.make_figures','--run-id',run_id,'--mode','both'],
        ['-m','src.summarize','--run-id',run_id],
    ]


def prepare_workspace(target):
    target=Path(target).resolve()
    if target==ROOT or target.is_relative_to(ROOT):
        raise ValueError('Use a separate workspace outside this submission directory')
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise FileExistsError('The reproduction workspace must be new or empty')
    target.mkdir(parents=True,exist_ok=True)
    for folder in ['src','tests','configs','docs']:
        shutil.copytree(ROOT/folder,target/folder,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    for name in ['README.md','requirements-lock.txt','pytest.ini','.gitattributes','.gitignore']:
        shutil.copy2(ROOT/name,target/name)
    if (ROOT/'VALIDATION.md').exists():
        shutil.copy2(ROOT/'VALIDATION.md',target/'VALIDATION.md')
    (target/'data').mkdir(exist_ok=True)
    shutil.copy2(ROOT/'data/README.md',target/'data/README.md')
    run_id=load_config(target/'configs/validation.yaml')['run_id']
    assert load_config(target/'configs/final.yaml')['run_id']==run_id
    return target,run_id


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true',help='Copy code/configs and print commands without downloading or training')
    args=p.parse_args(); target,run_id=prepare_workspace(args.workspace)
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: env[key]='4'
    env.pop('PYTHONPATH',None)
    for command in pipeline_commands(run_id):
        print('python '+' '.join(command),flush=True)
        if not args.prepare_only:
            subprocess.run([sys.executable,*command],cwd=target,env=env,check=True)
    print(f'Workspace: {target}',flush=True)


if __name__=='__main__': main()
