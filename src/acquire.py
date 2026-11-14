"""Read the official OGB archive directly, without a torch dependency."""
import argparse, zipfile, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, sha, save_json, environment, now, Resource

ARCHIVE='https://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip'
TEXT='https://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz'

def acquire_dataset(root=ROOT/'data', download_text=True):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    archive=root/'arxiv.zip'
    if not archive.exists(): urllib.request.urlretrieve(ARCHIVE,archive)
    target=root/'raw'
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            dest=(target/member.filename).resolve()
            if not dest.is_relative_to(target.resolve()): raise ValueError('Unsafe archive path')
        if not (target/'arxiv'/'raw'/'edge.csv.gz').exists(): z.extractall(target)
    p=target/'arxiv'/'raw'
    with Resource() as resource:
        arrays={
            'edges':pd.read_csv(p/'edge.csv.gz',header=None,dtype=np.int64).to_numpy(),
            'features':pd.read_csv(p/'node-feat.csv.gz',header=None,dtype=np.float32).to_numpy(),
            'years':pd.read_csv(p/'node_year.csv.gz',header=None).to_numpy().ravel(),
            'labels':pd.read_csv(p/'node-label.csv.gz',header=None).to_numpy().ravel()}
    assert arrays['features'].shape == (169343,128), 'Unexpected release shape; investigate before proceeding'
    assert arrays['edges'].shape == (1166243,2), 'Unexpected release shape; investigate before proceeding'
    cache=root/'original_arrays'; cache.mkdir(exist_ok=True)
    for k,v in arrays.items(): np.save(cache/f'{k}.npy',v)
    files=[archive]+sorted(target.rglob('*.gz'))
    text_status='not requested'
    if download_text:
        try:
            if not (root/'titleabs.tsv.gz').exists(): urllib.request.urlretrieve(TEXT,root/'titleabs.tsv.gz')
            files.append(root/'titleabs.tsv.gz'); text_status='downloaded; mapping performed by diagnostics'
        except Exception as e: text_status=f'optional metadata unavailable: {type(e).__name__}: {e}'
    manifest=dict(dataset='ogbn-arxiv', source_url=ARCHIVE, text_url=TEXT, release=1,
        source='https://ogb.stanford.edu/docs/nodeprop/#ogbn-arxiv', license='ODC-BY',
        loader='direct official archive; no inverse edges added', verified_at_utc=now(),
        download_times='file modification time records acquisition completion; verified_at is not download time',
        arrays={k:dict(shape=v.shape,dtype=str(v.dtype)) for k,v in arrays.items()},
        files=[dict(path=str(f.relative_to(root)),bytes=f.stat().st_size,sha256=sha(f),download_or_extraction_mtime_utc=__import__('datetime').datetime.fromtimestamp(f.stat().st_mtime,__import__('datetime').timezone.utc).isoformat()) for f in files],
        title_status=text_status, parse=resource.fields(), environment=environment(),
        official_split='<=2017 / 2018 / >=2019; project uses <=2016 -> 2017 and <=2017 -> 2018')
    save_json(ROOT/'outputs'/'manifests'/'data_manifest.json',manifest)
    print('Acquired:', {k:v.shape for k,v in arrays.items()},flush=True)
    return arrays

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--root',default=str(ROOT/'data')); parser.add_argument('--dataset',default='ogbn-arxiv')
    args=parser.parse_args(); assert args.dataset=='ogbn-arxiv'; acquire_dataset(args.root)
