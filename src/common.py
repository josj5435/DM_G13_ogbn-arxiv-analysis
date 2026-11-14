from pathlib import Path
import hashlib, json, platform, sys, time, threading
from datetime import datetime, timezone, timedelta
from importlib.metadata import version, PackageNotFoundError
import numpy as np
import pandas as pd
import psutil

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_RUN = 'arxiv_v2_20261106_review_20261107'

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, str(a.dtype))).encode() + a.tobytes()).hexdigest()

def jsonable(x):
    if isinstance(x, dict): return {str(k): jsonable(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray): return jsonable(x.tolist())
    if isinstance(x, np.generic): return jsonable(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    if isinstance(x, Path): return str(x)
    return x

def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def save_table(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if path.suffix == '.parquet': df.to_parquet(path, index=False)
    else: df.to_csv(path, index=False, encoding='utf-8-sig', na_rep='N/A')
    return df

def now(): return datetime.now(timezone.utc).isoformat()

def environment():
    packages = {}
    for name in ['numpy','scipy','pandas','scikit-learn','matplotlib','networkx','psutil','PyYAML','pyarrow','joblib','threadpoolctl','pytest','ogb']:
        try: packages[name] = version(name)
        except PackageNotFoundError: packages[name] = 'not installed (direct official archive)'
    return dict(python=sys.version, executable=sys.executable, packages=packages, os=platform.platform(), cpu=platform.processor(),
                logical_cpus=psutil.cpu_count(), physical_cpus=psutil.cpu_count(logical=False), ram_bytes=psutil.virtual_memory().total,
                utc=now(), kst=datetime.now(timezone(timedelta(hours=9))).isoformat())

class Resource:
    """Sample whole-process RSS every 50 ms; timings are wall time."""
    def __enter__(self):
        self.start=time.perf_counter(); self.peak=0; self.stop=threading.Event()
        def watch():
            proc=psutil.Process()
            while not self.stop.is_set():
                self.peak=max(self.peak, proc.memory_info().rss); self.stop.wait(.05)
        self.thread=threading.Thread(target=watch,daemon=True); self.thread.start(); return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join(); self.seconds=time.perf_counter()-self.start
    def fields(self): return dict(seconds=self.seconds, peak_rss_bytes=self.peak)

def load_data():
    p=ROOT/'data'/'processed'
    return {k:np.load(p/f'{k}.npy',mmap_mode='r') for k in ['edges','years','features','labels','feature_mask','year_mask']}

def load_config(path):
    import yaml
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))
