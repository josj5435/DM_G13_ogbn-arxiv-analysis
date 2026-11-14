import argparse
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from .common import ROOT, save_json, save_table, array_hash, load_data, Resource

def valid_years(years): return np.isfinite(years)&(years>=1800)&(years<=2026)&(years==np.floor(years))

def clean_edges(edges,years):
    n=len(years); e=np.asarray(edges).copy(); stats={'raw_nodes':n,'raw_edges':len(e)}
    keep=(e>=0).all(axis=1)&(e<n).all(axis=1)
    stats['invalid_endpoints']=int((~keep).sum()); e=e[keep]
    keep=e[:,0]!=e[:,1]; stats['self_loops']=int((~keep).sum()); e=e[keep]
    before=len(e); e=np.unique(e,axis=0); stats['duplicate_edges']=before-len(e)
    ym=valid_years(years); stats['invalid_year_nodes']=int((~ym).sum())
    keep=ym[e].all(axis=1); stats['invalid_year_edges']=int((~keep).sum()); e=e[keep]
    keep=years[e[:,0]]>=years[e[:,1]]; stats['reversed_year_edges']=int((~keep).sum()); e=e[keep]
    stats['clean_edges']=len(e); stats['same_year_edges']=int((years[e[:,0]]==years[e[:,1]]).sum())
    return e,stats

def build_snapshot(edges,years,cutoff):
    nodes=np.flatnonzero(valid_years(years)&(years<=cutoff)); mask=np.zeros(len(years),bool); mask[nodes]=True
    e=edges[mask[edges].all(axis=1)]; assert np.all(years[e]<=cutoff)
    return nodes,e

def build_baskets(nodes,edges):
    buckets={int(i):[] for i in nodes}
    for s,t in edges: buckets[int(s)].append(int(t))
    return {s:tuple(sorted(set(v))) for s,v in buckets.items()}

def eval_baskets(edges,years,cutoff):
    nodes=np.flatnonzero(years==cutoff+1)
    e=edges[(years[edges[:,0]]==cutoff+1)&(years[edges[:,1]]<=cutoff)]
    return build_baskets(nodes,e),e

def make_queries(baskets,seed,min_refs=5,ratio=.2):
    rng=np.random.default_rng(seed); queries=[]
    for q,b in sorted(baskets.items()):
        if len(b)<min_refs: continue
        h=sorted(rng.choice(np.array(b),size=max(1,int(np.floor(ratio*len(b)))),replace=False).tolist())
        o=sorted(set(b)-set(h)); assert not set(o)&set(h)
        queries.append(dict(query_id=q,mask_seed=seed,observed=o,hidden=h,n_refs=len(b)))
    return queries

def preprocess():
    p=ROOT/'data'/'original_arrays'; out=ROOT/'data'/'processed'; out.mkdir(parents=True,exist_ok=True)
    a={k:np.load(p/f'{k}.npy',mmap_mode='r') for k in ['edges','features','years','labels']}
    with Resource() as r:
        edges,stats=clean_edges(a['edges'],a['years'])
        mask=np.isfinite(a['features']).all(axis=1)&(np.linalg.norm(a['features'],axis=1)>0)
        stats['invalid_features']=int((~mask).sum()); stats['invalid_labels']=int((~np.isfinite(a['labels'])|(a['labels']<0)).sum())
        for k,v in dict(a,edges=edges,feature_mask=mask,year_mask=valid_years(a['years'])).items(): np.save(out/f'{k}.npy',v)
        for cutoff in [2016,2017]:
            nodes,e=build_snapshot(edges,a['years'],cutoff)
            inv=np.full(len(a['years']),-1); inv[nodes]=np.arange(len(nodes))
            adjacency=csr_matrix((np.ones(len(e)),(inv[e[:,0]],inv[e[:,1]])),shape=(len(nodes),len(nodes)))
            nc,cc=connected_components(adjacency,directed=False)
            indeg=np.bincount(e[:,1],minlength=len(a['years']))[nodes]; outdeg=np.bincount(e[:,0],minlength=len(a['years']))[nodes]
            stats.update({f'{cutoff}_nodes':len(nodes),f'{cutoff}_edges':len(e),f'{cutoff}_isolates':int(((indeg+outdeg)==0).sum()),f'{cutoff}_dangling':int((outdeg==0).sum()),f'{cutoff}_weak_components':nc,f'{cutoff}_outside_largest_component':len(nodes)-int(np.bincount(cc).max())})
            dest=ROOT/'outputs'/'snapshots'; dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f'{cutoff}.npz',nodes=nodes,edges=e)
            b,qe=eval_baskets(edges,a['years'],cutoff)
            eligible=sum(len(v)>=5 for v in b.values())
            stats[f'{cutoff+1}_queries_total']=len(b); stats[f'{cutoff+1}_queries_eligible']=eligible
            all_eval=edges[a['years'][edges[:,0]]==cutoff+1]
            stats[f'{cutoff+1}_query_refs_outside_candidate']=len(all_eval)-len(qe)
            for seed in range(5): save_table(ROOT/'outputs'/'splits'/f'queries_{cutoff}_{seed}.parquet',make_queries(b,seed))
            save_table(ROOT/'outputs'/'tables'/f'degrees_{cutoff}.csv',dict(node_id=nodes,in_degree=indeg,out_degree=outdeg))
    stats.update(r.fields()); save_table(ROOT/'outputs'/'tables'/'preprocessing_stats.csv',[stats])
    save_json(out/'manifest.json',dict(stats=stats,edges_hash=array_hash(edges),feature_mask_hash=array_hash(mask),year_policy='integer 1800..2026 inclusive'))
    print('Preprocessed:',stats,flush=True)

if __name__=='__main__':
    argparse.ArgumentParser(description='Prepare the fixed temporal snapshots and query masks.').parse_args()
    preprocess()
