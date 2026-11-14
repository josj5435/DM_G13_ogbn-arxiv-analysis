import itertools, time
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from .common import Resource,save_json,save_table,array_hash,sha
from .evaluate import common_silhouette
from .pairs import external_patterns

def normalize_features(features,ids):
    x=np.asarray(features[ids],dtype=np.float64)
    return x/np.linalg.norm(x,axis=1,keepdims=True)

def select_configuration(rows,cfg):
    eligible=[r for r in rows if np.isfinite(r['silhouette']) and (r['d']==128 or r['explained_variance']>=cfg['variance_min'])]
    if not eligible: raise RuntimeError('No eligible validation cluster model; redesign before final')
    best=max(r['silhouette'] for r in eligible)
    return sorted((r for r in eligible if r['silhouette']>=best-cfg['silhouette_tolerance']),key=lambda r:(r['d'],r['k'],-r['silhouette']))[0]

def run_clusters(data,cutoff,out,cfg,stage,selection=None,pairs=None,baskets=None,eval_edges=None):
    out.mkdir(parents=True,exist_ok=True); modeldir=out/'models'; modeldir.mkdir(exist_ok=True)
    train_ids=np.flatnonzero(data['feature_mask']&data['year_mask']&(data['years']<=cutoff))
    eval_ids=np.flatnonzero(data['feature_mask']&(data['years']==cutoff+1))
    x=normalize_features(data['features'],train_ids); y=normalize_features(data['features'],eval_ids)
    sample=np.sort(np.random.default_rng(cfg['silhouette_seed']).choice(len(y),min(len(y),cfg['silhouette_sample']),replace=False))
    np.savez_compressed(modeldir/'ids_masks.npz',train_ids=train_ids,eval_ids=eval_ids,silhouette_ids=eval_ids[sample],feature_mask=data['feature_mask'])
    with Resource() as pca_res: pca=PCA(n_components=128,svd_solver='full',whiten=False).fit(x)
    joblib.dump(pca,modeldir/'pca.joblib'); np.savez_compressed(modeldir/'pca.npz',mean=pca.mean_,components=pca.components_,explained_variance=pca.explained_variance_,explained_variance_ratio=pca.explained_variance_ratio_)
    save_json(modeldir/'normalization.json',dict(rule='per-paper L2, float64; no post-PCA normalization',train_ids_hash=array_hash(train_ids),feature_mask_hash=array_hash(data['feature_mask']),pca_fit=pca_res.fields()))
    with Resource() as tr_res: xp=pca.transform(x); yp=pca.transform(y)
    dims=cfg['dimensions']; dist_rng=np.random.default_rng(303)
    aa=dist_rng.integers(len(x),size=50000); bb=dist_rng.integers(len(x),size=50000); same=aa==bb
    while same.any(): bb[same]=dist_rng.integers(len(x),size=same.sum()); same=aa==bb
    np.savez_compressed(modeldir/'distance_pairs.npz',node_a=train_ids[aa],node_b=train_ids[bb])
    diagnostics=[]
    for d in dims:
        tx=x if d==128 else xp[:,:d]; ty=y if d==128 else yp[:,:d]
        distances=np.linalg.norm(tx[aa]-tx[bb],axis=1)
        mse_train=0. if d==128 else float(np.mean((x-(tx@pca.components_[:d]+pca.mean_))**2))
        mse_eval=0. if d==128 else float(np.mean((y-(ty@pca.components_[:d]+pca.mean_))**2))
        diagnostics.append(dict(stage=stage,cutoff=cutoff,evaluation_year=cutoff+1,d=d,n_train=len(x),n_eval=len(y),explained_variance=1. if d==128 else float(pca.explained_variance_ratio_[:d].sum()),reconstruction_mse_train=mse_train,reconstruction_mse_eval=mse_eval,distance_mean=distances.mean(),distance_sd=distances.std(),distance_cv=distances.std()/distances.mean(),distance_p05=np.quantile(distances,.05),distance_p50=np.quantile(distances,.5),distance_p95=np.quantile(distances,.95),pca_fit_seconds=0 if d==128 else pca_res.seconds,pca_transform_seconds=0 if d==128 else tr_res.seconds))
    save_table(out/'dimension_diagnostics.csv',diagnostics)
    if stage=='validation': configs=list(itertools.product(cfg['ks'],dims,[0]))
    else:
        kd={(selection['k'],d) for d in dims}|{(k,selection['d']) for k in cfg['ks']}
        configs=[(k,d,s) for k,d in sorted(kd) for s in cfg['final_seeds']]
    rows=[]; patterns=[]; cluster_frames=[]; assignments={}; stability=[]
    for k,d,seed in configs:
        tag=f'k{k}_d{d}_s{seed}'; tx=x if d==128 else xp[:,:d]; ty=y if d==128 else yp[:,:d]
        with Resource() as fitres:
            km=KMeans(n_clusters=k,init='k-means++',n_init=cfg['n_init'],max_iter=cfg['max_iter'],tol=cfg['tol'],random_state=seed,algorithm='lloyd').fit(tx)
        with Resource() as predres: pred=km.predict(ty)
        mpath=modeldir/f'{tag}.joblib'; joblib.dump(km,mpath); model_hash=sha(mpath)
        assigned=np.full(len(data['years']),-1,dtype=np.int16); assigned[train_ids]=km.labels_; assigned[eval_ids]=pred
        np.savez_compressed(modeldir/f'{tag}_assignments.npz',assignment=assigned)
        assignments[(k,d,seed)]=pred
        sizes=np.bincount(pred,minlength=k)
        row=dict(stage=stage,cutoff=cutoff,evaluation_year=cutoff+1,k=k,d=d,seed=seed,model_hash=model_hash,feature_mask_hash=array_hash(data['feature_mask']),n_train=len(train_ids),n_eval=len(eval_ids),n_silhouette=len(sample),silhouette=common_silhouette(y[sample],pred[sample]),explained_variance=1. if d==128 else float(pca.explained_variance_ratio_[:d].sum()),inertia=float(km.inertia_),iterations=int(km.n_iter_),converged=km.n_iter_<cfg['max_iter'],distinct_clusters=int((sizes>0).sum()),singleton_clusters=int((sizes==1).sum()),largest_cluster_fraction=float(sizes.max()/len(pred)),fit_seconds=fitres.seconds,predict_seconds=predres.seconds,peak_rss_bytes=fitres.peak,pca_fit_seconds=0 if d==128 else pca_res.seconds,pca_transform_seconds=0 if d==128 else tr_res.seconds,pipeline_seconds=fitres.seconds+predres.seconds+(0 if d==128 else pca_res.seconds+tr_res.seconds),method='kmeans')
        # Validation labels are deliberately not accessed; labels are final-only diagnostics.
        if stage=='final':
            valid=np.isfinite(data['labels'][eval_ids])&(data['labels'][eval_ids]>=0)
            row['ari']=float(adjusted_rand_score(data['labels'][eval_ids][valid],pred[valid])); row['n_labeled']=int(valid.sum())
        rows.append(row)
        for split,ids,labels in [('train',train_ids,km.labels_),('evaluation',eval_ids,pred)]:
            cluster_frames.append(pd.DataFrame(dict(node_id=ids,cluster_id=labels,split=split,stage=stage,cutoff=cutoff,evaluation_year=cutoff+1,k=k,d=d,seed=seed,model_hash=model_hash)))
        if stage=='final':
            for random_seed in range(5):
                random_labels=np.random.default_rng(random_seed).permutation(pred)
                rows.append(dict(row,method='B4_random',random_seed=random_seed,silhouette=common_silhouette(y[sample],random_labels[sample]),ari=float(adjusted_rand_score(data['labels'][eval_ids][valid],random_labels[valid]))))
            for item in external_patterns(pairs,eval_ids,train_ids,assigned,baskets,eval_edges,data['feature_mask'],seed): patterns.append(dict(stage=stage,k=k,d=d,seed=seed,cutoff=cutoff,evaluation_year=cutoff+1,**item))
        save_table(out/'clustering_metrics.csv',rows)
        save_table(out/'clusters.parquet',pd.concat(cluster_frames,ignore_index=True))
        if patterns: save_table(out/'cluster_citation_patterns.csv',patterns)
        print(f'{stage} clustering {tag}: silhouette={row["silhouette"]:.6f}; fit={fitres.seconds:.1f}s',flush=True)
    for k,d in sorted({(k,d) for k,d,s in assignments}):
        for s1,s2 in itertools.combinations(cfg['final_seeds'] if stage=='final' else [0],2):
            stability.append(dict(k=k,d=d,seed_a=s1,seed_b=s2,ari=float(adjusted_rand_score(assignments[k,d,s1],assignments[k,d,s2]))))
    if stability: save_table(out/'cluster_stability.csv',stability)
    return [r for r in rows if r['method']=='kmeans']
