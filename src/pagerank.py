import numpy as np
from scipy.sparse import csr_matrix

def compute_pagerank(nodes,edges,alpha=.85,tol=1e-8,max_iter=1000):
    n=len(nodes)
    if n==0: raise ValueError('Empty snapshot')
    src=np.searchsorted(nodes,edges[:,0]); dst=np.searchsorted(nodes,edges[:,1])
    assert np.array_equal(nodes[src],edges[:,0]) and np.array_equal(nodes[dst],edges[:,1])
    out=np.bincount(src,minlength=n); dangling=out==0
    pt=csr_matrix((1/out[src],(dst,src)),shape=(n,n))
    def step(p): return alpha*(pt@p+p[dangling].sum()/n)+(1-alpha)/n
    p=np.full(n,1/n)
    for iteration in range(1,max_iter+1):
        new=step(p); delta=float(np.abs(new-p).sum()); p=new
        if delta<tol: break
    residual=float(np.abs(p-step(p)).sum())
    meta=dict(iterations=iteration,iteration_delta=delta,fixed_point_residual=residual,converged=delta<tol,score_sum=float(p.sum()),minimum=float(p.min()))
    if not meta['converged']: raise RuntimeError(f'PageRank failed to converge: {meta}')
    assert np.isfinite(p).all() and (p>=0).all() and abs(p.sum()-1)<1e-10
    return p,meta
