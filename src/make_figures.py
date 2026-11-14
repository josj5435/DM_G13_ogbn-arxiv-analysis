import argparse
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
from .common import ROOT,read_json,save_json,save_table,sha,load_data,load_config

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})

def save_figure(fig,path,footer='Source: OGB ogbn-arxiv | publication-year retrospective restriction'):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fig.text(.02,.012,footer,fontsize=7,color='#555555'); fig.tight_layout(rect=(0,.045,1,1))
    for extension in ['png','svg']: fig.savefig(path.with_suffix('.'+extension),dpi=300,facecolor='white')
    plt.close(fig)

def performance_figures(root):
    final=root/'final'; val=root/'validation'; dest=root/'figures'/'metrics'
    pr=pd.read_csv(final/'pagerank_metrics.csv')
    fig,ax=plt.subplots(figsize=(7,4)); ax.bar(['In-degree','PageRank'],pr.ndcg100,color=['#A7B2C4','#225A99'])
    ax.set(ylabel='NDCG@100 (linear gain)',title=f'2017 snapshot → 2018 citations | {pr.n_candidates.iloc[0]:,} candidates',ylim=(0,1))
    for i,v in enumerate(pr.ndcg100): ax.text(i,v+.015,f'{v:.4f}',ha='center')
    save_figure(fig,dest/'pagerank_vs_indegree',f'Source: OGB | Data: final/pagerank_metrics.csv | alpha={pr.alpha.dropna().iloc[0]}')
    kmv=pd.read_csv(val/'clustering_metrics.csv'); diag=pd.read_csv(val/'dimension_diagnostics.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for k,g in kmv.groupby('k'):
        g=g.sort_values('d'); axes[0].plot(g.d,g.silhouette,'o-',label=f'k={k}')
    axes[0].legend(); axes[0].set(xlabel='Dimensions (128 = no PCA)',ylabel='Silhouette in common normalized 128D',title='2017 validation | same 2,000 papers')
    axes[0].set_xticks([16,32,64,128])
    dg=diag[diag.d<128].sort_values('d'); axes[1].plot(dg.d,dg.explained_variance,'o-',color='#225A99'); axes[1].axhline(.9,ls='--',color='#C97D32'); axes[1].set(xlabel='PCA dimensions',ylabel='Cumulative explained variance',title='PCA fitted only on ≤2016 papers',ylim=(0,1.02))
    axes[1].set_xticks([16,32,64]); axes[1].text(17,.92,'90% eligibility threshold',fontsize=9,color='#99571E')
    save_figure(fig,dest/'validation_and_variance','Source: OGB | Data: validation/clustering_metrics.csv; dimension_diagnostics.csv')
    frozen=read_json(root/'frozen.json'); selection=frozen['selection']['clusters']; km=pd.read_csv(final/'clustering_metrics.csv'); actual=km[(km.method=='kmeans')&(km.k==selection['k'])]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['silhouette','ari']):
        g=actual.groupby('d')[metric].agg(['mean','std']).sort_index()
        ax.bar(g.index.astype(str),g['mean'],yerr=g['std'],capsize=4,color=['#72A8CA' if d!=selection['d'] else '#225A99' for d in g.index]); ax.set(xlabel='Dimensions (128 = baseline)',ylabel=metric.upper() if metric=='ari' else 'Common-space Silhouette',title=f'2018 assignment | k={selection["k"]} | 5 initializations')
    save_figure(fig,dest/'final_clustering',f'Source: OGB | n_eval={actual.n_eval.iloc[0]:,}; silhouette n=2,000 | Bars: mean ± initialization SD | final/clustering_metrics.csv')
    chosen=km[(km.k==selection['k'])&(km.d==selection['d'])]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,metric in zip(axes,['silhouette','ari']):
        values=chosen.groupby('method')[metric].mean().reindex(['kmeans','B4_random'])
        ax.bar(['k-means','Size-preserving random'],values,color=['#225A99','#A7B2C4']); ax.axhline(0,color='#666666',lw=.7)
        ax.set(ylabel='Common-space Silhouette' if metric=='silhouette' else 'ARI against subject labels',title=f'2018 | frozen k={selection["k"]}, d={selection["d"]}')
    save_figure(fig,dest/'clustering_random_baseline','Source: OGB | 5 model seeds; 5 size-preserving shuffles per model | final/clustering_metrics.csv')
    ap=pd.read_csv(final/'apriori_metrics.csv'); fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    population=int(pd.read_csv(ROOT/'outputs'/'tables'/'preprocessing_stats.csv')['2018_queries_total'].iloc[0])
    eligible=int(ap.n_queries.iloc[0]); inclusion=eligible/population
    methods=['B2','A2','A3']; names=['Popularity','Apriori','Apriori + fill']
    g=ap.groupby('method').recall10.agg(['mean','std']).reindex(methods)
    axes[0].bar(names,g['mean'],yerr=g['std'],capsize=4,color=['#A7B2C4','#225A99','#4C9679'])
    axes[0].set(ylabel='Recall@10 (eligible queries only)',title=f'2018 eligibility: {eligible:,} / {population:,} ({inclusion:.1%})')
    g=ap[ap.method.isin(['A2','A3'])].groupby('method').rule_coverage.agg(['mean','std']).reindex(['A2','A3'])
    axes[1].bar(['Apriori','Apriori + fill'],g['mean'],yerr=g['std'],capsize=4,color=['#225A99','#4C9679'])
    axes[1].set(ylabel='Queries with ≥1 rule candidate / eligible queries',title='Rule availability is not a hit rate',ylim=(0,1))
    empty=ap[ap.method=='A2']['empty'].mean()
    save_figure(fig,dest/'apriori_vs_popularity',f'Source: OGB | Internal past references ≥5; 5 masks; mean ± masking SD | Pure Apriori empty lists: {empty:.1%}\nB2 RuleCoverage: N/A (no rules used). A3 coverage excludes popularity fill | final/apriori_metrics.csv')
    pats=pd.read_csv(final/'cluster_citation_patterns.csv'); pats=pats[(pats.k==selection['k'])&(pats.d==selection['d'])]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,kind,metric,ylabel in [(axes[0],'bibliographic','jaccard_difference','IPW mean Jaccard: same − different'),(axes[1],'direct_citation','same_fraction','Same-cluster directed citation fraction')]:
        g=pats[pats.kind==kind]; observed=g[g.baseline=='observed'][metric]; random=g[g.baseline!='observed'][metric]
        ax.bar(['Observed','Random baseline'],[observed.mean(),random.mean()],color=['#225A99','#A7B2C4']); ax.set(ylabel=ylabel,title=kind.replace('_',' ').title()+' | 2018')
    save_figure(fig,dest/'citation_pattern_baselines',f'Source: OGB | fixed 100,000 pairs; k={selection["k"]}, d={selection["d"]} | final/cluster_citation_patterns.csv\nIPW mean = mean of ordinary set Jaccard weighted by inverse pair inclusion probability; not weighted-item Jaccard.')
    data=load_data(); deg=pd.read_csv(ROOT/'outputs'/'tables'/'degrees_2017.csv'); stats=pd.read_csv(ROOT/'outputs'/'tables'/'preprocessing_stats.csv').iloc[0]
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    years,counts=np.unique(data['years'],return_counts=True); axes[0,0].bar(years,counts,color='#225A99'); axes[0,0].set(xlabel='Publication year',ylabel='Papers',title='All 169,343 papers')
    for name in ['in_degree','out_degree']: axes[0,1].hist(deg[name],bins=np.geomspace(1,deg[name].max()+1,45),histtype='step',label=name)
    axes[0,1].set(xscale='log',yscale='log',xlabel='Degree (zero reported in table)',ylabel='Papers',title='2017 restricted graph'); axes[0,1].legend()
    labels,counts=np.unique(data['labels'],return_counts=True); axes[1,0].bar(labels,counts,color='#4C9679'); axes[1,0].set(xlabel='Subject label ID',ylabel='Papers',title='Subject imbalance (all papers)')
    axes[1,1].bar(['Original edges','Removed: year reversal','Clean edges'],[stats.raw_edges,stats.reversed_year_edges,stats.clean_edges],color=['#A7B2C4','#C97D32','#225A99']); axes[1,1].tick_params(axis='x',labelrotation=12); axes[1,1].set(ylabel='Directed citations',title='Cleaning counts; no duplicates/self-loops')
    save_figure(fig,root/'figures'/'eda'/'dataset_overview','Source: OGB | Data: outputs/tables/preprocessing_stats.csv; degrees_2017.csv; data/processed')

def select_graph_targets(nodes,edges,clusters,cfg):
    rng=np.random.default_rng(cfg['selection_seed']); ids,sizes=np.unique(clusters[nodes],return_counts=True)
    eligible=ids[(ids>=0)&(sizes>=cfg['min_cluster_size'])]; linked=np.unique(edges)
    c=sorted(rng.choice(np.sort(eligible),min(cfg['cluster_count'],len(eligible)),replace=False).tolist())
    p=sorted(rng.choice(np.sort(linked),min(cfg['paper_count'],len(linked)),replace=False).tolist())
    return dict(clusters=c,papers=p,eligible_clusters=len(eligible),eligible_papers=len(linked),isolated_excluded=len(nodes)-len(linked),selection_seed=cfg['selection_seed'],missing_clusters=cfg['cluster_count']-len(c),missing_papers=cfg['paper_count']-len(p))

def extract_display_subgraph(g,clusters,mode,target,cfg):
    rng=np.random.default_rng(cfg['subgraph_seed']); limit=cfg['max_nodes']; anchor=None; required=set()
    if mode=='cluster':
        population=sorted(n for n in g if clusters[n]==target); sub=g.subgraph(population)
        if len(population)<=limit: chosen=population
        elif sub.number_of_edges()==0: chosen=sorted(rng.choice(population,limit,replace=False).tolist())
        else:
            linked=sorted({n for e in sub.edges for n in e}); anchor=int(rng.choice(linked)); chosen=[anchor]; seen={anchor}; queue=deque([(anchor,0)])
            while queue and len(chosen)<limit:
                n,depth=queue.popleft()
                if depth>=cfg['cluster_hops']: continue
                neighbors=np.array(sorted(set(sub.predecessors(n))|set(sub.successors(n))))
                rng.shuffle(neighbors)
                for other in neighbors:
                    other=int(other)
                    if other in seen: continue
                    seen.add(other); chosen.append(other); queue.append((other,depth+1))
                    if sub.has_edge(n,other): required.add((n,other))
                    elif sub.has_edge(other,n): required.add((other,n))
                    if len(chosen)==limit: break
    else:
        population=sorted(set(g.predecessors(target))|set(g.successors(target))|{target}); neighbors=sorted(set(population)-{target})
        if len(neighbors)>limit-1: neighbors=sorted(rng.choice(neighbors,limit-1,replace=False).tolist())
        chosen=sorted([target]+neighbors)
        required={(a,b) for a,b in g.subgraph(chosen).edges if a==target or b==target}
    chosen=sorted(chosen); candidates=sorted(g.subgraph(chosen).edges); edges=candidates
    if len(required)>cfg['max_edges']: raise ValueError('Edge limit cannot preserve required connections')
    if len(edges)>cfg['max_edges']:
        remainder=sorted(set(candidates)-required); indices=rng.choice(len(remainder),cfg['max_edges']-len(required),replace=False)
        edges=sorted(required|{remainder[i] for i in indices})
    meta=dict(mode=mode,target=target,anchor=anchor,population_nodes=len(population),display_nodes=len(chosen),candidate_edges=len(candidates),display_edges=len(edges),omitted_nodes=len(population)-len(chosen),omitted_edges=len(candidates)-len(edges),required_edges=len(required),is_partial=len(chosen)<len(population) or len(edges)<len(candidates),no_internal_citation=len(candidates)==0)
    if mode=='paper': meta.update(incoming_display=sum(g.has_edge(n,target) for n in chosen),outgoing_display=sum(g.has_edge(target,n) for n in chosen))
    assert len(chosen)<=limit and all(g.has_edge(a,b) for a,b in edges)
    return chosen,edges,meta

def network_figures(root,cfg):
    final=root/'final'; frozen=read_json(root/'frozen.json'); selection=frozen['selection']['clusters']; tag=f'k{selection["k"]}_d{selection["d"]}_s0'
    archive=np.load(ROOT/'outputs'/'snapshots'/'2017.npz'); nodes=archive['nodes']; edges=archive['edges']
    clusters=np.load(final/'models'/f'{tag}_assignments.npz')['assignment']; scores=pd.read_parquet(final/'pagerank_scores.parquet').set_index('node_id').pagerank
    q99=float(np.quantile(scores,.99)); data=load_data(); g=nx.DiGraph(); g.add_nodes_from(nodes.tolist()); g.add_edges_from(edges.tolist())
    selected=select_graph_targets(nodes,edges,clusters,cfg); selected.update(cutoff=2017,model=tag,config=cfg,run_id=root.name)
    dest=root/'visualization'; save_json(dest/'selection.json',selected)
    titles={}
    if (root/'diagnostics'/'titles.parquet').exists(): titles=pd.read_parquet(root/'diagnostics'/'titles.parquet').set_index('node_id').title.to_dict()
    for mode,targets in [('cluster',selected['clusters']),('paper',selected['papers'])]:
        for target in targets:
            chosen,ed,meta=extract_display_subgraph(g,clusters,mode,target,cfg); view=f'{mode}_{target}'; folder=dest/view
            graph=nx.DiGraph(); graph.add_nodes_from(chosen); graph.add_edges_from(ed)
            pos=nx.spring_layout(graph.to_undirected(),seed=cfg['layout_seed'],iterations=cfg['layout_iterations'])
            node_rows=[dict(original_node_id=n,cluster_id=int(clusters[n]),pagerank=float(scores[n]),year=int(data['years'][n]),label=int(data['labels'][n]),title=titles.get(n,''),is_center=mode=='paper' and n==target,is_anchor=n==meta['anchor']) for n in chosen]
            save_table(folder/'nodes.csv',node_rows); save_table(folder/'edges.csv',pd.DataFrame(ed,columns=['source','target'])); save_table(folder/'positions.csv',[dict(original_node_id=n,x=pos[n][0],y=pos[n][1]) for n in chosen])
            colors=[plt.get_cmap('tab20')(int(clusters[n])%20) if clusters[n]>=0 else '#AAAAAA' for n in chosen]
            size=[30+270*min(scores[n]/q99,1) for n in chosen]; emphasized=target if mode=='paper' else meta['anchor']
            fig,ax=plt.subplots(figsize=(10,8)); nx.draw_networkx_edges(graph,pos,ax=ax,arrows=True,arrowsize=9,width=.65,alpha=.4,edge_color='#485366',connectionstyle='arc3,rad=0.05')
            nx.draw_networkx_nodes(graph,pos,ax=ax,node_color=colors,node_size=size,edgecolors=['#101820' if n==emphasized else 'white' for n in chosen],linewidths=[2.7 if n==emphasized else .5 for n in chosen])
            top=sorted(chosen,key=lambda n:(-scores[n],n))[:cfg['label_top_k']]
            if emphasized is not None: top=list(set(top)|{emphasized})
            label_pos={n:(pos[n][0],pos[n][1]+.03) for n in top}
            nx.draw_networkx_labels(graph,label_pos,labels={n:str(n) for n in top},ax=ax,font_size=7,bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=.15))
            cluster_ids=sorted(set(int(clusters[n]) for n in chosen)); handles=[Line2D([0],[0],marker='o',color='white',markerfacecolor=plt.get_cmap('tab20')(c%20) if c>=0 else '#AAAAAA',label=f'Cluster {c}',markersize=6) for c in cluster_ids]
            ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(1.0,1),fontsize=7,frameon=False,ncol=1 if len(handles)<14 else 2)
            status='PARTIAL VIEW' if meta['is_partial'] else 'full local view'
            ax.set_title(f'{mode.title()} {target} | 2017 snapshot | {len(chosen)} nodes / {len(ed)} edges\n{status} | {root.name} | {tag}',fontsize=12)
            ax.set_axis_off()
            footer='Arrows: citing → cited | Area: full-graph PageRank, clipped at q99 | Color: content cluster (colors may repeat)\n'+f'Population nodes={meta["population_nodes"]:,}; omitted nodes={meta["omitted_nodes"]:,}, edges={meta["omitted_edges"]:,} | Source: OGB | visualization/{view}'
            if not ed: footer+=' | NO CITATIONS IN DISPLAY SAMPLE'
            save_figure(fig,root/'figures'/'networks'/view,footer)
            meta.update(view_id=view,cutoff=2017,run_id=root.name,graph_hash=sha(ROOT/'outputs'/'snapshots'/'2017.npz'),cluster_hash=sha(final/'models'/f'{tag}_assignments.npz'),config_hash=sha(root/'frozen.json'),seeds={k:cfg[k] for k in ['selection_seed','subgraph_seed','layout_seed']},layout_iterations=cfg['layout_iterations'],pagerank_q99=q99,unassigned_nodes=sum(clusters[n]<0 for n in chosen),networkx_version=nx.__version__,matplotlib_version=matplotlib.__version__)
            save_json(folder/'metadata.json',meta)
    print('Saved metric, EDA and 8 network views (PNG + SVG)',flush=True)

if __name__=='__main__':
    from .common import REFERENCE_RUN
    p=argparse.ArgumentParser(); p.add_argument('--run-id',default=REFERENCE_RUN)
    p.add_argument('--mode',choices=['metrics','networks','both'],default='both'); args=p.parse_args()
    root=ROOT/'outputs'/args.run_id; frozen=read_json(root/'frozen.json')
    if args.mode in ['metrics','both']: performance_figures(root)
    if args.mode in ['networks','both']: network_figures(root,frozen['config']['visualization'])
