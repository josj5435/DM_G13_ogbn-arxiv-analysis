import networkx as nx
import numpy as np
import pytest
from src.make_figures import extract_display_subgraph,select_graph_targets

CFG=dict(selection_seed=42,subgraph_seed=43,layout_seed=44,cluster_count=3,paper_count=5,min_cluster_size=10,max_nodes=80,max_edges=250,cluster_hops=2)

def test_paper_caps_required_edges_and_directions():
    g=nx.DiGraph(); g.add_nodes_from(range(150))
    g.add_edges_from((0,i) for i in range(1,150))
    g.add_edges_from((i,j) for i in range(1,150) for j in range(i+1,min(i+8,150)))
    chosen,edges,meta=extract_display_subgraph(g,np.zeros(150,int),'paper',0,CFG)
    assert len(chosen)==80 and len(edges)==250 and 0 in chosen
    assert all((0,i) in edges for i in chosen if i!=0)
    assert all(g.has_edge(a,b) for a,b in edges) and meta['omitted_nodes']==70
    assert (chosen,edges,meta)==extract_display_subgraph(g,np.zeros(150,int),'paper',0,CFG)

def test_cluster_hops_and_unlinked_case():
    g=nx.path_graph(200,create_using=nx.DiGraph); c=np.zeros(200,int)
    nodes,edges,meta=extract_display_subgraph(g,c,'cluster',0,CFG)
    assert len(nodes)<=5 and meta['anchor'] in nodes
    assert nx.is_connected(g.subgraph(nodes).to_undirected())
    empty=nx.empty_graph(100,create_using=nx.DiGraph)
    nodes,edges,meta=extract_display_subgraph(empty,np.zeros(100,int),'cluster',0,CFG)
    assert len(nodes)==80 and edges==[] and meta['anchor'] is None

def test_selection_eligible_and_invalid_cap():
    nodes=np.arange(100); edges=np.array([(0,i) for i in range(1,100)])
    c=nodes//20; a=select_graph_targets(nodes,edges,c,CFG); b=select_graph_targets(nodes,edges,c,CFG)
    assert a==b and len(a['clusters'])==3 and len(a['papers'])==5
    g=nx.DiGraph(); g.add_edges_from(edges)
    with pytest.raises(ValueError): extract_display_subgraph(g,c,'paper',0,dict(CFG,max_edges=2))
