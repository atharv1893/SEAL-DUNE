from __future__ import print_function
import numpy as np
import random
from tqdm import tqdm
import os, sys, pdb, math, time
import networkx as nx
import argparse
import scipy.io as sio
import scipy.sparse as ssp
from sklearn import metrics
from gensim.models import Word2Vec
import warnings
warnings.simplefilter('ignore', ssp.SparseEfficiencyWarning)
cur_dir = os.path.dirname(os.path.realpath(__file__))

sys.path.append('%s/..' % cur_dir)
sys.path.append('%s/../Python/software/node2vec/src' % cur_dir)
sys.path.append('%s/../pytorch_DGCNN' % cur_dir)

from util import GNNGraph
import node2vec
import multiprocessing as mp
from itertools import islice

"""Drop-in replacements for the corresponding functions in the pasted SEAL utility.

These functions expect the same imports already present in the original utility:
math, random, time, numpy as np, scipy.sparse as ssp, networkx as nx,
from tqdm import tqdm, from sklearn import metrics, and multiprocessing as mp.
"""


def sample_neg(net, test_ratio=0.1, train_pos=None, test_pos=None,
               max_train_num=None, all_unknown_as_negative=False):
    """Sample unique undirected non-edges from the upper triangle.

    In all_unknown_as_negative mode, train negatives are deliberately a subset
    of the full unknown-edge test candidate set, matching the original option's
    stated semantics. In the default mode, train/test negatives are disjoint.
    """
    net_triu = ssp.triu(net, k=1).tocsr()
    row, col = ssp.find(net_triu)[:2]

    if train_pos is None and test_pos is None:
        perm = np.random.permutation(len(row))
        row, col = row[perm], col[perm]
        split = int(math.ceil(len(row) * (1 - test_ratio)))
        train_pos = (row[:split], col[:split])
        test_pos = (row[split:], col[split:])

    if max_train_num is not None and train_pos is not None:
        perm = np.random.permutation(len(train_pos[0]))[:max_train_num]
        train_pos = (np.asarray(train_pos[0])[perm], np.asarray(train_pos[1])[perm])

    train_num = len(train_pos[0]) if train_pos is not None else 0
    test_num = len(test_pos[0]) if test_pos is not None else 0
    n = net.shape[0]
    total_pairs = n * (n - 1) // 2
    unknown_count = total_pairs - net_triu.nnz

    if all_unknown_as_negative:
        needed = train_num
    else:
        needed = train_num + test_num
    if needed > unknown_count:
        raise ValueError(
            f"Requested {needed} unique negative pairs, but only "
            f"{unknown_count} unknown upper-triangular pairs exist."
        )

    print('sampling negative links for train and test')
    selected = set()
    # Rejection sampling avoids materializing the O(n^2) complement of a sparse graph.
    while len(selected) < needed:
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i < j and net[i, j] == 0 and (i, j) not in selected:
            selected.add((i, j))

    selected = list(selected)
    neg_i = [p[0] for p in selected]
    neg_j = [p[1] for p in selected]
    train_neg = (neg_i[:train_num], neg_j[:train_num])

    if all_unknown_as_negative:
        test_neg_i, test_neg_j = ssp.find(ssp.triu(net == 0, k=1))[:2]
        test_neg = (test_neg_i.tolist(), test_neg_j.tolist())
    else:
        test_neg = (neg_i[train_num:train_num + test_num],
                    neg_j[train_num:train_num + test_num])

    return train_pos, train_neg, test_pos, test_neg


def links2subgraphs(A, train_pos, train_neg, test_pos, test_neg, h=1,
                    max_nodes_per_hop=None, node_information=None,
                    no_parallel=False, directed=False):
    """Extract subgraphs; explicitly propagate directed mode to every worker."""
    if h == 'auto':
        if directed:
            raise ValueError(
                "Automatic h selection uses undirected CN/AA heuristics. Choose "
                "h=1 or h=2 explicitly for a directed task."
            )
        _, _, val_test_pos, val_test_neg = sample_neg(A, 0.1)
        val_A = A.copy()
        val_A[val_test_pos[0], val_test_pos[1]] = 0
        val_A[val_test_pos[1], val_test_pos[0]] = 0
        val_auc_CN = CN(val_A, val_test_pos, val_test_neg)
        val_auc_AA = AA(val_A, val_test_pos, val_test_neg)
        print('Validation AUC of AA is {}, CN is {}'.format(val_auc_AA, val_auc_CN))
        h = 2 if val_auc_AA >= val_auc_CN else 1
        print('Choose h={}'.format(h))

    max_n_label = {'value': 0}

    def helper(links, g_label):
        if links is None or len(links[0]) == 0:
            return []
        args = [((i, j), A, h, max_nodes_per_hop, node_information,
                 False, directed)
                for i, j in zip(links[0], links[1])]
        if no_parallel:
            results = [parallel_worker(x) for x in tqdm(args)]
        else:
            start = time.time()
            with mp.Pool(mp.cpu_count()) as pool:
                results = list(tqdm(pool.imap(parallel_worker, args), total=len(args)))
            print(f"Time elapsed for subgraph extraction: {time.time() - start:.2f}s")

        graphs = []
        for g, labels, features in results:
            if len(labels):
                max_n_label['value'] = max(max_n_label['value'], max(labels))
            graphs.append(GNNGraph(g, g_label, labels, features))
        return graphs

    print('Enclosing subgraph extraction begins...')
    train_graphs = helper(train_pos, 1) + helper(train_neg, 0)
    test_graphs = helper(test_pos, 1) + helper(test_neg, 0)
    return train_graphs, test_graphs, max_n_label['value']


def parallel_worker(x):
    return subgraph_extraction_labeling(*x)


def subgraph_extraction_labeling(ind, A, h=1, max_nodes_per_hop=None,
                                 node_information=None, return_nodes=False,
                                 directed=False):
    """Extract an enclosing subgraph and optionally use directed distance labels."""
    nodes = {ind[0], ind[1]}
    visited = {ind[0], ind[1]}
    fringe = {ind[0], ind[1]}

    for _ in range(1, h + 1):
        fringe = neighbors(fringe, A, directed=directed) - visited
        visited.update(fringe)
        if max_nodes_per_hop is not None and max_nodes_per_hop < len(fringe):
            fringe = set(random.sample(sorted(fringe), max_nodes_per_hop))
        if not fringe:
            break
        nodes.update(fringe)

    # Keep query endpoints at local indices 0 and 1; sort all other nodes for
    # deterministic local indexing across repeated runs.
    nodes.discard(ind[0])
    nodes.discard(ind[1])
    nodes = [ind[0], ind[1]] + sorted(nodes)
    subgraph = A[nodes, :][:, nodes].tocsr(copy=True)

    if directed:
        # Mask u->v only; v->u remains available if observed in the context.
        subgraph = subgraph.tolil(copy=True)
        subgraph[0, 1] = 0
        subgraph = subgraph.tocsr()
        subgraph.eliminate_zeros()
    else:
        subgraph = subgraph.tolil(copy=True)
        subgraph[0, 1] = 0
        subgraph[1, 0] = 0
        subgraph = subgraph.tocsr()
        subgraph.eliminate_zeros()

    labels = node_label(subgraph, directed=directed)
    features = None if node_information is None else node_information[nodes]
    if directed:
        # Compatibility across NetworkX releases.
        if hasattr(nx, 'from_scipy_sparse_array'):
            g = nx.from_scipy_sparse_array(subgraph, create_using=nx.DiGraph())
        else:
            g = nx.from_scipy_sparse_matrix(subgraph, create_using=nx.DiGraph())
    else:
        if hasattr(nx, 'from_scipy_sparse_array'):
            g = nx.from_scipy_sparse_array(subgraph)
        else:
            g = nx.from_scipy_sparse_matrix(subgraph)

    if return_nodes:
        return g, labels.tolist(), features, nodes
    return g, labels.tolist(), features


def neighbors(fringe, A, directed=False):
    """Return incoming neighbours, plus outgoing neighbours in directed mode."""
    res = set()
    for node in fringe:
        incoming, _, _ = ssp.find(A[:, node])
        res.update(incoming.tolist())
        if directed:
            _, outgoing, _ = ssp.find(A[node, :])
            res.update(outgoing.tolist())
    return res
