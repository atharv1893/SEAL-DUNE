#!/usr/bin/env python3
"""SEAL edge-weight regression with Directional features for Stanford SNAP Bitcoin Alpha.

Canonical entrypoint for SEAL + Direction without Fairness & Goodness (FG).
Re-exports the implementation from seal_bitcoin.
"""
from __future__ import annotations

# Re-export the shared implementation.
from Bitcon_OTC_code.Python.seal_bitcoin import (
    DUNE,
    GNNGraph,
    SEALWeightRegressor,
    baseline_predictions,
    build_crossfit_train_graphs,
    build_graph_dataset,
    build_structural_graph,
    cmd_args,
    edge_direction_features,
    edge_pair_features,
    load_bitcoin_alpha,
    main,
    make_graph,
    node_direction_stats,
    regression_metrics,
    run_epoch,
    set_seed,
    subgraph_extraction_labeling,
)

if __name__ == "__main__":
    main()
