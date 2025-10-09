import mpbn
import bonesis
import sys
import os

def bonesis_ensemble_from_single_bn(bn, limit=1000,
                    exact_influence_graph=True,
                    fixedpoints=None,
                    extra_properties=lambda bo: bo,
                    diversity=True,
                    maxclause=32,
                    **domain_opts):
    """
    bn: a colomoto.minibn.BooleanNetwork object or bnet filename
    limit: maximum size of the ensemble to generate
    exact_influence_graph: if True, all returned BNs have the exact same influence graph as `bn`. Otherwise, it can be a subgraph of it.
    fixedpoints:
        - None: no constraint
        - "included": all BNs include at least the fixed points of `bn`
        - "same": all BNs have exactly the same fixed points as `bn` (recommended to set `exact_influence_graph=False`)
    extra_properties: function that takes as input a BoNesis object and completes it with additional constraints.
    diversity: use diverse ensemble generation from bonesis
    """

    bn = mpbn.MPBooleanNetwork.auto_cast(bn)
    dom = bonesis.InfluenceGraph(bn.influence_graph(),
                                 exact=exact_influence_graph,
                                 maxclause=maxclause, **domain_opts)

    data = {}
    if fixedpoints:
        for i, x in enumerate(bn.fixedpoints()):
            data[f"fp{i}"] = x

    bo = bonesis.BoNesis(dom, data)

    if fixedpoints in ["included", "same"]:
        for fp in data:
            bo.fixed(~bo.obs(fp))

    if fixedpoints == "same":
        bo.all_fixpoints({bo.obs(fp) for fp in data})

    extra_properties(bo)

    if diversity:
        view = bo.diverse_boolean_networks(limit=limit)
    else:
        view = bo.boolean_networks(limit=limit)

    return list(view)
