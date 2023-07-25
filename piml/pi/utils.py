from typing import List

import sympy as sp

import piml
import piml.pi


def invert_pi_target(pi_expr: sp.Expr, dim_output: piml.pi.DimSymbol) -> sp.Expr:
    """ Invert provided non-dim target expression so that it can be used to recover dimensional target """
    pi_inv = sp.solve(pi_expr - piml.pi.PI_Y_expr, dim_output.symbol)

    if len(pi_inv) > 1:
        raise ValueError(f"Inversion of {pi_expr} resulted in more than one solution. Abort!")

    return pi_inv[0]


def make_set_obj(pi_set: List[sp.Expr], set_id: int,
                 targets: List[sp.Expr], targets_inv: List[sp.Expr], targets_names: List[str]) -> piml.PiSet:
    """ Split set into features and target """
    feature_exprs = []
    target_expr = None
    target_inv_expr = None
    target_id = None

    for pi in pi_set:
        try:
            target_idx = targets.index(pi)
            target_expr = pi
            target_inv_expr = targets_inv[target_idx]
            target_id = targets_names[target_idx]
        except ValueError:
            feature_exprs.append(pi)

    # Sanity check that no pi group was lost from set
    assert len(feature_exprs) + 1 == len(pi_set)

    # Sort pi groups by number of free symbols
    feature_exprs = sorted(feature_exprs, key=lambda pi: len(pi.free_symbols))

    # Create actual PiSet object
    return piml.PiSet(
        id=set_id,
        feature_exprs=feature_exprs,
        target_id=target_id,
        target_expr=target_expr,
        target_inv_expr=target_inv_expr
    )


def pi_sets_to_latex(pi_sets: List[piml.PiSet]) -> str:
    """ Return Markdown style document listing sets and their variables as latex expressions """
    latex = ""

    for s in pi_sets:
        latex += f"# Set {s.id}\n"
        for i, pi in enumerate(s.feature_exprs):
            latex += "- $" + r"\pi_{" + f"{i:d}" + r"} = " + sp.latex(pi) + "$\n"
        latex += r"- $\pi_y = " + sp.latex(s.target_expr) + f"$ ({s.target_id}) \n"
        latex += "\n"

    return latex
