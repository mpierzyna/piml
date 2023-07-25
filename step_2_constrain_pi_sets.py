import itertools
from typing import List

import joblib
import numpy as np
import sympy as sp

import piml
import piml.config
import piml.pi


def sign_valid(pi: sp.Expr, dim_vars: piml.config.DimVars) -> bool:
    """
    Returns False if provided Pi group has an argument which is a **signed** symbol
    raised to an **even** or **non-integer** power.

    E.g. sqrt(shfx) would cause NaN in unstable conditions
    and shfx**2 would lose information. Therefore, we avoid these!
    """
    for arg in pi.args:
        if arg.is_Atom:
            continue
        elif arg.is_Pow:
            # Allow all power expressions with uneven integer exponents because that keeps original sign
            arg: sp.Pow
            if arg.exp.is_Integer and not arg.exp.is_even:
                continue

            # Only allow even or non-integer exponents if base is unsigned
            if dim_vars[arg.base].signed:
                print(f"Eliminating {pi} because of {arg}.")
                return False
        else:
            raise ValueError(f"Unexpected argument {type(arg)} in Pi group pi")

    return True


def contains_single_target(pi_set: List[sp.Expr], dim_target: sp.Expr) -> bool:
    """ True if dimless set contains only one term which is function of target """
    y_count = [1 for pi in pi_set if dim_target in pi.free_symbols]
    return sum(y_count) == 1


def valid_pi_set(pi_set: List[sp.Expr], dim_vars: piml.config.DimVars) -> bool:
    """ Pi set is valid if
    - it only contains one target group and
    - all pi groups have valid signs.
    """
    signs_valid = [sign_valid(pi, dim_vars=dim_vars) for pi in pi_set]
    return np.all(signs_valid) & contains_single_target(pi_set, dim_target=dim_vars.dim_output.symbol)


def invert_pi_target(pi_expr: sp.Expr, dim_output: piml.pi.DimSymbol) -> sp.Expr:
    """ Invert provided non-dim target expression so that it can be used to recover dimensional target """
    pi_inv = sp.solve(pi_expr - piml.pi.PI_Y_expr, dim_output.symbol)

    if len(pi_inv) > 1:
        raise ValueError(f"Inversion of {pi_expr} resulted in more than one solution. Abort!")

    return pi_inv[0]


def make_set_obj(pi_set: List[sp.Expr], set_id: str,
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


if __name__ == '__main__':
    # Load workspace from environment variable `PIML_WORKSPACE` or first argument passed to script.
    ws = piml.Workspace.auto()

    # Load full list of pi sets
    pi_sets_full = joblib.load(ws.data_raw / "pi_sets_full.joblib")

    # Filter full list, get unique Pi targets, and also invert them
    pi_sets_full = [s for s in pi_sets_full if valid_pi_set(s, dim_vars=ws.dim_vars)]
    unique_targets = {
        pi for pi in itertools.chain.from_iterable(pi_sets_full)
        if ws.dim_vars.dim_output.symbol in pi.free_symbols
    }
    unique_targets = sorted(unique_targets, key=str)  # sort by string expression to make target assignment reproducible
    unique_targets_inv = [invert_pi_target(pi, dim_output=ws.dim_vars.dim_output) for pi in unique_targets]
    unique_targets_names = [f"Pi_y_{i}" for i in range(len(unique_targets))]

    # Convert filtered sets to PiSet objects
    pi_sets: List[piml.PiSet] = [
        make_set_obj(
            s,
            set_id=f"{i:02d}",
            targets=unique_targets,
            targets_inv=unique_targets_inv,
            targets_names=unique_targets_names,
            # dim_vars=ws.dim_vars
        )
        for i, s in enumerate(pi_sets_full)
    ]
    joblib.dump(pi_sets, ws.data_extracted / "pi_sets_constrained.joblib")

    # Store sets as latex
    md_latex_file = ws.data_extracted / "pi_sets_constrained.md"
    md_latex_file.write_text(
        pi_sets_to_latex(pi_sets)
    )
