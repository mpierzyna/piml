import itertools
from typing import List

import joblib
import numpy as np
import sympy as sp

import piml
from piml.config.dim_vars import DimVarsConfig
from piml.pi.utils import invert_pi_target, make_set_obj, pi_sets_to_latex


def sign_valid(pi: sp.Expr, dim_vars: DimVarsConfig) -> bool:
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


def valid_pi_set(pi_set: List[sp.Expr], dim_vars: DimVarsConfig) -> bool:
    """ Pi set is valid if
    - it only contains one target group and
    - all pi groups have valid signs.
    """
    signs_valid = [sign_valid(pi, dim_vars=dim_vars) for pi in pi_set]
    return np.all(signs_valid) & contains_single_target(pi_set, dim_target=dim_vars.output.symbol)


if __name__ == '__main__':
    # Load workspace from environment variable `PIML_WORKSPACE` or first argument passed to script.
    ws = piml.Workspace.auto()

    # Load full list of pi sets
    pi_sets_full = joblib.load(ws.data_raw / "pi_sets_full.joblib")

    # Filter full list, get unique Pi targets, and also invert them
    dim_vars = ws.config.dim_vars
    pi_sets_full = [s for s in pi_sets_full if valid_pi_set(s, dim_vars=dim_vars)]
    unique_targets = {
        pi for pi in itertools.chain.from_iterable(pi_sets_full)
        if dim_vars.output.symbol in pi.free_symbols
    }
    unique_targets = sorted(unique_targets, key=str)  # sort by string expression to make target assignment reproducible
    unique_targets_inv = [invert_pi_target(pi, dim_output=dim_vars.output) for pi in unique_targets]
    unique_targets_names = [f"Pi_y_{i}" for i in range(len(unique_targets))]

    # Convert filtered sets to PiSet objects
    pi_sets: List[piml.PiSet] = [
        make_set_obj(
            s,
            set_id=i,
            targets=unique_targets,
            targets_inv=unique_targets_inv,
            targets_names=unique_targets_names,
        )
        for i, s in enumerate(pi_sets_full)
    ]
    joblib.dump(pi_sets, ws.data_extracted / "pi_sets_constrained.joblib")

    # Store sets as latex
    md_latex_file = ws.data_extracted / "pi_sets_constrained.md"
    md_latex_file.write_text(
        pi_sets_to_latex(pi_sets)
    )
