import piml
import joblib

from buckinghampy import BuckinghamPi

if __name__ == "__main__":
    # Load workspace from environment variable `PIML_WORKSPACE` or first argument passed to script.
    ws = piml.Workspace.auto()

    # Optical turbulence in the surface layer
    ot_sl = BuckinghamPi(n_jobs=-1)

    # Add all input variables to BuckinghamPi generator
    for s in ws.config.dim_vars.inputs:
        ot_sl.add_variable(name=str(s.symbol), dimensions=s.dimensions)

    # Also add output
    ot_sl.add_variable(
        name=str(ws.config.dim_vars.output.symbol),
        dimensions=ws.config.dim_vars.output.dimensions
    )

    # Generate pi terms and store them
    print("Generating Pi terms... This may take a while.")
    ot_sl.generate_pi_terms()

    # Save pi terms
    pi_output = ws.data_raw / "pi_sets_full.joblib"
    joblib.dump(ot_sl.pi_terms, pi_output)
    print(f"Done! Pi terms saved to {pi_output}.")
