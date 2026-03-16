import gzip
import importlib.util
from pathlib import Path


def _load_generate_instances():
    path = Path("/Users/lafifi/Codes/research/bb-ml/benchmarks/pipeline/generate_instances.py")
    spec = importlib.util.spec_from_file_location("bbml_generate_instances", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_ca_generator_matches_learn2branch_structure():
    module = _load_generate_instances()
    rng = module.np.random.default_rng(0)
    obj, constrs, _, binary = module.gen_ca(rng, n_items=100, n_bids=500)

    assert len(binary) == 500
    assert len(constrs) > 100
    assert all(name.startswith("item") for name, *_ in constrs)
    assert max(len(lhs) for _, lhs, _, _ in constrs) > 10


def test_sc_generator_matches_balasho_shape():
    module = _load_generate_instances()
    rng = module.np.random.default_rng(0)
    obj, constrs, _, binary = module.gen_sc(rng, n_rows=500, n_cols=1000, density=0.05)

    assert len(binary) == 1000
    assert len(constrs) == 500
    assert all(sense == ">=" and rhs == 1.0 for _, _, sense, rhs in constrs)
    assert min(len(lhs) for _, lhs, _, _ in constrs) >= 1


def test_cfl_generator_has_tightening_constraints():
    module = _load_generate_instances()
    rng = module.np.random.default_rng(0)
    obj, constrs, bounds, binary = module.gen_cfl(rng, n_cust=20, n_fac=10, ratio=5.0)

    assert len(binary) == 10
    assert bounds is not None
    assert "total_capacity" in {name for name, *_ in constrs}
    assert any(name.startswith("affectation_") for name, *_ in constrs)
    assert any(name.startswith("capacity_") for name, *_ in constrs)
    assert any(name.startswith("demand_") for name, *_ in constrs)


def test_mis_generator_uses_clique_inequalities():
    module = _load_generate_instances()
    rng = module.np.random.default_rng(0)
    obj, constrs, _, binary = module.gen_mis(rng, n_nodes=100, m=4)

    assert len(binary) == 100
    assert any(len(lhs) > 2 for _, lhs, _, _ in constrs)
    assert all(sense == "<=" and rhs == 1.0 for _, _, sense, rhs in constrs)


def test_writer_supports_gzipped_lp_output(tmp_path):
    module = _load_generate_instances()
    out_path = tmp_path / "sc_00000.lp.gz"

    module.write_sc(out_path, seed=0, n_rows=20, n_cols=40, density=0.1)

    assert out_path.is_file()
    with gzip.open(out_path, "rt", encoding="utf-8") as fh:
        text = fh.read()
    assert "\\Problem name: sc_00000" in text
    assert "subject to" in text
    assert "binary" in text
