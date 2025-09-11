def test_schema_import():
    from bbml.data.schema import BRANCH_CAND_COLS

    assert "var_id" in BRANCH_CAND_COLS
