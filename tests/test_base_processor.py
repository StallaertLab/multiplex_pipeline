# tests/test_baseop.py
from typing import Any, Mapping

import pytest

from multiplex_pipeline.processors.base import BaseOp


# ---- Test helper: minimal concrete class ----
class DummyOp(BaseOp):
    # simulate what your @register decorator would set
    kind = "unit_test_kind"
    type_name = "dummy"

    EXPECTED_INPUTS = 1
    EXPECTED_OUTPUTS = 1

    def validate_config(self, cfg: Mapping[str, Any]) -> None:
        # accept anything for these tests
        return

    def run(self, *sources):
        return sources


# ----------------- cfg / init -----------------
def test_cfg_is_stored_as_dict():
    op = DummyOp(a=1, b="x")
    assert op.cfg == {"a": 1, "b": "x"}


# -------------- _normalize_names --------------
@pytest.mark.parametrize(
    "value, expected",
    [
        (None, []),
        ("img", ["img"]),
        (["a"], ["a"]),
        (("x",), ["x"]),
    ],
)
def test_normalize_names_valid(value, expected):
    assert DummyOp._normalize_names(value, "inputs") == expected


@pytest.mark.parametrize("bad", [123, 3.14, ["a", 1], object()])
def test_normalize_names_invalid_raises(bad):
    with pytest.raises(TypeError):
        DummyOp._normalize_names(bad, "outputs")


# ---------------- validate_io OK --------------
def test_validate_io_exact_match():
    op = DummyOp()
    ins, outs = op.validate_io(inputs="src", outputs="dst")
    assert ins == ["src"]
    assert outs == ["dst"]


def test_validate_io_accepts_list_syntax():
    op = DummyOp()
    ins, outs = op.validate_io(inputs=["src"], outputs=["dst"])
    assert ins == ["src"]
    assert outs == ["dst"]


# ------------- validate_io errors -------------
def test_validate_io_wrong_input_count_raises():
    class TwoIn(DummyOp):
        EXPECTED_INPUTS = 2

    op = TwoIn()
    with pytest.raises(ValueError):
        op.validate_io(inputs=["a"], outputs="dst")  # too few

    with pytest.raises(ValueError):
        op.validate_io(inputs=["a", "b", "c"], outputs="dst")  # too many


def test_validate_io_wrong_output_count_raises():
    class TwoOut(DummyOp):
        EXPECTED_OUTPUTS = 2

    op = TwoOut()
    with pytest.raises(ValueError):
        op.validate_io(inputs="a", outputs=["x"])  # too few

    with pytest.raises(ValueError):
        op.validate_io(inputs="a", outputs=["x", "y", "z"])  # too many


# ------------- validate_io skip checks --------
def test_validate_io_skips_when_none():
    class Flexible(DummyOp):
        EXPECTED_INPUTS = None
        EXPECTED_OUTPUTS = None

    op = Flexible()
    # Any counts should pass when expected is None
    ins, outs = op.validate_io(inputs=["a", "b", "c"], outputs=["x", "y"])
    assert ins == ["a", "b", "c"]
    assert outs == ["x", "y"]


# --------------- __repr__ / __str__ ----------
def test_repr_includes_kind_type_and_cfg():
    op = DummyOp(alpha=42)
    r = repr(op)
    assert "DummyOp" in r
    assert "unit_test_kind" in r
    assert "dummy" in r
    assert "alpha" in r


def test_str_is_human_friendly():
    op = DummyOp(beta=True)
    s = str(op)
    # "kind:type cfg"
    assert "unit_test_kind:dummy" in s
    assert "beta" in s
