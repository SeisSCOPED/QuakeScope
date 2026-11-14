"""Picker precision remains safe on CPUs without AMX."""

import pytest


@pytest.mark.parametrize("has_amx, expected", [(False, "fp32"), (True, "bf16")])
def test_bf16_requires_amx(monkeypatch, has_amx, expected):
    from sb_catalog.src import picker

    monkeypatch.setattr(picker, "amx_bf16_available", lambda: has_amx)
    assert picker.resolve_precision("bf16") == expected
    assert picker.resolve_precision("fp32") == "fp32"


def test_amx_can_be_disabled(monkeypatch):
    from sb_catalog.src import picker

    monkeypatch.setenv("QS_AMX", "off")
    assert not picker.amx_bf16_available()
    assert picker.resolve_precision("bf16") == "fp32"
