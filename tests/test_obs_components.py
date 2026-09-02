"""The `obs` weight wants four components. The obs campaign has three.

`obs` declares `component_order: "Z12H"` and `in_channels: 4` - Z, two
horizontals and a hydrophone - while the worker requests `ZNE12`, which has no
`H`. That mismatch was flagged as a possible silent failure: a model that loads
and picks on mis-ordered or under-filled traces without complaining.

Checked 2026-09-02, and it is not a problem. Two facts settle it:

1. **There is no hydrophone to fetch.** Across the 3,389 stations in the obs
   campaign the band+instrument codes present are BH, CN, EH, EL, EP, HH, HN,
   SH and SL. A hydrophone is a pressure channel (`?D`, e.g. `BDH`), and no
   station offers one. The component loop also builds a channel by appending the
   component letter to the band+instrument code, so asking for `H` on a `BH`
   station would request `BHH`, not `BDH` - adding `H` to `--components` could
   not reach a hydrophone even where one existed.

2. **SeisBench handles both gaps itself**, which these tests pin: a missing
   component is zero-filled, and `N`/`E` map onto the `1`/`2` slots, so the
   station naming convention does not change the result.
"""

import numpy as np
import obspy
import pytest

sbm = pytest.importorskip("seisbench.models")


def _stream(letters, signals=None, n=6000, band="HH"):
    rng = np.random.default_rng(1)
    if signals is None:
        signals = {c: (rng.standard_normal(n) * 500).astype(np.float32)
                   for c in letters}
    st = obspy.Stream()
    for letter, data in zip(letters, (signals[c] for c in signals)):
        tr = obspy.Trace(np.asarray(data, dtype=np.float32))
        tr.stats.network, tr.stats.station, tr.stats.location = "XX", "OBS1", ""
        tr.stats.channel, tr.stats.sampling_rate = f"{band}{letter}", 100.0
        st += tr
    return st


@pytest.fixture(scope="module")
def obs_model():
    return sbm.PhaseNet.from_pretrained("obs")


def test_obs_expects_four_components(obs_model):
    assert obs_model.component_order == "Z12H"
    assert obs_model.in_channels == 4


def test_missing_hydrophone_is_zero_filled(obs_model):
    """Three components in, four expected: the absent one must be treated as
    silence, not as whatever happened to be adjacent in memory."""
    rng = np.random.default_rng(2)
    sig = {c: (rng.standard_normal(6000) * 500).astype(np.float32)
           for c in "Z12"}

    without = obs_model.annotate(_stream("Z12", sig))

    with_zero = _stream("Z12", sig)
    h = with_zero[0].copy()
    h.stats.channel = "HHH"
    h.data = np.zeros_like(h.data)
    with_zero += h
    explicit = obs_model.annotate(with_zero)

    assert len(without) == len(explicit)
    for a, b in zip(without, explicit):
        assert np.allclose(a.data, b.data, atol=1e-6)


def test_ne_naming_maps_onto_the_12_slots(obs_model):
    """obs stations use both conventions - 820 HHN/HHE against 331 HH1/HH2 in
    the campaign - so the same ground motion must annotate the same either way,
    and must not collapse to the vertical alone."""
    rng = np.random.default_rng(3)
    sig = {c: (rng.standard_normal(6000) * 500).astype(np.float32)
           for c in "Z12"}

    native = obs_model.annotate(_stream("Z12", sig))
    ne = obs_model.annotate(_stream("ZNE", sig))
    assert len(native) == len(ne)
    for a, b in zip(native, ne):
        assert np.allclose(a.data, b.data, atol=1e-6), \
            "N/E must map onto the 1/2 slots, not be dropped"

    vertical_only = obs_model.annotate(_stream("Z", {"Z": sig["Z"]}))
    differs = any(not np.allclose(a.data, b.data, atol=1e-6)
                  for a, b in zip(ne, vertical_only))
    assert differs, "horizontals were dropped rather than used"
