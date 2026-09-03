"""One channel code per station, chosen by a fixed order.

Picking every band a station carries duplicates the same ground motion at
different sampling rates, and includes bands that cannot produce a usable
arrival at all. The order is hard-coded because SEED band codes carry standard
sampling rates - it is a property of the code, not something to rediscover per
station-day.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sb_catalog.src.constants import (CHANNEL_PRIORITY, CHANNEL_PRIORITY_INDEX,
                                      select_channel)


def test_channel_selection():
    # The bands a deep-learning picker is trained on (BH, HH, EH, and SH as the
    # same instrument at a lower rate), the nodal geophone, and accelerometers
    # as a last resort. EP, EL and SL were dropped 2026-09-02: no training set
    # contains them, so picking on one is out of distribution rather than
    # merely second-best.
    #
    # BN joined 2026-09-03 with GeoNet, ranked at the BH tier: a BROADBAND
    # accelerometer rather than a strong-motion one. It shares HN's
    # out-of-distribution caveat - see tests/test_geonet.py.
    assert set(CHANNEL_PRIORITY) == {
        "HH", "EH", "SH", "BH", "BN", "DP", "HN", "CN"}
    for gone in ("EP", "EL", "SL"):
        assert gone not in CHANNEL_PRIORITY
        assert select_channel([gone]) is None, f"{gone} must not be pickable"

    def rank(c):
        return CHANNEL_PRIORITY_INDEX[c]

    # An accelerometer is picked only when the station offers no seismometer or
    # geophone at all. Their SNR on the small events that dominate a catalogue
    # is poor, so picking one where a broadband exists loses exactly the events
    # the catalogue is for. This outranks rate proximity, which is only a
    # resampling convenience.
    for accel in ("HN", "CN"):
        for other in ("HH", "EH", "SH", "BH", "DP"):
            assert rank(other) < rank(accel), f"{other} should outrank {accel}"

    # Everything else is ordered by |standard rate - 100 Hz|: the 100 Hz group
    # first, then S (~50), then B (40), then D (250).
    for hi in ("HH", "EH"):
        for lo in ("SH", "BH", "DP"):
            assert rank(hi) < rank(lo), f"{hi} should outrank {lo}"
    assert rank("SH") < rank("BH") < rank("DP")

    # A broadband station picks its 100 Hz channel, not its 40 Hz one.
    assert select_channel(["HH", "BH", "HN", "LH"]) == "HH"
    assert select_channel(["BH", "LH", "VH"]) == "BH"

    # The correction of 2026-09-02: a station with an accelerometer and a
    # broadband picks the broadband, even though HN is nearer 100 Hz. 139
    # station-locations across the five campaigns are affected.
    assert select_channel(["HN", "BH"]) == "BH"
    assert select_channel(["HN", "SH"]) == "SH"
    assert select_channel(["HN", "DP"]) == "DP"
    # ...but an accelerometer-only station is still picked, not skipped.
    assert select_channel(["HN"]) == "HN"
    assert select_channel(["HN", "LH"]) == "HN"

    # A nodal deployment offering only a geophone still gets picked. Note that
    # outside the 2014 campaign most DP station-days are not in miniSEED yet,
    # so these are expected to miss at read time rather than pick.
    assert select_channel(["DP"]) == "DP"
    assert select_channel(["EP", "DP"]) == "DP"   # EP is no longer pickable

    # Unlisted bands are ignored, not ranked. LH is 1 Hz: no usable arrival.
    assert select_channel(["LH", "VH", "UH"]) is None
    assert select_channel([]) is None

    # Order of the input must not matter - the priority does.
    assert select_channel(["HN", "EH"]) == select_channel(["EH", "HN"]) == "EH"

    # Whitespace and duplicates in metadata are tolerated.
    assert select_channel([" HH ", "HH", "BH"]) == "HH"

    # Both metadata formats in the repo: western_states.csv stores bands,
    # networks/*.zip stores full SEED codes. Accepting only bands made the
    # second select nothing, which skips the station silently.
    assert select_channel(["HHZ", "BHN", "EHZ", "HNE"]) == "HH"
    assert select_channel(["BHZ", "BHN", "BHE"]) == "BH"
    assert select_channel(["LHZ", "VHZ"]) is None
    assert select_channel(["HH", "BH"]) == select_channel(["HHZ", "BHZ"]) == "HH"

    print("PASS  the study's channel set, and nothing else")
    print("PASS  ordered by distance from 100 Hz, then instrument quality")
    print("PASS  broadband stations pick 100 Hz over 40 Hz")
    print("PASS  nodal-only stations still picked")
    print("PASS  unpickable bands ignored; no channel returns None")
    print("PASS  selection independent of input order")
    print("\nall channel-selection checks passed")


if __name__ == "__main__":
    test_channel_selection()
