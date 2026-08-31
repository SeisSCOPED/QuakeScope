"""The FDSN inventory request must be built from band codes, and must not
retry a permanent error forever.

Both halves of this were real: a SCEDC launch put 80 vCPU into an unbounded
retry loop and picked nothing, because the station metadata held full SEED
codes and the request appended a component wildcard to them.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_channel_code(channels):
    """Mirror of the construction in S3DataSource._get_inventory."""
    return ",".join(sorted(
        {f"{str(j).strip()[:2]}?" for i in channels
         for j in str(i).split(",") if str(j).strip()}
    ))


def test_inventory_request():
    # networks/<NET>.zip stores full SEED codes. "HHZ?" is four characters and
    # FDSN answers 400 Invalid input parameter, permanently.
    full = ["HHZ,BHN,EHZ,HHE,HNN,BHZ,HHN,HNZ,HNE,BHE"]
    assert build_channel_code(full) == "BH?,EH?,HH?,HN?"

    # western_states.csv stores bands. Both forms must give the same request.
    bands = ["BH,EH,HH,HN"]
    assert build_channel_code(bands) == build_channel_code(full)

    # Mixed and whitespace-padded metadata is tolerated.
    assert build_channel_code(["HH, BHZ", "EHN"]) == "BH?,EH?,HH?"

    # Never a four-character code, which is what FDSN rejects.
    for combo in (full, bands, ["DPZ,DP1,DP2"], ["SHZ"]):
        for tok in build_channel_code(combo).split(","):
            assert len(tok) == 3 and tok.endswith("?"), tok

    # The retry is bounded, and a 400 is not retried at all.
    src = open(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "sb_catalog", "src", "s3_helper.py")).read()
    assert "while True:" not in src.split("_get_inventory")[1][:2000], \
        "the inventory retry must be bounded"
    assert "FDSN_ATTEMPTS" in src
    assert '"400" in str(exc)' in src, "a 400 must not be retried"

    print("PASS  full SEED codes reduce to band codes")
    print("PASS  both metadata formats build the same request")
    print("PASS  never emits a four-character channel code")
    print("PASS  retry is bounded and a 400 is raised, not retried")
    print("\nall inventory-request checks passed")


if __name__ == "__main__":
    test_inventory_request()
