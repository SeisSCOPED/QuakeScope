"""Compaction must find partitions, and must never delete what it did not copy.

Every bug fixed here was silent. The compactor ran, exited 0, logged a summary,
and did nothing - or would have destroyed data while reporting success.
"""

import ast
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SRC = (pathlib.Path(__file__).parent.parent
       / "sb_catalog/src/parquet_compact.py")


def test_discovery_walks_all_three_partition_levels():
    """`fs.ls` lists ONE level.

    Against `picks/` it returns `network=CI` and stops, so the old test -
    "network=" and "year=" and "month=" all in one string - matched nothing.
    Every run reported "Found 0 partitions" and exited 0. A compactor that
    silently does nothing is worse than one that fails, because it looks like
    it worked.
    """
    src = SRC.read_text()
    assert "fs.glob(" in src, "partition discovery must recurse"
    assert 'network=*/year=*/month=*' in src


def test_paths_are_bucket_qualified():
    # s3fs paths carry the bucket. `f"{prefix}/picks/"` resolved to a bucket
    # literally named "scedc" and raised before listing anything.
    src = SRC.read_text()
    assert 'picks_prefix = f"{bucket}/{prefix}/picks/"' in src
    assert 'part_prefix = f"{bucket}/{prefix}/picks/{partition}/"' in src


def test_never_deletes_before_verifying():
    """The delete must be unreachable unless the copy is proven good.

    The original wrote the new files, logged any write failure as an ERROR,
    and then deleted the originals anyway - losing picks permanently, with the
    loss recorded only in a log. Compaction is an optimisation and must never
    be able to destroy data it failed to copy.
    """
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_compact_partition")

    # Find the loop containing fs.rm, then confirm a row-count comparison and
    # an early return guard both appear before it in the function body.
    rm_line = min(
        (n.lineno for n in ast.walk(fn)
         if isinstance(n, ast.Attribute) and n.attr == "rm"),
        default=None)
    assert rm_line is not None, "expected a delete step"

    guard_returns = [n.lineno for n in ast.walk(fn)
                     if isinstance(n, ast.Return) and n.lineno < rm_line]
    assert len(guard_returns) >= 3, (
        "expected guards for incomplete write, unverifiable read, and row "
        "mismatch, each returning before the delete"
    )

    src_before_rm = "\n".join(SRC.read_text().splitlines()[:rm_line])
    assert "rows_after != rows_before" in src_before_rm, (
        "the delete must be gated on a row-count comparison"
    )


def test_compaction_matches_the_writer_codec():
    """pq.write_table defaults to SNAPPY; the writer uses zstd + dictionary.

    Compacting with the defaults re-encoded everything and made the catalogue
    32% LARGER - 8.2 MB to 10.9 MB, measured on estmp4 - which is a poor trade
    for fewer objects and is invisible unless you check the byte counts.
    """
    src = SRC.read_text()
    writer = (pathlib.Path(__file__).parent.parent
              / "sb_catalog/src/parquet_writer.py").read_text()
    assert 'compression: str = "zstd"' in writer, "writer codec moved"
    assert 'COMPACT_COMPRESSION = "zstd"' in src
    assert "use_dictionary=True" in src
