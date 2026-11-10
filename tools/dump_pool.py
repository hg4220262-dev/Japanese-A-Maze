"""Dump the contents of bunsetsu_pool_*.pkl to a CSV file.

Usage (run from the folder that contains maze_japanese.py):
    python tools/dump_pool.py bunsetsu_pool_v21_g6.pkl pool_g6.csv

Output columns: text, pos, particle, freq, length
Also prints counts by part of speech and by particle, and the 20 most frequent entries.
"""
import csv
import pickle
import sys
from collections import Counter


class _BunsetsuEntry:
    """Same shape as _BunsetsuEntry in maze_japanese.py (for loading)."""
    __slots__ = ("text", "pos", "particle", "freq", "length")


class _Unpickler(pickle.Unpickler):
    # The pkl is created by running maze_japanese.py directly, so the class is
    # stored as __main__._BunsetsuEntry. Map it here so the file loads anywhere.
    def find_class(self, module, name):
        if name == "_BunsetsuEntry":
            return _BunsetsuEntry
        return super().find_class(module, name)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src.rsplit(".", 1)[0] + ".csv"

    with open(src, "rb") as f:
        state = _Unpickler(f).load()
    entries = state["entries"]

    with open(dst, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["text", "pos", "particle", "freq", "length"])
        for e in entries:
            w.writerow([e.text, e.pos, e.particle, f"{e.freq:.3f}", e.length])

    print(f"Wrote {len(entries):,} bunsetsu to {dst}")
    print("By POS:", dict(Counter(e.pos for e in entries)))
    print("By particle:", dict(Counter(e.particle or "(none)" for e in entries)))
    print("Top 20 by frequency:")
    for e in sorted(entries, key=lambda e: -e.freq)[:20]:
        print(f"  {e.text}\t{e.pos}\t{e.freq:.2f}")


if __name__ == "__main__":
    main()
