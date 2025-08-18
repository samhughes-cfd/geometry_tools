# n_slicer\parsers\base.py

from __future__ import annotations
import pandas as pd

def norm_colname(s: str) -> str:
    s = s.strip().lower()
    repl = {
        "β": "beta",
        "°": "deg",
        "#": "",
        "[": "",
        "]": "",
        "(": "",
        ")": "",
        "-": "_",
        " ": "_",
        "/": "_over_",
        "\\": "_over_",
        "%": "pct",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    while "__" in s:
        s = s.replace("__", "_")
    return s


class BaseCSVParser:
    """Common CSV conveniences: tolerate '#'-comments and odd headers."""
    def __init__(self, csv_path: str, assume_no_header: bool = False):
        self.csv_path = csv_path
        self.assume_no_header = assume_no_header

    def _read_csv_best_effort(self) -> pd.DataFrame:
        if self.assume_no_header:
            df = pd.read_csv(self.csv_path, comment="#", header=None)
            return df
        # Try with comment filtering first
        try:
            df = pd.read_csv(self.csv_path, comment="#")
            if df.shape[1] == 0 or set(df.columns) == {0}:
                raise Exception("Suspicious header, retry")
        except Exception:
            # Fall back to a straight read
            df = pd.read_csv(self.csv_path, header=0)
        # Strip leading '# ' from headers if present
        df.columns = [c.lstrip("# ").strip() for c in df.columns]
        return df