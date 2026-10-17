"""Reading and writing every table format pandas can handle.

Every loader returns an ordered mapping of ``name -> DataFrame``.  A CSV yields
exactly one entry; an .xlsx yields one per sheet; an .html file yields one per
table found on the page.  That single shape lets the GUI offer a uniform table
picker instead of special-casing each format.

Optional integrations (parquet, .xls, HDF5) are probed at import time and
reported through :func:`available_formats` rather than blowing up mid-load.
"""
from __future__ import annotations

import importlib
import importlib.util
import io
import os
import sqlite3
from dataclasses import dataclass, field

import pandas as pd

# --------------------------------------------------------------------------
# capability probing
# --------------------------------------------------------------------------


def _has(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


CAPS = {
    "pyarrow": _has("pyarrow"),
    "fastparquet": _has("fastparquet"),
    "xlrd": _has("xlrd"),
    "openpyxl": _has("openpyxl"),
    "odf": _has("odf"),
    "tables": _has("tables"),
    "sqlalchemy": _has("sqlalchemy"),
    "lxml": _has("lxml"),
    "clipboard": bool(
        __import__("shutil").which("wl-paste")
        or __import__("shutil").which("xclip")
        or __import__("shutil").which("xsel")
    ),
}


@dataclass
class FormatSpec:
    key: str
    label: str
    extensions: tuple[str, ...]
    needs: tuple[str, ...] = ()      # every one of these is required
    multi_table: bool = False
    note: str = ""
    needs_any: tuple[str, ...] = ()  # at least one of these is required

    @property
    def available(self) -> bool:
        if not all(CAPS.get(m, False) for m in self.needs):
            return False
        if self.needs_any and not any(CAPS.get(m, False) for m in self.needs_any):
            return False
        return True

    @property
    def missing(self) -> list[str]:
        gaps = [m for m in self.needs if not CAPS.get(m, False)]
        if self.needs_any and not any(CAPS.get(m, False) for m in self.needs_any):
            gaps.append(" or ".join(self.needs_any))
        return gaps


FORMATS: list[FormatSpec] = [
    FormatSpec("csv", "CSV / delimited text", (".csv", ".tsv", ".txt", ".dat", ".psv")),
    FormatSpec("fwf", "Fixed-width text", (".fwf",)),
    FormatSpec("excel", "Excel workbook", (".xlsx", ".xlsm", ".xlsb"), ("openpyxl",), True),
    FormatSpec("xls", "Excel 97-2003 (.xls)", (".xls",), ("xlrd",), True),
    FormatSpec("ods", "OpenDocument spreadsheet", (".ods",), ("odf",), True),
    FormatSpec("json", "JSON / JSON lines", (".json", ".jsonl", ".ndjson")),
    FormatSpec("html", "HTML table(s)", (".html", ".htm")),
    FormatSpec("xml", "XML", (".xml",)),
    FormatSpec("parquet", "Parquet", (".parquet",), (), False, "", ("pyarrow", "fastparquet")),
    FormatSpec("feather", "Feather / Arrow IPC", (".feather", ".arrow"), ("pyarrow",)),
    FormatSpec("hdf5", "HDF5", (".h5", ".hdf5", ".hdf"), ("tables",), True),
    FormatSpec("sql", "SQL database (SQLite file)", (".sqlite", ".db", ".sqlite3"), (), True),
    FormatSpec("pickle", "pandas pickle", (".pkl", ".pickle"), ()),
    FormatSpec("clipboard", "System clipboard", (), ("clipboard",)),
]

FORMAT_INDEX = {f.key: f for f in FORMATS}

_EXT_TO_FORMAT = {}
for _f in FORMATS:
    for _e in _f.extensions:
        _EXT_TO_FORMAT.setdefault(_e, _f.key)

TEXT_ENCODINGS = ("utf-8", "utf-8-sig", "latin-1", "cp1252")


def available_formats() -> list[FormatSpec]:
    return list(FORMATS)


def format_for_path(path: str) -> str | None:
    return _EXT_TO_FORMAT.get(os.path.splitext(path)[1].lower())


def describe_capabilities() -> str:
    parts = []
    for spec in FORMATS:
        parts.append(f"{spec.label}: {'yes' if spec.available else 'MISSING ' + ', '.join(spec.missing)}")
    return "\n".join(parts)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _clean_kwargs(kwargs: dict) -> dict:
    """Drop blank GUI values and coerce the couple of kwargs the GUI sends as text."""
    out = {}
    for k, v in kwargs.items():
        if v is None or v == "" or v == []:
            continue
        if k in ("header", "skiprows", "nrows", "index_col", "sheet_index", "table_index", "n_components"):
            try:
                v = int(v)
            except (TypeError, ValueError):
                continue
        if k == "na_values" and isinstance(v, str):
            v = [s.strip() for s in v.split(",") if s.strip()]
        out[k] = v
    return out


def _read_text_with_encoding(path: str, encoding: str, reader):
    """Try an explicit encoding, then fall back through the common ones."""
    encodings = [encoding] if encoding and encoding != "auto" else list(TEXT_ENCODINGS)
    last = None
    for enc in encodings:
        try:
            with open(path, "rb") as fh:
                head = fh.read(64 * 1024)
            head.decode(enc)
            return reader(enc)
        except UnicodeDecodeError as exc:
            last = exc
        except Exception as exc:  # pragma: no cover - surfaced to the GUI
            last = exc
            break
    if last is not None:
        raise last
    raise RuntimeError(f"could not decode {path}")


def list_tables(path: str, fmt: str | None = None) -> list[str]:
    """Names of the sheets/tables inside a file, without loading the data."""
    fmt = fmt or format_for_path(path) or "csv"
    try:
        if fmt in ("excel", "xls", "ods"):
            from openpyxl import load_workbook  # only used for the modern formats

            if fmt == "excel":
                wb = load_workbook(path, read_only=True, data_only=True)
                names = list(wb.sheetnames)
                wb.close()
                return names
            return []
        if fmt == "hdf5":
            with pd.HDFStore(path, mode="r") as store:
                return list(store.keys())
        if fmt == "sql":
            with sqlite3.connect(path) as con:
                rows = con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')").fetchall()
            return [r[0] for r in rows]
        if fmt == "html":
            tables = pd.read_html(path)
            return [f"table {i + 1} ({len(t)} rows)" for i, t in enumerate(tables)]
        if fmt in ("parquet", "feather"):
            import pyarrow.parquet as pq

            if fmt == "parquet":
                md = pq.ParquetFile(path)
                if md.num_row_groups > 1:
                    return [f"row group {i}" for i in range(md.num_row_groups)]
    except Exception:
        return []
    return []


# --------------------------------------------------------------------------
# readers
# --------------------------------------------------------------------------


def _read_csv(path, opts):
    encoding = opts.pop("encoding", "auto")
    sep = opts.pop("sep", None)
    if sep is None:
        sep = opts.pop("delimiter", None)
    if sep in ("\\t", "tab"):
        sep = "\t"
    if sep is None:
        sep = ","

    def attempt(separator, **extra):
        return _read_text_with_encoding(
            path, encoding, lambda enc: pd.read_csv(path, encoding=enc, sep=separator, **extra)
        )

    df = attempt(sep, **opts)
    # A wrong separator silently yields a single fat column. Probe the usual
    # suspects before handing that back to the caller.
    if df.shape[1] == 1 and not any(k in opts for k in ("names", "usecols")):
        for alt in (";", "\t", "|"):
            try:
                if attempt(alt, nrows=100).shape[1] > 1:
                    return {"data": attempt(alt, **opts)}
            except Exception:
                continue
    return {"data": df}


def _read_fwf(path, opts):
    encoding = opts.pop("encoding", "auto")
    return {"data": _read_text_with_encoding(path, encoding, lambda enc: pd.read_fwf(path, encoding=enc, **opts))}


def _read_excel(path, opts, engine=None):
    engine = engine or opts.pop("engine", None)
    sheet = opts.pop("sheet_name", None)
    kwargs = dict(**opts)
    if engine:
        kwargs["engine"] = engine
    if sheet in (None, "", "all"):
        book = pd.read_excel(path, sheet_name=None, **kwargs)
        return {str(k): v for k, v in book.items()}
    return {str(sheet): pd.read_excel(path, sheet_name=sheet, **kwargs)}


def _read_json(path, opts):
    orient = opts.pop("orient", None)
    lines = opts.pop("lines", False)
    kwargs = dict(**opts)
    if orient:
        kwargs["orient"] = orient
    if lines:
        kwargs["lines"] = True
    df = pd.read_json(path, **kwargs)
    return {"data": df}


def _read_html_tables(path, opts):
    match = opts.pop("match", None)
    try:
        tables = pd.read_html(path, match=match, **opts)
    except ValueError:
        # pandas can also read a plain file of pasted table text
        tables = pd.read_html(io.StringIO(open(path, encoding="utf-8", errors="replace").read()), **opts)
    return {f"table {i + 1}": t for i, t in enumerate(tables)}


def _read_xml(path, opts):
    return {"data": pd.read_xml(path, **opts)}


def _read_parquet(path, opts):
    return {"data": pd.read_parquet(path, **opts)}


def _read_feather(path, opts):
    return {"data": pd.read_feather(path, **opts)}


def _read_hdf(path, opts):
    key = opts.pop("key", None)
    if key:
        return {key: pd.read_hdf(path, key=key, **opts)}
    with pd.HDFStore(path, mode="r") as store:
        keys = list(store.keys())
    if not keys:
        raise ValueError("no tables inside the HDF5 file")
    return {k.lstrip("/"): pd.read_hdf(path, key=k, **opts) for k in keys}


def _read_sql(path, opts):
    query = opts.pop("query", None) or opts.pop("sql", None)
    table = opts.pop("table", None)
    with sqlite3.connect(path) as con:
        if query:
            return {"query": pd.read_sql_query(query, con, **opts)}
        if table:
            return {table: pd.read_sql_query(f'SELECT * FROM "{table}"', con, **opts)}
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view')")]
        return {n: pd.read_sql_query(f'SELECT * FROM "{n}"', con, **opts) for n in names}


def _read_pickle(path, opts):
    obj = pd.read_pickle(path)
    if isinstance(obj, pd.DataFrame):
        return {"data": obj}
    if isinstance(obj, dict):
        return {str(k): v for k, v in obj.items() if isinstance(v, pd.DataFrame)}
    raise ValueError(f"pickle holds a {type(obj).__name__}, not a DataFrame")


def _read_clipboard(path, opts):
    return {"data": pd.read_clipboard(**opts)}


_READERS = {
    "csv": _read_csv,
    "fwf": _read_fwf,
    "excel": lambda p, o: _read_excel(p, o, engine="openpyxl"),
    "xls": lambda p, o: _read_excel(p, o, engine="xlrd"),
    "ods": lambda p, o: _read_excel(p, o, engine="odf"),
    "json": _read_json,
    "html": _read_html_tables,
    "xml": _read_xml,
    "parquet": _read_parquet,
    "feather": _read_feather,
    "hdf5": _read_hdf,
    "sql": _read_sql,
    "pickle": _read_pickle,
    "clipboard": _read_clipboard,
}

# kwargs the GUI is allowed to pass through, per format
FORMAT_OPTIONS = {
    "csv": ["sep", "encoding", "header", "names", "index_col", "skiprows", "nrows", "na_values", "decimal", "quotechar", "comment"],
    "fwf": ["encoding", "header", "skiprows", "nrows", "na_values"],
    "excel": ["sheet_name", "header", "skiprows", "nrows", "usecols", "na_values"],
    "xls": ["sheet_name", "header", "skiprows", "nrows"],
    "ods": ["sheet_name", "header", "skiprows", "nrows"],
    "json": ["orient", "lines"],
    "html": ["match", "header", "skiprows"],
    "xml": ["xpath"],
    "parquet": ["columns"],
    "feather": ["columns"],
    "hdf5": ["key"],
    "sql": ["query", "table"],
    "pickle": [],
    "clipboard": ["header", "sep"],
}


def read_all(path: str, fmt: str | None = None, **options) -> dict[str, pd.DataFrame]:
    """Load every table found at ``path``.  Returns an ordered {name: DataFrame}."""
    fmt = fmt or format_for_path(path)
    if fmt is None:
        raise ValueError(f"unrecognised file extension for {path!r} — choose a format explicitly")
    spec = FORMAT_INDEX.get(fmt)
    if spec is None:
        raise ValueError(f"unknown format {fmt!r}")
    if not spec.available:
        raise RuntimeError(
            f"{spec.label} needs the {', '.join(spec.missing)} package.\n"
            f"Install it with:  .venv/bin/pip install {' '.join(spec.missing)}"
        )
    allowed = set(FORMAT_OPTIONS.get(fmt, []))
    opts = {k: v for k, v in _clean_kwargs(options).items() if k in allowed}
    tables = _READERS[fmt](path, opts)
    if not tables:
        raise ValueError("no tables were found in that file")
    return {k: v for k, v in tables.items() if isinstance(v, pd.DataFrame)}


def read_one(path: str, fmt: str | None = None, table: str | None = None, **options) -> pd.DataFrame:
    """Load a single table (the first one, unless ``table`` names another)."""
    tables = read_all(path, fmt=fmt, **options)
    if table and table in tables:
        return tables[table]
    if len(tables) > 1 and table is None:
        # unambiguous only when every sheet/table is identical in shape
        first = next(iter(tables.values()))
        if not all(t.shape == first.shape for t in tables.values()):
            raise ValueError(
                f"this file holds {len(tables)} tables ({', '.join(tables)}); pick one explicitly"
            )
    return next(iter(tables.values()))


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------


def save_df(df: pd.DataFrame, path: str, fmt: str | None = None, **options) -> str:
    fmt = fmt or format_for_path(path)
    if fmt is None:
        raise ValueError("cannot infer the format from the filename — choose one explicitly")
    opts = _clean_kwargs(options)
    index = opts.pop("index", False)
    if fmt in ("csv", "fwf"):
        df.to_csv(path, index=index, **opts)
    elif fmt == "excel":
        df.to_excel(path, index=index, engine="openpyxl", **opts)
    elif fmt == "ods":
        df.to_excel(path, index=index, engine="odf", **opts)
    elif fmt == "json":
        df.to_json(path, **opts)
    elif fmt == "html":
        df.to_html(path, **opts)
    elif fmt == "xml":
        df.to_xml(path, **opts)
    elif fmt == "parquet":
        df.to_parquet(path, **opts)
    elif fmt == "feather":
        df.to_feather(path, **opts)
    elif fmt == "hdf5":
        df.to_hdf(path, key=opts.pop("key", "data"), **opts)
    elif fmt == "pickle":
        df.to_pickle(path, **opts)
    elif fmt == "sql":
        with sqlite3.connect(path) as con:
            df.to_sql(opts.pop("table", "data"), con, if_exists=opts.pop("if_exists", "replace"), index=False)
    else:
        raise ValueError(f"writing {fmt!r} is not supported")
    return path
