import re
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# =========================
# Page config
# =========================
st.set_page_config(page_title="Engagement Analytics Dashboard", layout="wide")


# =========================
# Theme / Styling (Green + White)
# =========================
def inject_css():
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff; }

        section[data-testid="stSidebar"] {
            background: #f6fbf7;
            border-right: 1px solid #dbeee0;
        }

        h1, h2, h3, h4, h5, h6 { color: #0b3d2e !important; }

        .tetr-card {
            background: #ffffff;
            border: 1px solid #dbeee0;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 1px 8px rgba(11, 61, 46, 0.06);
        }

        .tetr-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid #bfe3c9;
            background: #eaf7ee;
            color: #0b3d2e;
            font-weight: 600;
            font-size: 12px;
        }

        /* Make metrics more compact */
        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dbeee0;
            border-radius: 14px;
            padding: 10px 12px;
            box-shadow: 0 1px 8px rgba(11, 61, 46, 0.06);
        }
        div[data-testid="stMetric"] label {
            color: #2e6b57 !important;
            font-weight: 700 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =========================
# Constants
# =========================
ALLOWED_SHEETS = {
    "PG - B1 & B2",
    "PG - B3 & B4",
    "UG B7",
    "UG B6",
    "UG B5",
    "UG - B1 to B4",
    "Tetr-X-UG",
    "Tetr-X-PG",
}

TETR_X_SHEETS = {"Tetr-X-UG", "Tetr-X-PG"}

GREEN_PALETTE = {
    "Paid / Admitted": "#0b3d2e",
    "Will Pay": "#1f7a56",
    "Pending": "#6bbf8a",
    "Refunded": "#93d8b1",
    "Not Paid": "#cfeedd",
}


# =========================
# Helpers
# =========================
def clean_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return (
        str(x)
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("\u00a0", " ")
        .strip()
    )


def make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        c = clean_text(c)
        if c == "":
            c = "Unnamed"
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out


def normalize_binary(x) -> int:
    """
    Robust Yes/No normalization.
    """
    try:
        if x is None:
            return 0
        if isinstance(x, float) and np.isnan(x):
            return 0
        s = str(x).strip().lower()
        return 1 if s in {"yes", "y", "true", "1", "attended", "present"} else 0
    except Exception:
        return 0


def parse_date_safe(x):
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT


def parse_event_date(val):
    """
    Handles:
      - Timestamp-like / excel dates
      - '2026-01-24 00:00:00'
      - '28-30.01.2026'  (takes start date)
      - '28.01 to 30.01.2026' (takes start date)
      - '28 Jan 2026', '28 Jan, 2026', '28 January 2026'
      - Extracts dd-mm-yyyy inside long event names
    """
    try:
        ts = pd.to_datetime(val, errors="coerce", dayfirst=True)
        if pd.notna(ts):
            return ts.normalize()
    except Exception:
        pass

    s = clean_text(val)
    if not s:
        return pd.NaT

    m = re.search(r"(\d{1,2})\D+(\d{1,2})\D+(\d{4})", s)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        try:
            return pd.Timestamp(int(yyyy), int(mm), int(dd))
        except Exception:
            return pd.NaT

    m2 = re.search(
        r"\b(\d{1,2})\s*(jan|january|feb|february|mar|march|apr|april|may|jun|june|jul|july|aug|august|sep|sept|september|oct|october|nov|november|dec|december)\s*,?\s*(\d{4})\b",
        s.lower(),
    )
    if m2:
        dd = int(m2.group(1))
        mon = m2.group(2)
        yyyy = int(m2.group(3))
        mon_map = {
            "jan": 1, "january": 1,
            "feb": 2, "february": 2,
            "mar": 3, "march": 3,
            "apr": 4, "april": 4,
            "may": 5,
            "jun": 6, "june": 6,
            "jul": 7, "july": 7,
            "aug": 8, "august": 8,
            "sep": 9, "sept": 9, "september": 9,
            "oct": 10, "october": 10,
            "nov": 11, "november": 11,
            "dec": 12, "december": 12,
        }
        try:
            return pd.Timestamp(yyyy, mon_map[mon], dd)
        except Exception:
            return pd.NaT

    return pd.NaT


def best_matching_col(df: pd.DataFrame, keywords, hard_excludes=None):
    hard_excludes = hard_excludes or []
    scored = []
    for c in df.columns:
        cl = clean_text(c).lower()
        if any(ex in cl for ex in hard_excludes):
            continue
        hit_strength = 0
        for k in keywords:
            if k in cl:
                hit_strength = max(hit_strength, len(k))
        if hit_strength > 0:
            scored.append((c, hit_strength, df[c].notna().sum()))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return scored[0][0]


def row_is_numeric_only(r):
    vals = [str(v).strip() for v in r.tolist() if str(v).strip() not in ("", "nan", "NaN")]
    if not vals:
        return True
    return all(re.fullmatch(r"\d+(\.\d+)?", v) for v in vals)


def is_probably_event_col(series: pd.Series) -> bool:
    """
    Attendance-like columns tend to be mostly in this allowed set.
    """
    s = series.fillna("").astype(str).str.strip().str.lower()
    if s.empty:
        return False
    allowed = {"yes", "no", "y", "n", "true", "false", "1", "0", "attended", "present", "absent", ""}
    return (s.isin(allowed)).mean() >= 0.6


def safe_datetime_range_with_padding(dates, pad_days=3):
    dates = [d for d in dates if pd.notna(d)]
    if not dates:
        return None
    mn = min(dates) - pd.Timedelta(days=pad_days)
    mx = max(dates) + pd.Timedelta(days=pad_days)
    return [mn, mx]


def infer_program_from_sheet(sheet_name: str) -> str:
    s = clean_text(sheet_name).lower()
    if "tetr-x-ug" in s or s.startswith("ug"):
        return "UG"
    if "tetr-x-pg" in s or s.startswith("pg"):
        return "PG"
    return ""


def infer_batch_from_sheet_name(sheet_name: str) -> str:
    """
    Derive batch label from sheet name:
      - 'UG B5' -> 'B5'
      - 'PG - B3 & B4' -> 'B3–B4'
      - 'UG - B1 To B4' -> 'B1–B4'
      - Tetr-X sheets don't encode batch -> return ""
    """
    s = clean_text(sheet_name).lower().replace("–", "-").replace("—", "-")

    if "tetr-x" in s:
        return ""

    # B1 to B4
    m = re.search(r"\bb\s*(\d+)\s*to\s*b\s*(\d+)\b", s)
    if m:
        return f"B{m.group(1)}–B{m.group(2)}"

    # B3 & B4
    m = re.search(r"\bb\s*(\d+)\s*&\s*b\s*(\d+)\b", s)
    if m:
        return f"B{m.group(1)}–B{m.group(2)}"

    # single B7
    m = re.search(r"\bb\s*(\d+)\b", s)
    if m:
        return f"B{m.group(1)}"

    return ""


# =========================
# Sheet-aware loader
# =========================
def load_sheet_structured(raw: pd.DataFrame):
    def row_text(i):
        return " | ".join([clean_text(v).lower() for v in raw.iloc[i, :].tolist()])

    header_row = None

    # Old style
    for i in range(min(40, len(raw))):
        t = row_text(i)
        if "student name" in t or "student names" in t:
            header_row = i
            break

    # New style
    if header_row is None:
        for i in range(min(80, len(raw))):
            t = row_text(i)
            has_name = "name" in t
            has_support = any(k in t for k in ["email", "batch", "country", "mobile", "contact", "status", "payment"])
            non_empty = sum(1 for v in raw.iloc[i, :].tolist() if clean_text(v) != "") >= 3
            if has_name and has_support and non_empty:
                header_row = i
                break

    if header_row is None:
        return None, None, "Could not find a header row ('Student Name' or 'Name') in top rows."

    # optional date row above header
    candidate_rows = list(range(max(0, header_row - 8), header_row))
    best_date_row, best_hits = None, -1
    for r in candidate_rows:
        hits = sum(1 for v in raw.iloc[r, :].tolist() if pd.notna(parse_event_date(v)))
        if hits > best_hits:
            best_hits, best_date_row = hits, r

    if best_date_row is not None and best_hits >= 3:
        date_row = best_date_row
        event_row = date_row - 1 if date_row - 1 >= 0 else None
    else:
        date_row = None
        event_row = None

    header_cells = [clean_text(x) for x in raw.iloc[header_row, :].tolist()]
    event_cells = (
        [clean_text(x) for x in raw.iloc[event_row, :].tolist()]
        if event_row is not None
        else [""] * len(header_cells)
    )

    cols = []
    for j, h in enumerate(header_cells):
        if h:
            cols.append(h)
        elif j < len(event_cells) and event_cells[j]:
            cols.append(event_cells[j])
        else:
            cols.append(f"Unnamed_{j}")
    cols = make_unique(cols)

    df = raw.iloc[header_row + 1 :, :].copy()
    df.columns = cols
    df = df.reset_index(drop=True).dropna(how="all")

    event_dates = {}
    if date_row is not None:
        date_cells = raw.iloc[date_row, :].tolist()
        for j, col in enumerate(cols):
            if j < len(date_cells):
                dt = parse_event_date(date_cells[j])
                if pd.notna(dt):
                    event_dates[col] = dt

    meta = {
        "header_row": header_row,
        "event_row": event_row,
        "date_row": date_row,
        "event_dates": event_dates,
        "date_hits": best_hits if best_hits >= 0 else None,
    }
    return df, meta, None


# =========================
# Excel I/O
# =========================
@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes: bytes):
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def read_raw_sheet(file_bytes: bytes, sheet_name: str):
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return pd.read_excel(xls, sheet_name=sheet_name, header=None).dropna(how="all")


# =========================
# Build dataset
# =========================
def build_dataset_from_sheet(file_bytes: bytes, sheet_name: str):
    raw = read_raw_sheet(file_bytes, sheet_name)
    df, meta, err = load_sheet_structured(raw)
    if err:
        return None, None, err

    # Add program + batch from sheet rules
    program = infer_program_from_sheet(sheet_name)
    inferred_batch = infer_batch_from_sheet_name(sheet_name)

    KW = {
        "name_strict": ["student name", "student names", "full name", "learner name", "student", "name"],
        "name_fallback": ["name"],
        "email": ["email", "e mail", "e-mail"],
        "phone": ["phone", "mobile", "mobile no", "phone number", "contact"],
        "country": ["country"],
        "batch": ["batch", "cohort", "group"],
        "conversion": ["conversion status", "conversion", "status", "admitted", "refunded", "pending", "will pay", "paid"],
        "payment_date": ["payment date", "payment", "paid date", "date of payment"],
    }

    name_col = best_matching_col(df, KW["name_strict"])
    if not name_col:
        name_col = best_matching_col(
            df,
            KW["name_fallback"],
            hard_excludes=["batch", "program", "session", "event", "country", "status", "payment"],
        )
    if not name_col:
        return None, None, "Student/Name column not detected."

    email_col = best_matching_col(df, KW["email"])
    phone_col = best_matching_col(df, KW["phone"])
    country_col = best_matching_col(df, KW["country"])
    batch_col = best_matching_col(df, KW["batch"])
    conversion_col = best_matching_col(df, KW["conversion"])
    payment_col = best_matching_col(df, KW["payment_date"])

    # Ensure Program exists
    df["Program"] = program if program else ""

    # Ensure Batch alignment ONLY for allowed sheets (and only using sheet logic)
    if inferred_batch:
        if not batch_col:
            df["Batch"] = inferred_batch
            batch_col = "Batch"
        else:
            bc = df[batch_col].astype(str).map(clean_text)
            df.loc[bc.eq(""), batch_col] = inferred_batch

    # Drop numeric-only summary rows where name is blank
    name_series = df[name_col].astype(str).map(clean_text)
    numeric_mask = df.apply(row_is_numeric_only, axis=1)
    blank_name_mask = name_series.map(lambda x: x == "" or x.lower() == "nan")
    df = df.loc[~(numeric_mask & blank_name_mask)].copy()

    # Detect event cols
    metadata_cols = {c for c in [name_col, email_col, phone_col, country_col, batch_col, conversion_col, payment_col, "Program"] if c}
    event_cols = [c for c in df.columns if c not in metadata_cols]
    event_cols = [c for c in event_cols if is_probably_event_col(df[c])]

    # Normalize events
    for c in event_cols:
        df[c] = df[c].apply(normalize_binary).astype(np.int8)

    df["participation_count"] = df[event_cols].sum(axis=1) if event_cols else 0

    # payment
    if payment_col:
        df[payment_col] = df[payment_col].apply(parse_date_safe)

    # conversion col
    if not conversion_col:
        df["Conversion Status"] = ""
        conversion_col = "Conversion Status"
    df[conversion_col] = df[conversion_col].astype(str).map(lambda x: clean_text(x).lower())

    # conversion category (with special handling for Tetr-X sheets)
    def conv_category(r):
        if sheet_name in TETR_X_SHEETS:
            return "Paid / Admitted"

        if payment_col and pd.notna(r.get(payment_col, pd.NaT)):
            return "Paid / Admitted"

        v = str(r.get(conversion_col, "")).lower()
        if "admitted" in v or "paid" in v:
            return "Paid / Admitted"
        if "will" in v:
            return "Will Pay"
        if "refund" in v:
            return "Refunded"
        if "pending" in v:
            return "Pending"
        return "Not Paid"

    df["conversion_category"] = df.apply(conv_category, axis=1)

    # Event dates: meta first, otherwise parse from event name
    event_dates = meta.get("event_dates", {}) or {}
    if not event_dates:
        for ev in event_cols:
            dt = parse_event_date(ev)
            if pd.notna(dt):
                event_dates[ev] = dt

    # retained: paid + attended after payment date
    def retained_flag(r):
        if sheet_name in TETR_X_SHEETS:
            # For paid-only sheets, retention still depends on after-payment attendance if payment date + event dates exist
            pass

        if not payment_col:
            return np.nan

        pay = r.get(payment_col, pd.NaT)
        if pd.isna(pay):
            # In Tetr-X, some payment date might be missing; still treat them as paid but retention cannot be timed
            return 1 if r.get("participation_count", 0) > 0 else 0

        if not event_dates:
            return 1 if r.get("participation_count", 0) > 0 else 0

        pay_d = pay.normalize()
        for ev in event_cols:
            if r.get(ev, 0) == 1:
                dt = event_dates.get(ev, pd.NaT)
                if pd.notna(dt) and dt > pay_d:
                    return 1
        return 0

    df["retained"] = df.apply(retained_flag, axis=1)

    # lead score
    def lead_score(r):
        score = int(r.get("participation_count", 0)) * 10
        for ev in event_cols:
            if r.get(ev, 0) == 1:
                evl = ev.lower()
                if "hackathon" in evl or "hackerthon" in evl:
                    score += 20
                elif "ama" in evl:
                    score += 15
                elif "masterclass" in evl:
                    score += 15
                elif "webinar" in evl:
                    score += 10
        cat = r.get("conversion_category", "Not Paid")
        if cat == "Paid / Admitted":
            score += 30
        elif cat == "Will Pay":
            score += 15
        elif cat == "Pending":
            score += 10
        if payment_col and r.get("retained", 0) == 1:
            score += 10
        return score

    df["lead_score"] = df.apply(lead_score, axis=1)

    ctx = {
        "meta": meta,
        "name_col": name_col,
        "email_col": email_col,
        "phone_col": phone_col,
        "country_col": country_col,
        "batch_col": batch_col,
        "program_col": "Program",
        "conversion_col": conversion_col,
        "payment_col": payment_col,
        "event_cols": event_cols,
        "event_dates": event_dates,
    }
    return df, ctx, None


# =========================
# UI helpers
# =========================
def compact_kpis(df_all, df_filtered, ctx):
    """
    Native Streamlit KPI row: compact + reliable (no HTML rendering issues).
    """
    name_col = ctx["name_col"]
    payment_col = ctx["payment_col"]

    total_all = int(df_all[name_col].notna().sum())
    active_all = int((df_all["participation_count"] > 0).sum())
    paid_all = int((df_all["conversion_category"] == "Paid / Admitted").sum())
    participants_all = int((df_all["participation_count"] > 0).sum())
    conv_all = (paid_all / participants_all * 100) if participants_all else 0.0

    total_f = int(df_filtered[name_col].notna().sum())
    active_f = int((df_filtered["participation_count"] > 0).sum())
    paid_f = int((df_filtered["conversion_category"] == "Paid / Admitted").sum())
    participants_f = int((df_filtered["participation_count"] > 0).sum())
    conv_f = (paid_f / participants_f * 100) if participants_f else 0.0

    if payment_col:
        ret_all = float(df_all["retained"].mean() * 100) if len(df_all) else 0.0
        ret_f = float(df_filtered["retained"].mean() * 100) if len(df_filtered) else 0.0
    else:
        ret_all, ret_f = None, None

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students", f"{total_f}", help=f"All: {total_all}")
    c2.metric("Active Students", f"{active_f}", help=f"All: {active_all}")
    c3.metric("Paid / Admitted", f"{paid_f}", help=f"All: {paid_all}")
    c4.metric("Conversion Rate", f"{conv_f:.1f}%", help=f"All: {conv_all:.1f}%")
    if payment_col:
        c5.metric("Retention Rate", f"{ret_f:.1f}%", help=f"All: {ret_all:.1f}%")
    else:
        c5.metric("Retention Rate", "N/A", help="Needs Payment Date column")


def plotly_green_layout(fig, height=None, x_tickangle=None, bottom_margin=None):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#0b3d2e"),
        title_font=dict(color="#0b3d2e"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8f4ec")
    fig.update_yaxes(showgrid=True, gridcolor="#e8f4ec")
    if height:
        fig.update_layout(height=height)
    if x_tickangle is not None:
        fig.update_layout(xaxis_tickangle=x_tickangle)
    if bottom_margin is not None:
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=bottom_margin))
    return fig


# =========================
# App Header
# =========================
st.markdown(
    """
    <div class="tetr-card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
        <div>
          <div style="font-size:28px; font-weight:900; color:#0b3d2e;">📊 Engagement Analytics Dashboard</div>
          <div style="margin-top:4px; color:#2e6b57; font-weight:600;">
            Analyzes only UG/PG batch sheets + Tetr-X paid sheets • Batch-aligned • Conversion + retention + event intelligence
          </div>
        </div>
        <div class="tetr-badge">Green / White Theme</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


uploaded_file = st.file_uploader("Upload Master Engagement Tracker Excel File", type=["xlsx"])
if not uploaded_file:
    st.stop()

file_bytes = uploaded_file.getvalue()
all_sheets = get_sheet_names(file_bytes)

# Only allow the specified sheets
sheets = [s for s in all_sheets if s in ALLOWED_SHEETS]
missing = [s for s in sorted(ALLOWED_SHEETS) if s not in set(all_sheets)]

if not sheets:
    st.error("None of the required batch sheets were found in this workbook.")
    st.write("Expected sheets:")
    for s in sorted(ALLOWED_SHEETS):
        st.write(f"- {s}")
    st.stop()

if missing:
    with st.expander("⚠️ Some required sheets are missing in this workbook (click to view)"):
        for s in missing:
            st.write(f"- {s}")


# =========================
# Sidebar: mode + sheet selection
# =========================
st.sidebar.markdown("## Mode")
mode = st.sidebar.radio("Choose view", ["Single sheet", "Compare (multi-sheet)"], index=1)
st.sidebar.markdown("---")

if mode == "Single sheet":
    selected_sheet = st.sidebar.selectbox("Select Sheet", sheets)
    selected_sheets = [selected_sheet]
else:
    # Default select all allowed sheets found
    selected_sheets = st.sidebar.multiselect("Select sheets to compare", options=sheets, default=sheets)

if not selected_sheets:
    st.warning("Select at least one sheet.")
    st.stop()


# =========================
# Load datasets
# =========================
dfs = []
contexts = []
errors = []

for sname in selected_sheets:
    df_i, ctx_i, err = build_dataset_from_sheet(file_bytes, sname)
    if err:
        errors.append((sname, err))
        continue
    df_i = df_i.copy()
    df_i["__sheet__"] = sname
    dfs.append(df_i)
    contexts.append((sname, ctx_i))

if errors and not dfs:
    st.error("Could not parse any selected sheets.")
    for sname, err in errors:
        st.write(f"- **{sname}**: {err}")
    st.stop()

if errors:
    with st.expander("⚠️ Some selected sheets could not be parsed (click to view)"):
        for sname, err in errors:
            st.write(f"- **{sname}**: {err}")


# =========================
# ✅ Robust event alignment (prevents .astype(int) crash)
# =========================
all_event_cols = sorted(set().union(*[set(ctx["event_cols"]) for _, ctx in contexts])) if contexts else []

all_event_dates = {}
for _, ctx in contexts:
    for ev, dt in (ctx.get("event_dates") or {}).items():
        if pd.isna(dt):
            continue
        if ev not in all_event_dates:
            all_event_dates[ev] = dt
        else:
            all_event_dates[ev] = min(all_event_dates[ev], dt)

for i in range(len(dfs)):
    for ev in all_event_cols:
        if ev not in dfs[i].columns:
            dfs[i][ev] = 0
        dfs[i][ev] = dfs[i][ev].apply(normalize_binary).astype(np.int8)

    dfs[i]["participation_count"] = dfs[i][all_event_cols].sum(axis=1) if all_event_cols else 0

df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

primary_ctx = contexts[0][1]
name_col = primary_ctx["name_col"]
country_col = primary_ctx["country_col"]
batch_col = primary_ctx["batch_col"] or ("Batch" if "Batch" in df.columns else None)
program_col = primary_ctx.get("program_col", "Program")
conversion_col = primary_ctx["conversion_col"]
payment_col = primary_ctx["payment_col"]
event_cols = all_event_cols
event_dates = all_event_dates


# =========================
# Sidebar filters
# =========================
st.sidebar.markdown("## Filters")

all_conv_vals = ["Paid / Admitted", "Will Pay", "Pending", "Refunded", "Not Paid"]
present_conv = [c for c in all_conv_vals if c in set(df["conversion_category"].unique())]
conv_filter = st.sidebar.multiselect("Conversion category", present_conv, default=present_conv)

min_part = st.sidebar.slider("Minimum participation count", 0, int(df["participation_count"].max() or 0), 0)

# Sheet filter in compare mode
sheet_filter = None
if "__sheet__" in df.columns and len(selected_sheets) > 1:
    st.sidebar.markdown("### Compare control")
    all_sheet_vals = sorted(df["__sheet__"].dropna().astype(str).unique().tolist())
    sheet_filter = st.sidebar.multiselect("Limit to sheet(s)", all_sheet_vals, default=all_sheet_vals)

# Program filter (UG/PG)
program_filter = None
if program_col in df.columns:
    progs = sorted([p for p in df[program_col].dropna().astype(str).map(clean_text).unique().tolist() if p])
    if progs:
        program_filter = st.sidebar.multiselect("Program", progs, default=progs)

# Batch filter
batch_filter = None
if batch_col and batch_col in df.columns:
    batches = sorted([b for b in df[batch_col].dropna().astype(str).map(clean_text).unique().tolist() if b])
    if batches:
        batch_filter = st.sidebar.multiselect("Batch", batches, default=batches)

# Country filter
country_filter = None
if country_col and country_col in df.columns:
    countries = sorted([c for c in df[country_col].dropna().astype(str).map(clean_text).unique().tolist() if c])
    if countries:
        country_filter = st.sidebar.multiselect("Country", countries, default=countries)

search_text = st.sidebar.text_input("Search student (contains)", "")

# Apply filters
fdf = df.copy()
fdf = fdf[fdf["conversion_category"].isin(conv_filter)]
fdf = fdf[fdf["participation_count"] >= min_part]
if sheet_filter is not None:
    fdf = fdf[fdf["__sheet__"].astype(str).isin(set(sheet_filter))]
if program_filter is not None and program_col in fdf.columns:
    fdf = fdf[fdf[program_col].astype(str).map(clean_text).isin(set(program_filter))]
if batch_col and batch_filter is not None and batch_col in fdf.columns:
    fdf = fdf[fdf[batch_col].astype(str).map(clean_text).isin(set(batch_filter))]
if country_col and country_filter is not None and country_col in fdf.columns:
    fdf = fdf[fdf[country_col].astype(str).map(clean_text).isin(set(country_filter))]
if search_text.strip():
    q = search_text.strip().lower()
    fdf = fdf[fdf[name_col].astype(str).str.lower().str.contains(q, na=False)]


# =========================
# Diagnostics / caption
# =========================
st.caption(f"Sheets loaded: {len(dfs)} | Mode: {mode} | Filtered rows: {len(fdf)} / {len(df)}")

if payment_col and not event_dates:
    st.warning("Retention: Payment Date exists but event dates were not parsed/inferred → retention falls back to “any participation”.")


# =========================
# KPIs (compact + clean)
# =========================
compact_kpis(df, fdf, primary_ctx)
st.write("")


# =========================
# Downloads
# =========================
c_dl1, c_dl2 = st.columns([1, 1])
with c_dl1:
    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=csv,
        file_name=("compare_filtered.csv" if len(selected_sheets) > 1 else f"{selected_sheets[0]}_filtered.csv"),
        mime="text/csv",
        use_container_width=True,
    )

with c_dl2:
    st.download_button(
        "⬇️ Download event summary (CSV)",
        data=pd.DataFrame({
            "Event": event_cols,
            "Date (if available)": [event_dates.get(e, pd.NaT) for e in event_cols],
            "Participants (filtered)": [int(fdf[e].sum()) if e in fdf.columns else 0 for e in event_cols]
        }).to_csv(index=False).encode("utf-8"),
        file_name="event_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()


# =========================
# Tabs
# =========================
tab_overview, tab_students, tab_conversion, tab_events, tab_timeline, tab_quality = st.tabs(
    ["🌿 Overview", "👥 Students", "💳 Conversion & Retention", "🎯 Events", "🗓️ Timeline", "🧪 Data Quality"]
)


# =========================
# OVERVIEW
# =========================
with tab_overview:
    st.subheader("Participation distribution & segmentation")

    if len(fdf) == 0:
        st.info("No data in current filters.")
    else:
        dist = fdf["participation_count"].value_counts().sort_index().reset_index()
        dist.columns = ["Participation Count", "Students"]

        fig_dist = px.bar(dist, x="Participation Count", y="Students", title="Participation Count Distribution")
        fig_dist = plotly_green_layout(fig_dist, height=360)
        st.plotly_chart(fig_dist, use_container_width=True)

        seg = fdf.copy()
        seg["segment"] = pd.cut(
            seg["participation_count"],
            bins=[-1, 0, 2, 5, 999999],
            labels=["Dormant (0)", "Warm (1–2)", "Engaged (3–5)", "Super Engaged (6+)"],
        )
        seg_summary = seg.groupby(["segment", "conversion_category"], as_index=False).size().rename(columns={"size": "Students"})

        fig_seg = px.bar(
            seg_summary,
            x="segment",
            y="Students",
            color="conversion_category",
            color_discrete_map=GREEN_PALETTE,
            barmode="stack",
            title="Segments vs Conversion (stacked)"
        )
        fig_seg = plotly_green_layout(fig_seg, height=380, x_tickangle=-15)
        st.plotly_chart(fig_seg, use_container_width=True)

    st.subheader("Cohort performance")
    # Prefer Batch, else Program, else Country
    group_col = None
    for cand in [batch_col, program_col, country_col]:
        if cand and cand in fdf.columns:
            group_col = cand
            break

    if group_col and len(fdf) > 0:
        tmp = fdf.copy()
        tmp[group_col] = tmp[group_col].astype(str).map(clean_text)
        grp = tmp.groupby(group_col, as_index=False).agg(
            Students=(name_col, "count"),
            Active=("participation_count", lambda s: int((s > 0).sum())),
            Avg_Participation=("participation_count", "mean"),
            Paid_Rate=("conversion_category", lambda s: float((s == "Paid / Admitted").mean() * 100))
        )
        grp["Active Rate %"] = (grp["Active"] / grp["Students"] * 100).round(1)
        grp["Avg_Participation"] = grp["Avg_Participation"].round(2)
        grp["Paid_Rate"] = grp["Paid_Rate"].round(1)
        grp = grp.sort_values(["Paid_Rate", "Active Rate %", "Students"], ascending=False)

        st.dataframe(grp, use_container_width=True, height=340)

        fig_paid = px.bar(grp.head(25), x=group_col, y="Paid_Rate", title=f"Paid Rate % by {group_col} (Top 25)")
        fig_paid.update_traces(marker_color="#0b3d2e")
        fig_paid = plotly_green_layout(fig_paid, height=360, x_tickangle=-30, bottom_margin=120)
        st.plotly_chart(fig_paid, use_container_width=True)
    else:
        st.info("No cohort column available for grouping.")


# =========================
# STUDENTS
# =========================
with tab_students:
    st.subheader("Top participating students (Filtered)")
    cols_show = [name_col, "participation_count", "conversion_category", "lead_score"]
    for extra in [program_col, batch_col, "__sheet__"]:
        if extra and extra in fdf.columns and extra not in cols_show:
            cols_show.append(extra)

    st.dataframe(
        fdf.sort_values("participation_count", ascending=False)[cols_show].head(250),
        use_container_width=True,
        height=420,
    )

    st.write("")
    cA, cB = st.columns(2)

    with cA:
        st.markdown("#### 🎯 High engagement but not paid (best conversion targets)")
        tcols = [name_col, "participation_count", "lead_score", "conversion_category"]
        for extra in [program_col, batch_col, "__sheet__"]:
            if extra and extra in fdf.columns and extra not in tcols:
                tcols.append(extra)

        targets = (
            fdf[(fdf["conversion_category"] != "Paid / Admitted") & (fdf["participation_count"] >= 3)]
            .sort_values(["participation_count", "lead_score"], ascending=False)[tcols]
            .head(25)
        )
        st.dataframe(targets, use_container_width=True, height=360)

    with cB:
        st.markdown("#### 🧯 Paid but low engagement (retention risk)")
        rcols = [name_col, "participation_count", "lead_score"]
        if payment_col:
            rcols.append(payment_col)
        for extra in [program_col, batch_col, "__sheet__"]:
            if extra and extra in fdf.columns and extra not in rcols:
                rcols.append(extra)

        risks = (
            fdf[(fdf["conversion_category"] == "Paid / Admitted") & (fdf["participation_count"] <= 1)]
            .sort_values(["participation_count", "lead_score"], ascending=True)[rcols]
            .head(25)
        )
        st.dataframe(risks, use_container_width=True, height=360)


# =========================
# CONVERSION & RETENTION
# =========================
with tab_conversion:
    st.subheader("Conversion breakdown (Filtered)")

    cols_base = [name_col, conversion_col, "participation_count", "conversion_category"]
    for extra in [program_col, batch_col, "__sheet__"]:
        if extra and extra in fdf.columns and extra not in cols_base:
            cols_base.append(extra)
    if payment_col and payment_col in fdf.columns:
        cols_base.insert(2, payment_col)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Conversion category counts")
        st.dataframe(
            fdf["conversion_category"].value_counts().reset_index().rename(columns={"index": "Category", "conversion_category": "Count"}),
            use_container_width=True,
            height=240,
        )

    with c2:
        st.markdown("### Raw conversion status values")
        raw_conv = fdf[conversion_col].replace({"": np.nan, "nan": np.nan}).dropna()
        raw_conv_counts = raw_conv.value_counts().reset_index()
        raw_conv_counts.columns = ["Conversion Status", "Count"]
        st.dataframe(raw_conv_counts, use_container_width=True, height=240)

    st.divider()

    st.subheader("Retention")
    if payment_col:
        f_ret = float(fdf["retained"].mean() * 100) if len(fdf) else 0.0
        st.markdown(f"**Retention rate (filtered):** {f_ret:.2f}%")

        retained_cols = [name_col, payment_col, "participation_count"]
        for extra in [program_col, batch_col, "__sheet__"]:
            if extra and extra in fdf.columns and extra not in retained_cols:
                retained_cols.append(extra)

        retained = fdf[fdf["retained"] == 1][retained_cols].copy()
        st.markdown("Retained students (paid + attended after payment date)")
        st.dataframe(retained, use_container_width=True, height=280)
    else:
        st.info("Retention not available (no Payment Date column detected).")


# =========================
# EVENTS
# =========================
with tab_events:
    st.subheader("Event-wise participation (Filtered)")

    if not event_cols:
        st.info("No event columns detected after parsing.")
    else:
        event_counts = fdf[event_cols].sum().sort_values(ascending=False)
        event_pct = (event_counts / max(len(fdf), 1) * 100).round(1)

        event_table = pd.DataFrame(
            {
                "Event": event_counts.index,
                "Participants": event_counts.values.astype(int),
                "Participation %": event_pct.values,
                "Event Date (if known)": [event_dates.get(ev, pd.NaT) for ev in event_counts.index],
            }
        )
        st.dataframe(event_table, use_container_width=True, height=360)

        fig_bar = px.bar(
            event_table.head(40),
            x="Event",
            y="Participants",
            hover_data=["Participation %", "Event Date (if known)"],
            title="Top events by participants (Top 40)",
        )
        fig_bar.update_traces(marker_color="#1f7a56")
        fig_bar = plotly_green_layout(fig_bar, height=440, x_tickangle=-45, bottom_margin=160)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    st.subheader("Event calendar (date-sorted)")
    if event_dates:
        cal = []
        for ev in event_cols:
            dt = event_dates.get(ev, pd.NaT)
            if pd.notna(dt):
                cal.append({
                    "Event Date": dt,
                    "Event": ev,
                    "Participants (filtered)": int(fdf[ev].sum()) if ev in fdf.columns else 0
                })
        cal_df = pd.DataFrame(cal).sort_values("Event Date") if cal else pd.DataFrame()
        if cal_df.empty:
            st.info("Event dates not available for detected event columns.")
        else:
            st.dataframe(cal_df, use_container_width=True, height=360)
    else:
        st.info("No event dates detected/inferred.")


# =========================
# TIMELINE
# =========================
with tab_timeline:
    st.subheader("Participation trend over time")
    if event_dates and event_cols:
        dated_events = [(ev, event_dates.get(ev, pd.NaT)) for ev in event_cols]
        dated_events = [(ev, dt) for ev, dt in dated_events if pd.notna(dt)]
        if dated_events:
            trend = [(dt, int(fdf[ev].sum())) for ev, dt in dated_events if ev in fdf.columns]
            trend_df = (
                pd.DataFrame(trend, columns=["Event Date", "Participants"])
                .groupby("Event Date", as_index=False).sum()
                .sort_values("Event Date")
            )

            fig_tr = px.line(trend_df, x="Event Date", y="Participants", markers=True, title="Participants over time")
            fig_tr.update_traces(line_color="#0b3d2e")
            rng = safe_datetime_range_with_padding(trend_df["Event Date"].tolist(), pad_days=3)
            if rng:
                fig_tr.update_xaxes(range=rng)
            fig_tr = plotly_green_layout(fig_tr, height=380)
            st.plotly_chart(fig_tr, use_container_width=True)
        else:
            st.info("Event dates exist but none mapped to detected event columns.")
    else:
        st.info("No event dates detected in this view.")


# =========================
# DATA QUALITY / DIAGNOSTICS
# =========================
with tab_quality:
    st.subheader("Parsing diagnostics (per sheet)")

    diag_rows = []
    for sname, ctx_i in contexts:
        diag_rows.append({
            "Sheet": sname,
            "Program inferred": infer_program_from_sheet(sname),
            "Batch inferred": infer_batch_from_sheet_name(sname),
            "Header row": ctx_i["meta"]["header_row"],
            "Date row": ctx_i["meta"]["date_row"],
            "Event row": ctx_i["meta"]["event_row"],
            "Detected events (sheet)": len(ctx_i["event_cols"]),
            "Events (union)": len(event_cols),
            "Has payment col": bool(ctx_i["payment_col"]),
            "Has batch col": bool(ctx_i["batch_col"] or ("Batch" in df.columns)),
        })
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, height=320)

    st.write("")
    st.subheader("Potential issues")
    issues = []
    if len(fdf) == 0:
        issues.append("Filtered dataset is empty (filters too strict or sheet has no rows).")
    if not event_cols:
        issues.append("No event columns detected (attendance columns might not be yes/no-style).")
    if payment_col and not event_dates:
        issues.append("Payment exists but event dates missing → retention may be less accurate.")
    if issues:
        for it in issues:
            st.warning(it)
    else:
        st.success("No obvious issues detected.")


# =========================
# Footer
# =========================
st.markdown(
    """
    <div style="margin-top:14px; color:#4e7f6a; font-weight:600;">
      Tip: Use <b>Program</b> and <b>Sheet</b> filters to compare UG vs PG and the Tetr-X paid sheets.
    </div>
    """,
    unsafe_allow_html=True,
)
