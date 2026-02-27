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
        /* App background */
        .stApp { background: #ffffff; }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #f6fbf7;
            border-right: 1px solid #dbeee0;
        }

        /* Titles */
        h1, h2, h3, h4, h5, h6 { color: #0b3d2e !important; }

        /* Accent */
        .tetr-accent { color: #0b3d2e; font-weight: 700; }

        /* Card */
        .tetr-card {
            background: #ffffff;
            border: 1px solid #dbeee0;
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 1px 8px rgba(11, 61, 46, 0.06);
        }

        /* Subtle badge */
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

        /* Metric row container */
        .tetr-metrics {
            display: grid;
            grid-template-columns: repeat(5, minmax(0, 1fr));
            gap: 10px;
        }

        /* Metric box */
        .tetr-metric {
            background: #ffffff;
            border: 1px solid #dbeee0;
            border-radius: 14px;
            padding: 10px 12px;
            box-shadow: 0 1px 8px rgba(11, 61, 46, 0.06);
        }
        .tetr-metric .label {
            color: #2e6b57;
            font-weight: 700;
            font-size: 12px;
            margin-bottom: 4px;
        }
        .tetr-metric .value {
            color: #0b3d2e;
            font-weight: 800;
            font-size: 22px;
            line-height: 1.1;
        }
        .tetr-metric .sub {
            color: #4e7f6a;
            font-weight: 600;
            font-size: 12px;
            margin-top: 4px;
        }

        /* Buttons */
        .stDownloadButton button, .stButton button {
            background: #0b3d2e !important;
            border: 1px solid #0b3d2e !important;
            color: #ffffff !important;
            border-radius: 10px !important;
        }
        .stDownloadButton button:hover, .stButton button:hover {
            background: #0f5a43 !important;
            border-color: #0f5a43 !important;
        }

        /* Inputs */
        .stTextInput input, .stSelectbox div, .stMultiSelect div {
            border-radius: 10px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()

# =========================
# Constants
# =========================
# Only these sheets should be analyzed, and only these get batch alignment.
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

GREEN_PALETTE = {
    "Paid / Admitted": "#0b3d2e",  # dark green
    "Will Pay": "#1f7a56",         # green
    "Pending": "#6bbf8a",          # light green
    "Refunded": "#93d8b1",         # pale green
    "Not Paid": "#cfeedd",         # very light green
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

    # numeric dd ? mm ? yyyy anywhere
    m = re.search(r"(\d{1,2})\D+(\d{1,2})\D+(\d{4})", s)
    if m:
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        try:
            return pd.Timestamp(int(yyyy), int(mm), int(dd))
        except Exception:
            return pd.NaT

    # Month-name formats: 28 Jan 2026
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
    # More robust for sparse attendance columns
    s = series.fillna("").astype(str).str.strip().str.lower()
    if s.empty:
        return False
    allowed = {"yes", "no", "y", "n", "true", "false", "1", "0", "attended", "present", "absent", ""}
    return (s.isin(allowed)).mean() >= 0.6


def safe_datetime_range_with_padding(dates, pad_days=3):
    dates = [d for d in dates if pd.notna(d)]
    if not dates:
        return None
    mn = min(dates)
    mx = max(dates)
    mn = mn - pd.Timedelta(days=pad_days)
    mx = mx + pd.Timedelta(days=pad_days)
    return [mn, mx]


def infer_batch_from_sheet_name(sheet_name: str) -> str:
    """
    Derive batch label from sheet name:
      - 'UG B5' -> 'B5'
      - 'PG - B3 & B4' -> 'B3–B4'
      - 'UG - B1 to B4' -> 'B1–B4'
    """
    s = clean_text(sheet_name).lower()
    s = s.replace("–", "-").replace("—", "-")

    # b1 to b4 / b1 - b4
    m = re.search(r"\bb\s*(\d+)\s*(to|-)\s*b\s*(\d+)\b", s)
    if m:
        return f"B{m.group(1)}–B{m.group(3)}"

    # b3 & b4
    m = re.search(r"\bb\s*(\d+)\s*&\s*b\s*(\d+)\b", s)
    if m:
        return f"B{m.group(1)}–B{m.group(2)}"

    # single b#
    m = re.search(r"\bb\s*(\d+)\b", s)
    if m:
        return f"B{m.group(1)}"

    # Tetr-X sheets might not encode batch
    return ""


# =========================
# Sheet-aware loader (robust)
# =========================
def load_sheet_structured(raw: pd.DataFrame):
    """
    Supports:
      - old: header row with Student Name(s)
      - new: header row with Name + Email/Batch/Country/Payment etc
    Date row above header is optional.
    """
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

    # optional date row above
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
# Excel I/O (fast + safe)
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
# Build dataset (single sheet)
# =========================
def build_dataset_from_sheet(file_bytes: bytes, sheet_name: str):
    raw = read_raw_sheet(file_bytes, sheet_name)
    df, meta, err = load_sheet_structured(raw)
    if err:
        return None, None, err

    # Keyword maps
    KW = {
        "name_strict": ["student name", "student names", "full name", "learner name", "student", "name"],
        "name_fallback": ["name"],
        "email": ["email", "e mail", "e-mail"],
        "phone": ["phone", "mobile", "mobile no", "phone number", "contact"],
        "country": ["country"],
        "batch": ["batch", "cohort", "group"],
        "conversion": ["conversion status", "conversion", "status", "admitted", "refunded", "pending", "will pay"],
        "payment_date": ["payment date", "payment", "paid date", "date of payment"],
        "engagement_score": ["overall engagement score", "overall engagement", "engagement score", "engagement"],
        "community_status": ["community status"],
        "date_of_exit": ["date of exit", "exit date", "community exit date"],
    }

    name_col = best_matching_col(df, KW["name_strict"])
    if not name_col:
        name_col = best_matching_col(
            df, KW["name_fallback"],
            hard_excludes=["batch", "program", "session", "event", "country", "status"]
        )
    if not name_col:
        return None, None, "Student/Name column not detected."

    email_col = best_matching_col(df, KW["email"])
    phone_col = best_matching_col(df, KW["phone"])
    country_col = best_matching_col(df, KW["country"])
    batch_col = best_matching_col(df, KW["batch"])
    conversion_col = best_matching_col(df, KW["conversion"])
    payment_col = best_matching_col(df, KW["payment_date"])
    engagement_col = best_matching_col(df, KW["engagement_score"])
    community_col = best_matching_col(df, KW["community_status"])
    exit_col = best_matching_col(df, KW["date_of_exit"])

    # Batch alignment ONLY for allowed sheets
    if sheet_name in ALLOWED_SHEETS:
        inferred = infer_batch_from_sheet_name(sheet_name)
        if inferred:
            # If batch col missing/empty, set it
            if not batch_col:
                df["Batch"] = inferred
                batch_col = "Batch"
            else:
                # Fill missing batch values with inferred, without overwriting existing
                bc = df[batch_col].astype(str).map(clean_text)
                df.loc[bc.eq(""), batch_col] = inferred

    # Drop numeric-only summary rows only when name is blank
    name_series = df[name_col].astype(str).map(clean_text)
    numeric_mask = df.apply(row_is_numeric_only, axis=1)
    blank_name_mask = name_series.map(lambda x: x == "" or x.lower() == "nan")
    df = df.loc[~(numeric_mask & blank_name_mask)].copy()

    # Event columns = everything except metadata
    metadata_cols = {
        c for c in [
            name_col, email_col, phone_col, country_col, batch_col,
            conversion_col, payment_col, engagement_col, community_col, exit_col
        ] if c
    }

    # Candidate event cols are non-metadata
    event_cols = [c for c in df.columns if c not in metadata_cols]
    # Keep only those that look like yes/no attendance
    event_cols = [c for c in event_cols if is_probably_event_col(df[c])]

    for c in event_cols:
        df[c] = df[c].apply(normalize_binary)

    df["participation_count"] = df[event_cols].sum(axis=1) if event_cols else 0

    if payment_col:
        df[payment_col] = df[payment_col].apply(parse_date_safe)

    if not conversion_col:
        df["Conversion Status"] = ""
        conversion_col = "Conversion Status"
    df[conversion_col] = df[conversion_col].astype(str).map(lambda x: clean_text(x).lower())

    def conv_category(r):
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

    def retained_flag(r):
        if not payment_col:
            return np.nan
        pay = r.get(payment_col, pd.NaT)
        if pd.isna(pay):
            return 0
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
        "conversion_col": conversion_col,
        "payment_col": payment_col,
        "engagement_col": engagement_col,
        "community_col": community_col,
        "exit_col": exit_col,
        "event_cols": event_cols,
        "event_dates": event_dates,
    }
    return df, ctx, None


# =========================
# UI helpers
# =========================
def metric_row(items):
    html = '<div class="tetr-metrics">'
    for label, value, sub in items:
        html += f"""
            <div class="tetr-metric">
                <div class="label">{label}</div>
                <div class="value">{value}</div>
                <div class="sub">{sub}</div>
            </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def kpi_cards(df, ctx, fdf):
    name_col = ctx["name_col"]
    payment_col = ctx["payment_col"]

    total_students = int(df[name_col].notna().sum())
    active_students = int((df["participation_count"] > 0).sum())
    paid_count = int((df["conversion_category"] == "Paid / Admitted").sum())
    participants = int((df["participation_count"] > 0).sum())
    conversion_rate = (paid_count / participants * 100) if participants else 0.0

    f_total = int(fdf[name_col].notna().sum())
    f_active = int((fdf["participation_count"] > 0).sum())
    f_paid = int((fdf["conversion_category"] == "Paid / Admitted").sum())
    f_participants = int((fdf["participation_count"] > 0).sum())
    f_conv_rate = (f_paid / f_participants * 100) if f_participants else 0.0

    if payment_col:
        retention_all = float(df["retained"].mean() * 100) if len(df) else 0.0
        retention_f = float(fdf["retained"].mean() * 100) if len(fdf) else 0.0
    else:
        retention_all, retention_f = None, None

    metric_row([
        ("Total Students", f"{f_total}", f"All: {total_students}"),
        ("Active Students", f"{f_active}", f"All: {active_students}"),
        ("Paid / Admitted", f"{f_paid}", f"All: {paid_count}"),
        ("Conversion Rate", f"{f_conv_rate:.1f}%", f"All: {conversion_rate:.1f}%"),
        ("Retention Rate", f"{retention_f:.1f}%" if payment_col else "N/A",
         f"All: {retention_all:.1f}%" if payment_col else "Needs Payment"),
    ])


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
            Only analyzes approved batch sheets • Batch-aligned parsing • Conversion + retention + event intelligence
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

# Only allow the 8 batch sheets
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
mode = st.sidebar.radio(
    "Choose view",
    ["Single sheet", "Compare batches (multi-sheet)"],
    index=1
)
st.sidebar.markdown("---")

if mode == "Single sheet":
    selected_sheet = st.sidebar.selectbox("Select Sheet", sheets)
    selected_sheets = [selected_sheet]
else:
    selected_sheets = st.sidebar.multiselect(
        "Select batch sheets to compare",
        options=sheets,
        default=sheets,  # all allowed by default
    )

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
# Align event columns + event dates across selected sheets (important!)
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

# Ensure every df has every event column; recompute participation_count consistently
for i in range(len(dfs)):
    for ev in all_event_cols:
        if ev not in dfs[i].columns:
            dfs[i][ev] = 0
        dfs[i][ev] = dfs[i][ev].fillna(0).astype(int)
    dfs[i]["participation_count"] = dfs[i][all_event_cols].sum(axis=1) if all_event_cols else 0

# Aggregate
df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

# Use first context as "primary" for naming; events use union
primary_ctx = contexts[0][1]
name_col = primary_ctx["name_col"]
batch_col = primary_ctx["batch_col"] or ("Batch" if "Batch" in df.columns else None)
country_col = primary_ctx["country_col"]
conversion_col = primary_ctx["conversion_col"]
payment_col = primary_ctx["payment_col"]
event_cols = all_event_cols
event_dates = all_event_dates
meta = primary_ctx["meta"]

# =========================
# Sidebar filters
# =========================
st.sidebar.markdown("## Filters")

all_conv_vals = ["Paid / Admitted", "Will Pay", "Pending", "Refunded", "Not Paid"]
present_conv = [c for c in all_conv_vals if c in set(df["conversion_category"].unique())]
conv_filter = st.sidebar.multiselect("Conversion category", present_conv, default=present_conv)

min_part = st.sidebar.slider(
    "Minimum participation count",
    0,
    int(df["participation_count"].max() or 0),
    0
)

# Sheet filter (useful in compare)
sheet_filter = None
if "__sheet__" in df.columns and len(selected_sheets) > 1:
    st.sidebar.markdown("### Compare control")
    all_sheet_vals = sorted(df["__sheet__"].dropna().astype(str).unique().tolist())
    sheet_filter = st.sidebar.multiselect("Limit to sheet(s)", all_sheet_vals, default=all_sheet_vals)

batch_filter = None
if batch_col and batch_col in df.columns:
    batches = sorted([b for b in df[batch_col].dropna().astype(str).map(clean_text).unique().tolist() if b])
    if batches:
        batch_filter = st.sidebar.multiselect("Batch", batches, default=batches)

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
st.caption(
    f"Sheets loaded: {len(dfs)} | "
    f"Mode: {mode} | "
    f"Filtered rows: {len(fdf)} / {len(df)}"
)

if payment_col and not event_dates:
    st.warning("Retention: Payment Date exists but event dates were not parsed/inferred → retention falls back to “any participation”.")

# =========================
# KPIs
# =========================
kpi_cards(df, primary_ctx, fdf)
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
        fig_dist = plotly_green_layout(fig_dist, height=380)
        st.plotly_chart(fig_dist, use_container_width=True)

        seg = fdf.copy()
        seg["segment"] = pd.cut(
            seg["participation_count"],
            bins=[-1, 0, 2, 5, 999999],
            labels=["Dormant (0)", "Warm (1–2)", "Engaged (3–5)", "Super Engaged (6+)"],
        )
        seg_summary = seg.groupby(["segment", "conversion_category"], as_index=False).size()
        seg_summary = seg_summary.rename(columns={"size": "Students"})

        fig_seg = px.bar(
            seg_summary,
            x="segment",
            y="Students",
            color="conversion_category",
            color_discrete_map=GREEN_PALETTE,
            barmode="stack",
            title="Segments vs Conversion (stacked)"
        )
        fig_seg = plotly_green_layout(fig_seg, height=420, x_tickangle=-15)
        st.plotly_chart(fig_seg, use_container_width=True)

    st.subheader("Cohort performance (Batch/Country)")

    group_col = batch_col if (batch_col and batch_col in fdf.columns) else (country_col if (country_col and country_col in fdf.columns) else None)
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

        st.dataframe(grp, use_container_width=True, height=360)

        fig_paid = px.bar(grp.head(25), x=group_col, y="Paid_Rate", title=f"Paid Rate % by {group_col} (Top 25)")
        fig_paid.update_traces(marker_color="#0b3d2e")
        fig_paid = plotly_green_layout(fig_paid, height=380, x_tickangle=-30, bottom_margin=120)
        st.plotly_chart(fig_paid, use_container_width=True)
    else:
        st.info("No Batch/Country column detected to show cohort performance.")

# =========================
# STUDENTS
# =========================
with tab_students:
    st.subheader("Top participating students (Filtered)")
    cols_show = [name_col, "participation_count", "conversion_category", "lead_score"]
    if "__sheet__" in fdf.columns:
        cols_show.append("__sheet__")
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
        if "__sheet__" in fdf.columns:
            tcols.append("__sheet__")
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
        if "__sheet__" in fdf.columns:
            rcols.append("__sheet__")
        risks = (
            fdf[(fdf["conversion_category"] == "Paid / Admitted") & (fdf["participation_count"] <= 1)]
            .sort_values(["participation_count", "lead_score"], ascending=True)[rcols]
            .head(25)
        )
        st.dataframe(risks, use_container_width=True, height=360)

    st.write("")
    st.subheader("Students with NO participation (Filtered)")
    cols_no = [name_col, conversion_col]
    if payment_col:
        cols_no.append(payment_col)
    if "__sheet__" in fdf.columns:
        cols_no.append("__sheet__")
    st.dataframe(
        fdf[fdf["participation_count"] == 0][cols_no],
        use_container_width=True,
        height=320,
    )

    st.write("")
    st.subheader("🏆 Lead score leaderboard (Filtered)")
    cols_lead = [name_col, "lead_score", "participation_count", "conversion_category"]
    if "__sheet__" in fdf.columns:
        cols_lead.append("__sheet__")
    st.dataframe(
        fdf.sort_values("lead_score", ascending=False)[cols_lead].head(250),
        use_container_width=True,
        height=420,
    )

# =========================
# CONVERSION & RETENTION
# =========================
with tab_conversion:
    st.subheader("Conversion breakdown (Filtered)")

    paid_df = fdf[fdf["conversion_category"] == "Paid / Admitted"].copy()
    will_df = fdf[fdf["conversion_category"] == "Will Pay"].copy()
    pending_df = fdf[fdf["conversion_category"] == "Pending"].copy()
    refunded_df = fdf[fdf["conversion_category"] == "Refunded"].copy()
    not_df = fdf[fdf["conversion_category"] == "Not Paid"].copy()

    cols_base = [name_col, conversion_col, "participation_count"]
    if payment_col:
        cols_base.insert(2, payment_col)
    if "__sheet__" in fdf.columns:
        cols_base.append("__sheet__")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### ✅ Paid / Admitted")
        st.dataframe(paid_df[cols_base], use_container_width=True, height=260)
    with c2:
        st.markdown("### 🟡 Will Pay")
        st.dataframe(will_df[cols_base], use_container_width=True, height=260)
    with c3:
        st.markdown("### 🟠 Pending")
        st.dataframe(pending_df[cols_base], use_container_width=True, height=260)

    c4, c5 = st.columns(2)
    with c4:
        st.markdown("### 🔵 Refunded")
        st.dataframe(refunded_df[cols_base], use_container_width=True, height=240)
    with c5:
        st.markdown("### 🔴 Not Paid")
        st.dataframe(not_df[cols_base], use_container_width=True, height=240)

    st.write("")
    st.markdown("### Raw conversion status values (Filtered)")
    raw_conv = fdf[conversion_col].replace({"": np.nan, "nan": np.nan}).dropna()
    raw_conv_counts = raw_conv.value_counts().reset_index()
    raw_conv_counts.columns = ["Conversion Status", "Count"]
    st.dataframe(raw_conv_counts, use_container_width=True, height=240)

    st.divider()

    st.subheader("Retention")
    if payment_col:
        f_ret = float(fdf["retained"].mean() * 100) if len(fdf) else 0.0
        st.markdown(f"**Retention rate (filtered):** <span class='tetr-accent'>{f_ret:.2f}%</span>", unsafe_allow_html=True)

        retained_cols = [name_col, payment_col, "participation_count"]
        if "__sheet__" in fdf.columns:
            retained_cols.append("__sheet__")
        retained = fdf[fdf["retained"] == 1][retained_cols].copy()
        st.markdown("Retained students (paid + attended after payment date)")
        st.dataframe(retained, use_container_width=True, height=280)

        group_col = batch_col if (batch_col and batch_col in fdf.columns) else (country_col if (country_col and country_col in fdf.columns) else None)
        if group_col:
            tmp = fdf.copy()
            tmp[group_col] = tmp[group_col].astype(str).map(clean_text)
            coh = tmp.groupby(group_col, as_index=False).agg(
                Students=(name_col, "count"),
                Paid=("conversion_category", lambda s: int((s == "Paid / Admitted").sum())),
                Retained=("retained", "sum"),
            )
            coh["Retention % (Paid only)"] = np.where(
                coh["Paid"] > 0,
                (coh["Retained"] / coh["Paid"] * 100).round(1),
                np.nan
            )
            coh = coh.sort_values("Retention % (Paid only)", ascending=False)

            st.markdown(f"### Retention by {group_col}")
            st.dataframe(coh, use_container_width=True, height=320)
    else:
        st.info("Retention not available (no Payment Date column detected).")

    st.divider()

    st.subheader("Engagement Before vs After Payment Date (Filtered)")
    if payment_col and event_dates and event_cols and len(fdf) > 0:
        rows = []
        for _, r in fdf.iterrows():
            pay = r.get(payment_col, pd.NaT)
            if pd.isna(pay):
                continue
            pay = pay.normalize()
            for ev in event_cols:
                ev_dt = event_dates.get(ev, pd.NaT)
                if pd.isna(ev_dt):
                    continue
                rows.append(
                    {
                        "Student": str(r.get(name_col, "")),
                        "Payment Date": pay,
                        "Event": ev,
                        "Event Date": ev_dt,
                        "Attended": int(r.get(ev, 0)),
                        "Days From Payment": int((ev_dt - pay).days),
                    }
                )

        long_df = pd.DataFrame(rows)
        if long_df.empty:
            st.info("Not enough data to compute engagement around payment date for this view.")
        else:
            min_d, max_d = int(long_df["Days From Payment"].min()), int(long_df["Days From Payment"].max())
            left = max(-60, min_d)
            right = min(60, max_d)

            window = st.slider(
                "Window (days relative to payment date)",
                min_value=left,
                max_value=right,
                value=(-14, 30),
            )

            wdf = long_df[(long_df["Days From Payment"] >= window[0]) & (long_df["Days From Payment"] <= window[1])]
            agg = (
                wdf.groupby("Days From Payment", as_index=False)
                .agg(Attended=("Attended", "sum"), Events=("Attended", "count"))
            )
            agg["Attendance Rate"] = (agg["Attended"] / agg["Events"]).fillna(0.0)

            fig_pay = px.line(
                agg,
                x="Days From Payment",
                y="Attendance Rate",
                markers=True,
                hover_data=["Attended", "Events"],
                title="Attendance Rate vs Days From Payment Date",
            )
            fig_pay.add_vline(x=0, line_dash="dash", line_color="#0b3d2e")
            fig_pay.update_traces(line_color="#0b3d2e")
            fig_pay = plotly_green_layout(fig_pay, height=420)
            st.plotly_chart(fig_pay, use_container_width=True)
            st.caption("Day 0 = payment date. Positive days = events after payment.")
    else:
        st.info("Needs both Payment Date and Event Dates for this view.")

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

    st.subheader("Pareto (80/20) — do a few events drive most participation?")
    if event_cols and len(fdf) > 0:
        event_counts = fdf[event_cols].sum().sort_values(ascending=False)
        if event_counts.sum() == 0:
            st.info("No event participation in current filtered view.")
        else:
            pareto = pd.DataFrame({"Event": event_counts.index, "Participants": event_counts.values.astype(int)})
            pareto["Cumulative %"] = (pareto["Participants"].cumsum() / pareto["Participants"].sum() * 100).round(1)

            fig_par = go.Figure()
            fig_par.add_trace(go.Bar(x=pareto["Event"], y=pareto["Participants"], name="Participants",
                                     marker_color="#0f5a43"))
            fig_par.add_trace(go.Scatter(x=pareto["Event"], y=pareto["Cumulative %"], yaxis="y2",
                                         name="Cumulative %", line=dict(color="#0b3d2e")))
            fig_par.update_layout(
                title="Event counts + cumulative share (Pareto)",
                xaxis_tickangle=-45,
                height=480,
                margin=dict(l=10, r=10, t=60, b=170),
                yaxis=dict(title="Participants"),
                yaxis2=dict(title="Cumulative %", overlaying="y", side="right", range=[0, 100]),
                showlegend=True,
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="#0b3d2e"),
            )
            fig_par.update_xaxes(showgrid=True, gridcolor="#e8f4ec")
            fig_par.update_yaxes(showgrid=True, gridcolor="#e8f4ec")
            st.plotly_chart(fig_par, use_container_width=True)
    else:
        st.info("Pareto needs event columns.")

    st.divider()

    st.subheader("Event conversion impact (Paid rate among attendees)")
    if event_cols and len(fdf) > 0:
        rows = []
        for ev in event_cols:
            attendees = fdf[fdf[ev] == 1]
            n_att = len(attendees)
            if n_att == 0:
                continue
            paid_rate = (attendees["conversion_category"].eq("Paid / Admitted").mean()) * 100
            rows.append({"Event": ev, "Attendees": n_att, "Paid Rate % (Attendees)": round(paid_rate, 1)})

        impact = pd.DataFrame(rows)
        if impact.empty:
            st.info("No attendee data found for events in this view.")
        else:
            impact = impact.sort_values(["Paid Rate % (Attendees)", "Attendees"], ascending=False)
            min_att = st.slider(
                "Minimum attendees to include an event",
                1,
                max(1, int(impact["Attendees"].max())),
                min(10, max(1, int(impact["Attendees"].max()))),
            )
            impact_f = impact[impact["Attendees"] >= min_att]
            if impact_f.empty:
                st.info("No events meet the minimum attendee threshold.")
            else:
                fig_imp = px.bar(
                    impact_f.head(40),
                    x="Event",
                    y="Paid Rate % (Attendees)",
                    hover_data=["Attendees"],
                    title="Paid/Admitted rate among attendees (higher = stronger conversion signal)",
                )
                fig_imp.update_traces(marker_color="#0b3d2e")
                fig_imp = plotly_green_layout(fig_imp, height=440, x_tickangle=-45, bottom_margin=160)
                st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("No event columns detected.")

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

    st.divider()

    st.subheader("Per-student participation timeline (Filtered)")
    students = fdf[name_col].dropna().astype(str).unique().tolist()
    if not students:
        st.info("No students detected in this filtered view.")
    else:
        selected_student = st.selectbox("Select Student", students)
        row = fdf[fdf[name_col].astype(str) == str(selected_student)].iloc[0]

        use_dates = bool(event_dates)
        timeline_events = event_cols[:]

        if use_dates:
            timeline_events = sorted(
                timeline_events,
                key=lambda ev: (pd.isna(event_dates.get(ev, pd.NaT)), event_dates.get(ev, pd.NaT)),
            )

        x_vals, attended = [], []
        event_dt_list = []
        for i, ev in enumerate(timeline_events, start=1):
            if use_dates and pd.notna(event_dates.get(ev, pd.NaT)):
                dt = event_dates[ev]
                x_vals.append(dt)
                event_dt_list.append(dt)
            else:
                x_vals.append(i)
            attended.append(int(row.get(ev, 0)))

        attended_x, attended_y = [], []
        missed_x, missed_y = [], []
        for xv, a in zip(x_vals, attended):
            if a == 1:
                attended_x.append(xv)
                attended_y.append(1)
            else:
                missed_x.append(xv)
                missed_y.append(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=[0.5] * len(x_vals),
            mode="lines",
            hoverinfo="skip",
            showlegend=False,
            line=dict(width=1, dash="dot", color="#6bbf8a"),
            opacity=0.35,
        ))

        fig.add_trace(go.Scatter(
            x=attended_x, y=attended_y,
            mode="lines+markers",
            name="Attended",
            connectgaps=False,
            line=dict(color="#0b3d2e"),
            marker=dict(color="#0b3d2e", size=8)
        ))

        fig.add_trace(go.Scatter(
            x=missed_x, y=missed_y,
            mode="markers",
            name="Missed",
            marker=dict(symbol="x", size=10, color="#1f7a56")
        ))

        if payment_col and pd.notna(row.get(payment_col, pd.NaT)) and use_dates:
            pay_dt = row[payment_col]
            fig.add_trace(go.Scatter(
                x=[pay_dt], y=[1.18], mode="markers+text",
                text=["<b>✔ Payment</b>"],
                textposition="top center",
                name="Payment Date",
                marker=dict(symbol="star", size=18, color="#0f5a43"),
                textfont=dict(size=14, color="#0f5a43"),
            ))

        fig.update_yaxes(range=[-0.2, 1.3], tickvals=[0, 1], title="Participation (1=Attended, 0=Missed)")
        fig.update_layout(
            height=560,
            margin=dict(l=10, r=10, t=60, b=80),
            title=f"Timeline — {selected_student}",
            xaxis_title="Event Date" if use_dates else "Event Sequence",
            showlegend=True,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="#0b3d2e"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#e8f4ec")
        fig.update_yaxes(showgrid=True, gridcolor="#e8f4ec")

        if use_dates and event_dt_list:
            rng = safe_datetime_range_with_padding(event_dt_list, pad_days=4)
            if rng:
                fig.update_xaxes(range=rng, autorange=False)

        if not use_dates:
            fig.update_xaxes(
                tickmode="array",
                tickvals=list(range(1, len(timeline_events) + 1)),
                ticktext=timeline_events,
                tickangle=-45
            )

        st.plotly_chart(fig, use_container_width=True)

# =========================
# DATA QUALITY / DIAGNOSTICS
# =========================
with tab_quality:
    st.subheader("Parsing diagnostics")

    diag_rows = []
    for sname, ctx_i in contexts:
        diag_rows.append({
            "Sheet": sname,
            "Header row": ctx_i["meta"]["header_row"],
            "Date row": ctx_i["meta"]["date_row"],
            "Event row": ctx_i["meta"]["event_row"],
            "Detected events (sheet)": len(ctx_i["event_cols"]),
            "Events (union)": len(event_cols),
            "Has payment col": bool(ctx_i["payment_col"]),
            "Has batch col": bool(ctx_i["batch_col"] or ("Batch" in df.columns)),
        })
    st.dataframe(pd.DataFrame(diag_rows), use_container_width=True, height=260)

    st.write("")
    st.subheader("Column mapping (primary view)")
    st.write({
        "name_col": primary_ctx["name_col"],
        "email_col": primary_ctx["email_col"],
        "phone_col": primary_ctx["phone_col"],
        "country_col": primary_ctx["country_col"],
        "batch_col": batch_col,
        "conversion_col": primary_ctx["conversion_col"],
        "payment_col": primary_ctx["payment_col"],
        "engagement_col": primary_ctx["engagement_col"],
        "community_col": primary_ctx["community_col"],
        "exit_col": primary_ctx["exit_col"],
    })

    st.write("")
    st.subheader("Event date coverage (union)")
    if event_cols:
        cover = pd.DataFrame({
            "Event": event_cols,
            "Has date?": [pd.notna(event_dates.get(e, pd.NaT)) for e in event_cols],
            "Parsed date": [event_dates.get(e, pd.NaT) for e in event_cols],
        })
        st.dataframe(cover, use_container_width=True, height=360)
    else:
        st.info("No event columns detected.")

    st.write("")
    st.subheader("Potential issues")
    issues = []
    if len(fdf) == 0:
        issues.append("Filtered dataset is empty (filters too strict or sheet has no rows).")
    if not event_cols:
        issues.append("No event columns detected (attendance columns might not be yes/no-style).")
    if payment_col and not event_dates:
        issues.append("Payment exists but event dates missing → retention may be less accurate.")
    if not batch_col and mode != "Single sheet":
        issues.append("Batch column not detected in primary context; cohort analyses may fallback to Country.")
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
      Tip: In <b>Compare batches</b> mode you can load multiple batch sheets and use the <b>sheet filter</b> in the sidebar
      to quickly compare them.
    </div>
    """,
    unsafe_allow_html=True,
)
