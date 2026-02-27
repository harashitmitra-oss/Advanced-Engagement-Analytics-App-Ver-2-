# ============================================================
# 🌿 Advanced Engagement + Competition Analytics Dashboard
# ✅ Keeps ALL previous features
# ✅ Fixes compare-mode dtype errors (safe union of event columns)
# ✅ Dropdown filters (selectboxes)
# ✅ Adds requested features:
#    - Total Attendees (BOFU) fixed (default 600, editable)
#    - Unique Attendees = students who attend ONLY ONE event category/type
#      (AMA-only, Masterclass-only, Competition-only, Online-only, General-only, Other-only)
#    - Avg attendees per session (UG vs PG)
#    - Top performing AMAs
#    - Attendance vs Country correlation + heatmap
#    - Competition performance: participation %, paid&participated %, paid&win/spotlight %
#    - Overall competitions: UG vs PG, batchwise UG/PG, type-wise (CEO/Hackathon/Video/Photo/AI/etc)
# ✅ Adds extra useful features:
#    - Multi-sheet compare mode + sheet dropdown
#    - Event calendar + dated trend
#    - Conversion funnel buckets
#    - Cohort (Batch/Country) performance tables
#    - Data quality diagnostics
#
# Requirements: streamlit pandas numpy openpyxl plotly
# Run: streamlit run Analytics_App.py
# ============================================================

import re
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Engagement Analytics Dashboard", layout="wide")


# =========================
# Theme / Styling (Light: dark green + light green + white)
# =========================
DARK_GREEN = "#0b3d2e"
GREEN = "#1f7a56"
LIGHT_GREEN = "#6bbf8a"
PALE_GREEN = "#cfeedd"
GRID_GREEN = "#e8f4ec"

GREEN_PALETTE = {
    "Paid / Admitted": DARK_GREEN,
    "Will Pay": GREEN,
    "Pending": LIGHT_GREEN,
    "Refunded": "#93d8b1",
    "Not Paid": PALE_GREEN,
}


def inject_css():
    st.markdown(
        """
        <style>
        .stApp { background: #ffffff; }
        section[data-testid="stSidebar"] {
            background: #f4fbf6;
            border-right: 1px solid #d8eee0;
        }
        h1,h2,h3,h4,h5,h6 { color: #0b3d2e !important; }
        .tetr-card {
            background: #ffffff;
            border: 1px solid #d8eee0;
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 1px 10px rgba(11,61,46,0.06);
        }
        .tetr-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 999px;
            border: 1px solid #bfe3c9;
            background: #eaf7ee;
            color: #0b3d2e;
            font-weight: 800;
            font-size: 12px;
        }
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(6, minmax(0, 1fr));
            gap: 10px;
        }
        .kpi {
            background: #ffffff;
            border: 1px solid #d8eee0;
            border-radius: 16px;
            padding: 10px 12px;
            box-shadow: 0 1px 10px rgba(11,61,46,0.06);
        }
        .kpi .label { color: #2e6b57; font-weight: 900; font-size: 12px; margin-bottom: 4px; }
        .kpi .value { color: #0b3d2e; font-weight: 900; font-size: 22px; line-height: 1.1; }
        .kpi .sub { color: #4e7f6a; font-weight: 800; font-size: 12px; margin-top: 4px; }
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
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


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
      - Excel date / timestamp
      - dd-mm-yyyy embedded in text
      - dd month yyyy formats
      - ranges, etc (takes first match)
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
    s = series.dropna().astype(str).str.strip().str.lower()
    if s.empty:
        return False
    allowed = {"yes", "no", "y", "n", "true", "false", "1", "0", "attended", "present", "absent", ""}
    return (s.isin(allowed)).mean() >= 0.5


def safe_datetime_range_with_padding(dates, pad_days=3):
    dates = [d for d in dates if pd.notna(d)]
    if not dates:
        return None
    mn = min(dates)
    mx = max(dates)
    if mn == mx:
        mn = mn - pd.Timedelta(days=pad_days)
        mx = mx + pd.Timedelta(days=pad_days)
    else:
        mn = mn - pd.Timedelta(days=pad_days)
        mx = mx + pd.Timedelta(days=pad_days)
    return [mn, mx]


def safe_event_sum(df_: pd.DataFrame, col: str) -> int:
    if col not in df_.columns:
        return 0
    s = pd.to_numeric(df_[col], errors="coerce").fillna(0)
    return int(s.sum())


def infer_batch_from_sheet_name(sheet_name: str) -> str:
    s = clean_text(sheet_name).lower()

    m = re.search(r"\bb\s*(\d+)\s*to\s*b\s*(\d+)\b", s)
    if m:
        return f"B{m.group(1)}–B{m.group(2)}"

    m = re.search(r"\bb\s*(\d+)\s*&\s*b\s*(\d+)\b", s)
    if m:
        return f"B{m.group(1)}–B{m.group(2)}"

    m = re.search(r"\bb\s*(\d+)\b", s)
    if m:
        return f"B{m.group(1)}"

    return ""


def infer_program_from_sheet_name(sheet_name: str) -> str:
    s = clean_text(sheet_name).lower()
    if "ug" in s:
        return "UG"
    if "pg" in s:
        return "PG"
    return "Unknown"


def is_ama_event(event_name: str) -> bool:
    return "ama" in clean_text(event_name).lower()


def classify_event_category(event_name: str) -> str:
    """
    Event TYPE categories (edit keywords here if you want stricter matching):
      - AMA, Masterclass, Competition, Online Event, General, Other
    """
    s = clean_text(event_name).lower()

    if "ama" in s:
        return "AMA"

    if "masterclass" in s or "master class" in s or re.search(r"\bmc\b", s):
        return "Masterclass"

    if any(k in s for k in ["competition", "hackathon", "hack", "challenge", "contest", "case comp", "pitch"]):
        return "Competition"

    if any(k in s for k in ["webinar", "online", "zoom", "google meet", "meet link", "live session", "session", "workshop"]):
        return "Online Event"

    if any(k in s for k in ["meetup", "orientation", "community", "kickoff", "kick-off", "townhall", "town hall", "general"]):
        return "General"

    return "Other"


def classify_competition_type(event_name: str) -> str:
    s = clean_text(event_name).lower()
    if "ceo" in s:
        return "CEO"
    if "hack" in s:
        return "Hackathon"
    if "video" in s:
        return "Video"
    if "photo" in s:
        return "Photo"
    if re.search(r"\bai\b", s) or "genai" in s or "ml" in s:
        return "AI"
    if "competition" in s:
        return "Competition"
    return "Other"


def looks_like_binaryish(series: pd.Series) -> bool:
    s = series.dropna().astype(str).str.strip().str.lower()
    if s.empty:
        return False
    allowed = {"yes", "no", "y", "n", "true", "false", "1", "0", "attended", "present", "absent", ""}
    return (s.isin(allowed)).mean() >= 0.3


def compute_unique_type_attendees(df_: pd.DataFrame, event_cols_: list[str]) -> pd.DataFrame:
    """
    Unique Attendees = students who attended ONLY ONE event category/type
    (not just one event).
    Creates:
      - unique_type
      - is_unique_type_attendee (0/1)
      - type_<Category> counts
    """
    df_ = df_.copy()
    if not event_cols_:
        df_["unique_type"] = ""
        df_["is_unique_type_attendee"] = 0
        return df_

    ev_type = {ev: classify_event_category(ev) for ev in event_cols_}
    categories = sorted(set(ev_type.values()))

    for cat in categories:
        cols = [ev for ev, c in ev_type.items() if c == cat and ev in df_.columns]
        df_[f"type_{cat}"] = df_[cols].sum(axis=1) if cols else 0

    type_cols = [f"type_{cat}" for cat in categories]
    df_["_type_count_nonzero"] = (df_[type_cols] > 0).sum(axis=1)

    def pick_unique_type(r):
        if int(r["_type_count_nonzero"]) != 1:
            return ""
        for cat in categories:
            if int(r.get(f"type_{cat}", 0)) > 0:
                return cat
        return ""

    df_["unique_type"] = df_.apply(pick_unique_type, axis=1)
    df_["is_unique_type_attendee"] = (df_["unique_type"] != "").astype(int)
    df_.drop(columns=["_type_count_nonzero"], inplace=True, errors="ignore")
    return df_


# =========================
# Plot helpers
# =========================
def plotly_green_layout(fig, height=None, x_tickangle=None, bottom_margin=None):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=DARK_GREEN),
        title_font=dict(color=DARK_GREEN),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_GREEN)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_GREEN)
    if height:
        fig.update_layout(height=height)
    if x_tickangle is not None:
        fig.update_layout(xaxis_tickangle=x_tickangle)
    if bottom_margin is not None:
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=bottom_margin))
    else:
        fig.update_layout(margin=dict(l=10, r=10, t=60, b=10))
    return fig


def kpi_grid(items):
    html = '<div class="kpi-grid">'
    for label, value, sub in items:
        html += f"""
          <div class="kpi">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="sub">{sub}</div>
          </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =========================
# Loader (robust header detection)
# =========================
def load_sheet_structured(raw: pd.DataFrame):
    """
    - header row includes 'Student Name(s)' OR 'Name' + support columns like email/batch/country/payment
    - optional date row above header (older formats)
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

    # Optional date row above
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

    df_ = raw.iloc[header_row + 1:, :].copy()
    df_.columns = cols
    df_ = df_.reset_index(drop=True).dropna(how="all")

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
    return df_, meta, None


@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes):
    xls = pd.ExcelFile(file_bytes)
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def read_raw_sheet(file_bytes, sheet_name):
    return pd.read_excel(file_bytes, sheet_name=sheet_name, header=None).dropna(how="all")


# =========================
# Batch-aligned sheets (ONLY these are inferred if Batch missing)
# =========================
BATCH_ALIGNED_SHEETS = {
    "PG - B1 & B2",
    "PG - B3 & B4",
    "UG B7",
    "UG B6",
    "UG B5",
    "UG - B1 to B4",
    "Tetr-X-UG",
    "Tetr-X-PG",
}


# =========================
# Build dataset per sheet
# =========================
def build_dataset_from_sheet(file_bytes, sheet_name: str):
    raw = read_raw_sheet(file_bytes, sheet_name)
    df_, meta, err = load_sheet_structured(raw)
    if err:
        return None, None, err

    KW = {
        "name_strict": ["student name", "student names", "full name", "learner name", "student", "name"],
        "name_fallback": ["name"],
        "email": ["email", "e mail", "e-mail"],
        "phone": ["phone", "mobile", "mobile no", "phone number", "contact"],
        "country": ["country"],
        "batch": ["batch", "cohort", "group"],
        "conversion": ["conversion status", "conversion", "status", "admitted", "refunded", "pending", "will pay", "paid"],
        "payment_date": ["payment date", "payment", "paid date", "date of payment"],
        "community_status": ["community status"],
        "date_of_exit": ["date of exit", "exit date"],
    }

    name_col = best_matching_col(df_, KW["name_strict"])
    if not name_col:
        name_col = best_matching_col(
            df_,
            KW["name_fallback"],
            hard_excludes=["batch", "program", "session", "event", "country", "status"],
        )
    if not name_col:
        return None, None, "Name column not detected."

    email_col = best_matching_col(df_, KW["email"])
    phone_col = best_matching_col(df_, KW["phone"])
    country_col = best_matching_col(df_, KW["country"])
    batch_col = best_matching_col(df_, KW["batch"])
    conversion_col = best_matching_col(df_, KW["conversion"])
    payment_col = best_matching_col(df_, KW["payment_date"])
    community_col = best_matching_col(df_, KW["community_status"])
    exit_col = best_matching_col(df_, KW["date_of_exit"])

    # Batch alignment only for your specified sheets
    if sheet_name in BATCH_ALIGNED_SHEETS and not batch_col:
        inferred = infer_batch_from_sheet_name(sheet_name)
        if inferred:
            df_["Batch"] = inferred
            batch_col = "Batch"

    # Drop numeric-only summary rows when name blank
    name_series = df_[name_col].astype(str).map(clean_text)
    numeric_mask = df_.apply(row_is_numeric_only, axis=1)
    blank_name_mask = name_series.map(lambda x: x == "" or x.lower() == "nan")
    df_ = df_.loc[~(numeric_mask & blank_name_mask)].copy()

    # Payment parse
    if payment_col:
        df_[payment_col] = df_[payment_col].apply(parse_date_safe)

    # Conversion parse
    if not conversion_col:
        df_["Conversion Status"] = ""
        conversion_col = "Conversion Status"
    df_[conversion_col] = df_[conversion_col].astype(str).map(lambda x: clean_text(x).lower())

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

    df_["conversion_category"] = df_.apply(conv_category, axis=1)

    # Event columns = everything except metadata cols, filtered by attendance-like values
    metadata_cols = {c for c in [name_col, email_col, phone_col, country_col, batch_col,
                                 conversion_col, payment_col, community_col, exit_col] if c}
    event_cols = [c for c in df_.columns if c not in metadata_cols]
    event_cols = [c for c in event_cols if is_probably_event_col(df_[c])]

    # Normalize attendance to 0/1
    for c in event_cols:
        df_[c] = df_[c].apply(normalize_binary)

    df_["participation_count"] = df_[event_cols].sum(axis=1) if event_cols else 0

    # Event dates: prefer meta date row; else parse from event name
    event_dates = meta.get("event_dates", {}) or {}
    if not event_dates:
        for ev in event_cols:
            dt = parse_event_date(ev)
            if pd.notna(dt):
                event_dates[ev] = dt

    # Retention flag
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

    df_["retained"] = df_.apply(retained_flag, axis=1)

    # Program inferred from sheet name
    df_["Program"] = infer_program_from_sheet_name(sheet_name)

    ctx = {
        "meta": meta,
        "name_col": name_col,
        "email_col": email_col,
        "phone_col": phone_col,
        "country_col": country_col,
        "batch_col": batch_col,
        "conversion_col": conversion_col,
        "payment_col": payment_col,
        "community_col": community_col,
        "exit_col": exit_col,
        "event_cols": event_cols,
        "event_dates": event_dates,
    }
    return df_, ctx, None


# =========================
# Header
# =========================
st.markdown(
    """
    <div class="tetr-card">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
        <div>
          <div style="font-size:28px; font-weight:900; color:#0b3d2e;">🌿 Advanced Engagement & Competition Dashboard</div>
          <div style="margin-top:4px; color:#2e6b57; font-weight:800;">
            Robust sheet parsing • Batch-aligned UG/PG • AMAs • Country correlation • Competition analytics
          </div>
        </div>
        <div class="tetr-badge">Green / White</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.write("")


# =========================
# Upload
# =========================
uploaded_file = st.file_uploader("Upload Master Engagement Tracker Excel File", type=["xlsx"])
if not uploaded_file:
    st.stop()

file_bytes = uploaded_file.getvalue()
all_sheets = get_sheet_names(file_bytes)

# Keep your original behavior: ignore first 2 sheets
sheets = all_sheets[2:] if len(all_sheets) > 2 else all_sheets


# =========================
# Sidebar: mode + totals
# =========================
st.sidebar.markdown("## Mode")
mode = st.sidebar.selectbox("View", ["Single sheet", "Compare (multi-sheet)"], index=1)

st.sidebar.markdown("## BOFU / Headline Numbers")
bofu_total = st.sidebar.number_input("Total Attendees (BOFU)", min_value=0, value=600, step=10)
unique_target = st.sidebar.number_input("Unique Attendees target", min_value=0, value=500, step=10)

st.sidebar.markdown("---")

if mode == "Single sheet":
    selected_sheets = [st.sidebar.selectbox("Select sheet", sheets)]
else:
    default_multi = [s for s in sheets if s in BATCH_ALIGNED_SHEETS]
    if not default_multi:
        default_multi = sheets[:5]
    selected_sheets = st.sidebar.multiselect(
        "Select sheets to include",
        options=sheets,
        default=default_multi,
    )

if not selected_sheets:
    st.warning("Select at least one sheet.")
    st.stop()


# =========================
# Load selected sheets
# =========================
dfs = []
contexts = []
errors = []

for sname in selected_sheets:
    dfi, ctxi, err = build_dataset_from_sheet(file_bytes, sname)
    if err:
        errors.append((sname, err))
        continue
    dfi = dfi.copy()
    dfi["__sheet__"] = sname
    dfs.append(dfi)
    contexts.append((sname, ctxi))

if errors and not dfs:
    st.error("Could not parse any selected sheets.")
    for sname, err in errors:
        st.write(f"- **{sname}**: {err}")
    st.stop()

if errors:
    with st.expander("⚠️ Some selected sheets could not be parsed"):
        for sname, err in errors:
            st.write(f"- **{sname}**: {err}")

df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
primary_ctx = contexts[0][1]

name_col = primary_ctx["name_col"]
country_col = primary_ctx["country_col"]
batch_col = primary_ctx["batch_col"]
conversion_col = primary_ctx["conversion_col"]
payment_col = primary_ctx["payment_col"]

# UNION event columns + dates across sheets (fixes your earlier crash)
all_event_cols = set()
event_dates = {}
for _, ctxi in contexts:
    for ev in ctxi["event_cols"]:
        all_event_cols.add(ev)
    for ev, dt in (ctxi["event_dates"] or {}).items():
        if pd.notna(dt):
            event_dates.setdefault(ev, dt)

event_cols = sorted(all_event_cols)

# Ensure union columns exist + numeric 0/1 across combined df
for ev in event_cols:
    if ev not in df.columns:
        df[ev] = 0
    df[ev] = pd.to_numeric(df[ev], errors="coerce").fillna(0).astype(int)

# Recompute participation_count safely on union
df["participation_count"] = df[event_cols].sum(axis=1) if event_cols else 0


# =========================
# Outcome columns for competitions (win/spotlight)
# =========================
# Scan all columns for "win/winner/spotlight" style flags and normalize if binary-ish
outcome_keywords = ["win", "winner", "spotlight"]
outcome_cols = []
for c in df.columns:
    cl = clean_text(c).lower()
    if any(k in cl for k in outcome_keywords):
        if looks_like_binaryish(df[c]):
            outcome_cols.append(c)

for c in outcome_cols:
    df[c] = df[c].apply(normalize_binary)

# =========================
# Filters (dropdowns)
# =========================
st.sidebar.markdown("## Filters (Dropdowns)")

conv_present = sorted(df["conversion_category"].dropna().unique().tolist())
conv_pick = st.sidebar.selectbox("Conversion category", ["All"] + conv_present, index=0)

min_part = st.sidebar.slider("Minimum participation count", 0, int(df["participation_count"].max() or 0), 0)

sheet_pick = "All"
if "__sheet__" in df.columns and len(selected_sheets) > 1:
    sheet_vals = sorted(df["__sheet__"].dropna().astype(str).unique().tolist())
    sheet_pick = st.sidebar.selectbox("Sheet", ["All"] + sheet_vals, index=0)

program_pick = st.sidebar.selectbox("Program", ["All", "UG", "PG", "Unknown"], index=0)

batch_pick = "All"
if batch_col and batch_col in df.columns:
    batches = sorted([b for b in df[batch_col].dropna().astype(str).map(clean_text).unique().tolist() if b])
    if batches:
        batch_pick = st.sidebar.selectbox("Batch", ["All"] + batches, index=0)

country_pick = "All"
if country_col and country_col in df.columns:
    countries = sorted([c for c in df[country_col].dropna().astype(str).map(clean_text).unique().tolist() if c])
    if countries:
        country_pick = st.sidebar.selectbox("Country", ["All"] + countries, index=0)

search_text = st.sidebar.text_input("Search student (contains)", "")

# Apply filters
fdf = df.copy()
if conv_pick != "All":
    fdf = fdf[fdf["conversion_category"] == conv_pick]
fdf = fdf[fdf["participation_count"] >= min_part]
if sheet_pick != "All":
    fdf = fdf[fdf["__sheet__"].astype(str) == sheet_pick]
if program_pick != "All":
    fdf = fdf[fdf["Program"] == program_pick]
if batch_col and batch_pick != "All":
    fdf = fdf[fdf[batch_col].astype(str).map(clean_text) == batch_pick]
if country_col and country_pick != "All":
    fdf = fdf[fdf[country_col].astype(str).map(clean_text) == country_pick]
if search_text.strip():
    q = search_text.strip().lower()
    fdf = fdf[fdf[name_col].astype(str).str.lower().str.contains(q, na=False)]

# Ensure fdf has all event cols numeric
for ev in event_cols:
    if ev not in fdf.columns:
        fdf[ev] = 0
    fdf[ev] = pd.to_numeric(fdf[ev], errors="coerce").fillna(0).astype(int)
fdf["participation_count"] = fdf[event_cols].sum(axis=1) if event_cols else 0

# Recompute unique-type attendees for df and fdf
df_u = compute_unique_type_attendees(df, event_cols)
fdf_u = compute_unique_type_attendees(fdf, event_cols)

# =========================
# Core KPIs (keeps older + adds requested)
# =========================
total_students_all = int(df[name_col].notna().sum())
total_students_f = int(fdf[name_col].notna().sum())

active_all = int((df["participation_count"] > 0).sum())
active_f = int((fdf["participation_count"] > 0).sum())

paid_all = int((df["conversion_category"] == "Paid / Admitted").sum())
paid_f = int((fdf["conversion_category"] == "Paid / Admitted").sum())

participants_all = active_all
participants_f = active_f

conv_rate_all = (paid_all / participants_all * 100) if participants_all else 0.0
conv_rate_f = (paid_f / participants_f * 100) if participants_f else 0.0

ret_all = float(df["retained"].mean() * 100) if payment_col and payment_col in df.columns else None
ret_f = float(fdf["retained"].mean() * 100) if payment_col and payment_col in fdf.columns else None

unique_type_all = int(df_u["is_unique_type_attendee"].sum()) if "is_unique_type_attendee" in df_u.columns else 0
unique_type_f = int(fdf_u["is_unique_type_attendee"].sum()) if "is_unique_type_attendee" in fdf_u.columns else 0

kpi_grid([
    ("Total Attendees (BOFU)", f"{int(bofu_total)}", "Fixed (editable in sidebar)"),
    ("Unique Attendees", f"{unique_type_f}", f"All: {unique_type_all} | Target: {int(unique_target)}"),
    ("Total Students", f"{total_students_f}", f"All: {total_students_all}"),
    ("Active Students", f"{active_f}", f"All: {active_all}"),
    ("Paid / Admitted", f"{paid_f}", f"All: {paid_all}"),
    ("Conversion Rate", f"{conv_rate_f:.1f}%", f"All: {conv_rate_all:.1f}%"),
])

st.caption(f"Rows (filtered / all): {len(fdf)} / {len(df)} | Events detected (union): {len(event_cols)}")
st.write("")

# =========================
# Downloads (keeps previous)
# =========================
c1, c2 = st.columns(2)
with c1:
    st.download_button(
        "⬇️ Download filtered data (CSV)",
        data=fdf.to_csv(index=False).encode("utf-8"),
        file_name="filtered.csv",
        mime="text/csv",
        use_container_width=True,
    )
with c2:
    event_summary = pd.DataFrame({
        "Event": event_cols,
        "Event Category": [classify_event_category(e) for e in event_cols],
        "Event Date (if known)": [event_dates.get(e, pd.NaT) for e in event_cols],
        "Participants (filtered)": [safe_event_sum(fdf, e) for e in event_cols],
        "Participants (all)": [safe_event_sum(df, e) for e in event_cols],
    }).sort_values(["Participants (filtered)", "Event"], ascending=[False, True])

    st.download_button(
        "⬇️ Download event summary (CSV)",
        data=event_summary.to_csv(index=False).encode("utf-8"),
        file_name="event_summary.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.divider()

# =========================
# Tabs (keeps old sections + adds new)
# =========================
tab_overview, tab_students, tab_conv_ret, tab_events, tab_ama, tab_country, tab_comp, tab_timeline, tab_quality = st.tabs(
    ["🌿 Overview", "👥 Students", "💳 Conversion & Retention", "🎯 Events", "🎙️ AMAs", "🌍 Country", "🏁 Competitions", "🗓️ Timeline", "🧪 Data Quality"]
)

# ============================================================
# 🌿 OVERVIEW
# ============================================================
with tab_overview:
    st.subheader("Unique Attendees breakdown (attended only one category/type)")
    if len(fdf_u) == 0:
        st.info("No rows in filtered view.")
    else:
        b = (
            fdf_u[fdf_u.get("unique_type", "") != ""]
            .groupby("unique_type", as_index=False)
            .agg(Unique_Attendees=(name_col, "count"))
            .sort_values("Unique_Attendees", ascending=False)
        )
        cA, cB = st.columns([1.1, 1])
        with cA:
            st.dataframe(b, use_container_width=True, height=260)
        with cB:
            if not b.empty:
                fig = px.bar(b, x="unique_type", y="Unique_Attendees", title="Unique attendees by type")
                fig.update_traces(marker_color=GREEN)
                st.plotly_chart(plotly_green_layout(fig, height=360), use_container_width=True)
            else:
                st.info("No unique-type attendees found in this filtered view.")

    st.divider()

    st.subheader("Average attendees per session (UG vs PG)")
    if not event_cols or len(fdf) == 0:
        st.info("Not enough data to compute session averages.")
    else:
        def avg_attendees_per_session(subdf: pd.DataFrame):
            if len(subdf) == 0 or not event_cols:
                return 0.0, 0
            counts = [safe_event_sum(subdf, e) for e in event_cols]
            nonzero = [c for c in counts if c > 0]
            if not nonzero:
                return 0.0, 0
            return float(np.mean(nonzero)), len(nonzero)

        ug = fdf[fdf["Program"] == "UG"]
        pg = fdf[fdf["Program"] == "PG"]
        ug_avg, ug_sessions = avg_attendees_per_session(ug)
        pg_avg, pg_sessions = avg_attendees_per_session(pg)
        overall_avg, overall_sessions = avg_attendees_per_session(fdf)

        kpi_grid([
            ("UG Avg / Session", f"{ug_avg:.1f}", f"Sessions with attendance: {ug_sessions}"),
            ("PG Avg / Session", f"{pg_avg:.1f}", f"Sessions with attendance: {pg_sessions}"),
            ("Overall Avg / Session", f"{overall_avg:.1f}", f"Sessions with attendance: {overall_sessions}"),
            ("UG Active", f"{int((ug['participation_count']>0).sum())}", "Filtered"),
            ("PG Active", f"{int((pg['participation_count']>0).sum())}", "Filtered"),
            ("Events Detected", f"{len(event_cols)}", "Union across selected sheets"),
        ])

        dist = fdf["participation_count"].value_counts().sort_index().reset_index()
        dist.columns = ["Participation Count", "Students"]
        fig = px.bar(dist, x="Participation Count", y="Students", title="Participation distribution (filtered)")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(plotly_green_layout(fig, height=380), use_container_width=True)

    st.divider()

    st.subheader("Engagement → Conversion Funnel (bucketed participation)")
    if len(fdf) > 0:
        tmp = fdf.copy()
        tmp["eng_bucket"] = pd.cut(
            tmp["participation_count"],
            bins=[-1, 0, 2, 5, 999999],
            labels=["0", "1–2", "3–5", "6+"],
        )
        funnel = (
            tmp.groupby(["eng_bucket", "conversion_category"], as_index=False)
            .size()
            .rename(columns={"size": "Students"})
        )
        fig_fun = px.bar(
            funnel,
            x="eng_bucket",
            y="Students",
            color="conversion_category",
            barmode="stack",
            color_discrete_map=GREEN_PALETTE,
            title="How conversion changes with participation level",
        )
        st.plotly_chart(plotly_green_layout(fig_fun, height=420), use_container_width=True)
    else:
        st.info("No rows available in the filtered view.")

    st.divider()

    st.subheader("Cohort performance (Batch/Country)")
    group_col = batch_col if batch_col and batch_col in fdf.columns else (country_col if country_col and country_col in fdf.columns else None)
    if group_col and len(fdf) > 0:
        tmp = fdf.copy()
        tmp[group_col] = tmp[group_col].astype(str).map(clean_text)
        grp = tmp.groupby(group_col, as_index=False).agg(
            Students=(name_col, "count"),
            Active=("participation_count", lambda s: int((s > 0).sum())),
            AvgParticipation=("participation_count", "mean"),
            PaidRate=("conversion_category", lambda s: float((s == "Paid / Admitted").mean() * 100)),
        )
        grp["ActiveRate%"] = (grp["Active"] / grp["Students"] * 100).round(1)
        grp["AvgParticipation"] = grp["AvgParticipation"].round(2)
        grp["PaidRate"] = grp["PaidRate"].round(1)
        grp = grp.sort_values(["PaidRate", "ActiveRate%", "Students"], ascending=False)

        st.dataframe(grp, use_container_width=True, height=360)
        fig = px.bar(grp.head(25), x=group_col, y="PaidRate", title=f"Paid rate % by {group_col} (Top 25)")
        fig.update_traces(marker_color=DARK_GREEN)
        st.plotly_chart(plotly_green_layout(fig, height=380, x_tickangle=-30, bottom_margin=120), use_container_width=True)
    else:
        st.info("No Batch/Country column detected for cohort performance.")


# ============================================================
# 👥 STUDENTS (keeps older lists + action lists)
# ============================================================
with tab_students:
    st.subheader("Top participating students (Filtered)")
    cols = [name_col, "participation_count", "conversion_category"]
    if "__sheet__" in fdf.columns:
        cols.append("__sheet__")
    st.dataframe(
        fdf.sort_values("participation_count", ascending=False)[cols].head(300),
        use_container_width=True,
        height=420,
    )

    st.write("")
    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### 🎯 High engagement but not paid (best conversion targets)")
        targets = (
            fdf[(fdf["conversion_category"] != "Paid / Admitted") & (fdf["participation_count"] >= 3)]
            .sort_values(["participation_count"], ascending=False)
            [[name_col, "participation_count", "conversion_category"] + (["__sheet__"] if "__sheet__" in fdf.columns else [])]
            .head(25)
        )
        st.dataframe(targets, use_container_width=True, height=320)

    with colB:
        st.markdown("#### 🧯 Paid but low engagement (retention risk)")
        risk_cols = [name_col, "participation_count"]
        if payment_col and payment_col in fdf.columns:
            risk_cols.insert(1, payment_col)
        if "__sheet__" in fdf.columns:
            risk_cols.append("__sheet__")
        risks = (
            fdf[(fdf["conversion_category"] == "Paid / Admitted") & (fdf["participation_count"] <= 1)]
            .sort_values(["participation_count"], ascending=True)
            [risk_cols]
            .head(25)
        )
        st.dataframe(risks, use_container_width=True, height=320)

    st.divider()

    st.subheader("Students with NO event participation (Filtered)")
    no_cols = [name_col, conversion_col] if conversion_col in fdf.columns else [name_col, "conversion_category"]
    if payment_col and payment_col in fdf.columns:
        no_cols.append(payment_col)
    if "__sheet__" in fdf.columns:
        no_cols.append("__sheet__")
    st.dataframe(
        fdf[fdf["participation_count"] == 0][no_cols],
        use_container_width=True,
        height=320,
    )


# ============================================================
# 💳 CONVERSION & RETENTION (keeps older conversion tables + before/after payment)
# ============================================================
with tab_conv_ret:
    st.subheader("Conversion buckets (Filtered)")
    paid_df = fdf[fdf["conversion_category"] == "Paid / Admitted"].copy()
    will_df = fdf[fdf["conversion_category"] == "Will Pay"].copy()
    pending_df = fdf[fdf["conversion_category"] == "Pending"].copy()
    refunded_df = fdf[fdf["conversion_category"] == "Refunded"].copy()
    not_df = fdf[fdf["conversion_category"] == "Not Paid"].copy()

    base_cols = [name_col, "conversion_category", "participation_count"]
    if conversion_col and conversion_col in fdf.columns:
        base_cols.insert(1, conversion_col)
    if payment_col and payment_col in fdf.columns:
        base_cols.insert(2, payment_col)
    if "__sheet__" in fdf.columns:
        base_cols.append("__sheet__")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### ✅ Paid / Admitted")
        st.dataframe(paid_df[base_cols], use_container_width=True, height=260)
    with c2:
        st.markdown("### 🟡 Will Pay")
        st.dataframe(will_df[base_cols], use_container_width=True, height=260)
    with c3:
        st.markdown("### 🟠 Pending")
        st.dataframe(pending_df[base_cols], use_container_width=True, height=260)

    c4, c5 = st.columns(2)
    with c4:
        st.markdown("### 🔵 Refunded")
        st.dataframe(refunded_df[base_cols], use_container_width=True, height=240)
    with c5:
        st.markdown("### 🔴 Not Paid")
        st.dataframe(not_df[base_cols], use_container_width=True, height=240)

    st.divider()

    st.subheader("Raw conversion status values (Filtered)")
    if conversion_col and conversion_col in fdf.columns:
        raw_conv = fdf[conversion_col].replace({"": np.nan, "nan": np.nan}).dropna()
        raw_conv_counts = raw_conv.value_counts().reset_index()
        raw_conv_counts.columns = ["Conversion Status", "Count"]
        st.dataframe(raw_conv_counts, use_container_width=True, height=260)
    else:
        st.info("No raw conversion column detected (using conversion_category only).")

    st.divider()

    st.subheader("Retention analysis")
    if payment_col and payment_col in fdf.columns:
        st.metric("Retention Rate (Filtered)", f"{(ret_f or 0):.2f}%")
        retained = fdf[fdf["retained"] == 1][[name_col, payment_col, "participation_count"] + (["__sheet__"] if "__sheet__" in fdf.columns else [])].copy()
        st.markdown("Retained students (paid + attended after payment date)")
        st.dataframe(retained, use_container_width=True, height=280)

        # Retention by cohort
        group_col = batch_col if batch_col and batch_col in fdf.columns else (country_col if country_col and country_col in fdf.columns else None)
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

    # Engagement before vs after payment (restored)
    st.subheader("Engagement Before vs After Payment Date (Filtered)")
    if payment_col and payment_col in fdf.columns and event_dates and event_cols and len(fdf) > 0:
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
            fig_pay.add_vline(x=0, line_dash="dash", line_color=DARK_GREEN)
            fig_pay.update_traces(line_color=DARK_GREEN)
            st.plotly_chart(plotly_green_layout(fig_pay, height=420), use_container_width=True)
            st.caption("Day 0 = payment date. Positive days = events after payment.")
    else:
        st.info("Needs both Payment Date and Event Dates for this view.")


# ============================================================
# 🎯 EVENTS (keeps event table + bar + Pareto + impact + heatmap)
# ============================================================
with tab_events:
    st.subheader("Event-wise participation (Filtered)")
    if not event_cols:
        st.info("No event columns detected.")
    else:
        event_counts = pd.Series({ev: safe_event_sum(fdf, ev) for ev in event_cols}).sort_values(ascending=False)
        event_pct = (event_counts / max(len(fdf), 1) * 100).round(1)
        event_table = pd.DataFrame({
            "Event": event_counts.index,
            "Category": [classify_event_category(e) for e in event_counts.index],
            "Participants": event_counts.values.astype(int),
            "Participation %": event_pct.values,
            "Event Date (if known)": [event_dates.get(ev, pd.NaT) for ev in event_counts.index],
        })
        st.dataframe(event_table, use_container_width=True, height=360)

        fig_bar = px.bar(
            event_table.head(40),
            x="Event",
            y="Participants",
            color="Category",
            title="Top events by participants (Top 40)",
        )
        st.plotly_chart(plotly_green_layout(fig_bar, height=460, x_tickangle=-45, bottom_margin=170), use_container_width=True)

    st.divider()

    st.subheader("Pareto (80/20) — event contribution to total participation")
    if event_cols and len(fdf) > 0:
        counts = pd.Series({ev: safe_event_sum(fdf, ev) for ev in event_cols}).sort_values(ascending=False)
        if counts.sum() == 0:
            st.info("No event participation in current filtered view.")
        else:
            pareto = pd.DataFrame({"Event": counts.index, "Participants": counts.values.astype(int)})
            pareto["Cumulative %"] = (pareto["Participants"].cumsum() / pareto["Participants"].sum() * 100).round(1)

            fig_par = go.Figure()
            fig_par.add_trace(go.Bar(x=pareto["Event"], y=pareto["Participants"], name="Participants",
                                     marker_color=GREEN))
            fig_par.add_trace(go.Scatter(x=pareto["Event"], y=pareto["Cumulative %"], yaxis="y2",
                                         name="Cumulative %", line=dict(color=DARK_GREEN)))
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
                font=dict(color=DARK_GREEN),
            )
            fig_par.update_xaxes(showgrid=True, gridcolor=GRID_GREEN)
            fig_par.update_yaxes(showgrid=True, gridcolor=GRID_GREEN)
            st.plotly_chart(fig_par, use_container_width=True)
    else:
        st.info("Pareto needs event columns.")

    st.divider()

    st.subheader("Event conversion impact — Paid rate among attendees")
    if event_cols and len(fdf) > 0:
        rows = []
        for ev in event_cols:
            attendees = fdf[fdf[ev] == 1]
            n_att = len(attendees)
            if n_att == 0:
                continue
            paid_rate = float((attendees["conversion_category"].eq("Paid / Admitted")).mean() * 100)
            rows.append({"Event": ev, "Category": classify_event_category(ev), "Attendees": n_att, "Paid Rate %": round(paid_rate, 1)})
        impact = pd.DataFrame(rows)
        if impact.empty:
            st.info("No attendee data found in this filtered view.")
        else:
            impact = impact.sort_values(["Paid Rate %", "Attendees"], ascending=False)
            min_att = st.slider("Minimum attendees to include an event", 1, int(max(1, impact["Attendees"].max())), min(10, int(max(1, impact["Attendees"].max()))))
            impact_f = impact[impact["Attendees"] >= min_att].head(40)

            fig_imp = px.bar(
                impact_f,
                x="Event",
                y="Paid Rate %",
                color="Category",
                hover_data=["Attendees"],
                title="Paid rate among attendees (higher = stronger conversion signal)",
            )
            st.plotly_chart(plotly_green_layout(fig_imp, height=480, x_tickangle=-45, bottom_margin=170), use_container_width=True)
    else:
        st.info("No event columns detected.")

    st.divider()

    st.subheader("Cohort heatmap — Participation % by Event (Batch/Country)")
    group_col = batch_col if batch_col and batch_col in fdf.columns else (country_col if country_col and country_col in fdf.columns else None)
    if group_col and event_cols and len(fdf) > 0:
        heat = fdf.copy()
        heat[group_col] = heat[group_col].astype(str).map(clean_text)
        grp_sizes = heat.groupby(group_col)[name_col].count()
        groups = [g for g in grp_sizes.index.tolist() if g]

        mat = []
        for g in groups:
            gdf = heat[heat[group_col] == g]
            denom = max(len(gdf), 1)
            row = {"Group": g}
            for ev in event_cols:
                row[ev] = round((gdf[ev].sum() / denom) * 100, 1)
            mat.append(row)

        mat_df = pd.DataFrame(mat).set_index("Group")
        if mat_df.empty:
            st.info("Not enough data to build heatmap.")
        else:
            fig_hm = px.imshow(mat_df, aspect="auto", title=f"Participation % by Event across {group_col}")
            st.plotly_chart(plotly_green_layout(fig_hm, height=560), use_container_width=True)
    else:
        st.info("Heatmap needs Batch/Country and event columns.")


# ============================================================
# 🎙️ AMAs (Top performing)
# ============================================================
with tab_ama:
    st.subheader("Top performing AMAs")
    ama_events = [e for e in event_cols if is_ama_event(e)]
    if not ama_events:
        st.info("No AMA events detected (event name must contain 'AMA').")
    else:
        rows = []
        for ev in ama_events:
            attendees = fdf[fdf[ev] == 1]
            n_att = len(attendees)
            if n_att == 0:
                continue
            paid_rate = float((attendees["conversion_category"] == "Paid / Admitted").mean() * 100)
            rows.append({
                "AMA": ev,
                "Attendees": n_att,
                "Paid Rate % among attendees": round(paid_rate, 1),
                "UG%": round((attendees["Program"] == "UG").mean() * 100, 1) if len(attendees) else 0.0,
                "Event Date (if known)": event_dates.get(ev, pd.NaT),
            })
        ama_df = pd.DataFrame(rows).sort_values(["Paid Rate % among attendees", "Attendees"], ascending=False)
        st.dataframe(ama_df, use_container_width=True, height=360)

        fig1 = px.bar(ama_df.head(20), x="AMA", y="Attendees",
                      hover_data=["Paid Rate % among attendees", "Event Date (if known)"],
                      title="Top AMAs by attendance (Top 20)")
        fig1.update_traces(marker_color=GREEN)
        st.plotly_chart(plotly_green_layout(fig1, height=440, x_tickangle=-45, bottom_margin=170), use_container_width=True)

        fig2 = px.bar(ama_df.head(20), x="AMA", y="Paid Rate % among attendees",
                      hover_data=["Attendees"],
                      title="Top AMAs by paid rate among attendees (Top 20)")
        fig2.update_traces(marker_color=DARK_GREEN)
        st.plotly_chart(plotly_green_layout(fig2, height=440, x_tickangle=-45, bottom_margin=170), use_container_width=True)


# ============================================================
# 🌍 COUNTRY (correlation + heatmap)
# ============================================================
with tab_country:
    st.subheader("Attendance and country correlation")
    if not country_col or country_col not in fdf.columns:
        st.info("No Country column detected in this view.")
    elif len(fdf) == 0:
        st.info("No rows in filtered view.")
    else:
        tmp = fdf.copy()
        tmp[country_col] = tmp[country_col].astype(str).map(clean_text)
        tmp = tmp[tmp[country_col] != ""]

        country_stats = tmp.groupby(country_col, as_index=False).agg(
            Students=(name_col, "count"),
            Active=("participation_count", lambda s: int((s > 0).sum())),
            AvgParticipation=("participation_count", "mean"),
            PaidRate=("conversion_category", lambda s: float((s == "Paid / Admitted").mean() * 100)),
        )
        country_stats["Active Rate %"] = (country_stats["Active"] / country_stats["Students"] * 100).round(1)
        country_stats["AvgParticipation"] = country_stats["AvgParticipation"].round(2)
        country_stats["PaidRate"] = country_stats["PaidRate"].round(1)
        country_stats = country_stats.sort_values(["Students"], ascending=False)

        st.dataframe(country_stats, use_container_width=True, height=360)

        fig = px.scatter(
            country_stats,
            x="AvgParticipation",
            y="PaidRate",
            size="Students",
            hover_name=country_col,
            title="Country correlation: Avg participation vs Paid rate (size = students)",
        )
        fig.update_traces(marker=dict(color=GREEN, line=dict(color=DARK_GREEN, width=1)))
        st.plotly_chart(plotly_green_layout(fig, height=420), use_container_width=True)

        # Heatmap: Event Category participation rate by country (top N countries)
        st.subheader("Country × Event Category heatmap (participation %)")
        topN = st.slider("Top countries (by students)", 5, min(50, len(country_stats)), min(15, len(country_stats)))
        top_countries = country_stats.head(topN)[country_col].tolist()
        sub = tmp[tmp[country_col].isin(top_countries)].copy()

        # Build category participation per country (any attendance in that category)
        if event_cols:
            # per-student category totals exist in compute_unique_type_attendees output; rebuild quickly for this sub
            sub_u = compute_unique_type_attendees(sub, event_cols)
            cat_cols = [c for c in sub_u.columns if c.startswith("type_")]
            if cat_cols:
                # compute "has any" per category per student
                for c in cat_cols:
                    sub_u[c] = (sub_u[c] > 0).astype(int)

                heat_rows = []
                for ctry in top_countries:
                    cdf = sub_u[sub_u[country_col] == ctry]
                    denom = max(len(cdf), 1)
                    row = {"Country": ctry}
                    for c in cat_cols:
                        row[c.replace("type_", "")] = round((cdf[c].sum() / denom) * 100, 1)
                    heat_rows.append(row)

                heat_df = pd.DataFrame(heat_rows).set_index("Country")
                fig_h = px.imshow(heat_df, aspect="auto", title="Participation % by event category across countries")
                st.plotly_chart(plotly_green_layout(fig_h, height=520), use_container_width=True)
            else:
                st.info("Could not compute category heatmap (no category columns).")
        else:
            st.info("No event columns detected.")


# ============================================================
# 🏁 COMPETITIONS
# ============================================================
with tab_comp:
    st.subheader("Competition analytics")

    if not event_cols or len(fdf) == 0:
        st.info("No data to compute competition performance.")
    else:
        # Competition-related events (by event category OR keywords)
        comp_events = [e for e in event_cols if classify_event_category(e) == "Competition" or any(k in e.lower() for k in ["competition", "hack", "ceo", "video", "photo", "ai", "genai", "spotlight", "win"])]

        # If outcome cols not found, we still show participation + paid/participated
        # If found, "Paid & Win/Spotlight" uses the best matching outcome col
        used_outcome = None
        if outcome_cols:
            # prefer spotlight over win if present
            ranked = sorted(outcome_cols, key=lambda c: ("spotlight" in c.lower(), "win" in c.lower()), reverse=True)
            used_outcome = ranked[0]

        if not comp_events:
            st.info("No competition-like event columns detected (needs keywords like competition/hackathon/ceo/video/photo/AI).")
        else:
            comp_pick = st.selectbox("Select a competition/event to analyze", comp_events)

            participants = fdf[fdf[comp_pick] == 1] if comp_pick in fdf.columns else fdf.iloc[0:0]
            n_part = len(participants)
            total_base = len(fdf)
            participation_pct = (n_part / total_base * 100) if total_base else 0.0

            paid_and_part = participants[participants["conversion_category"] == "Paid / Admitted"]
            n_paid_part = len(paid_and_part)
            paid_part_pct = (n_paid_part / n_part * 100) if n_part else 0.0

            n_paid_win = 0
            paid_win_pct = 0.0
            if used_outcome and used_outcome in participants.columns:
                n_paid_win = int((paid_and_part[used_outcome] == 1).sum())
                paid_win_pct = (n_paid_win / n_paid_part * 100) if n_paid_part else 0.0

            # KPI row matching your example
            kpi_grid([
                ("Participation", f"{n_part}", f"{participation_pct:.1f}% of filtered base"),
                ("Paid & Participated", f"{n_paid_part}", f"{paid_part_pct:.1f}% of participants"),
                ("Paid & Win/Spotlight", f"{n_paid_win}", f"{paid_win_pct:.1f}% of paid participants" if used_outcome else "Outcome col not detected"),
                ("Type", classify_competition_type(comp_pick), "Auto from event name"),
                ("UG% (participants)", f"{(participants['Program']=='UG').mean()*100:.1f}%" if n_part else "0.0%", "Among participants"),
                ("PG% (participants)", f"{(participants['Program']=='PG').mean()*100:.1f}%" if n_part else "0.0%", "Among participants"),
            ])

            show_cols = [name_col, "Program", "conversion_category", "participation_count"]
            if batch_col and batch_col in fdf.columns:
                show_cols.insert(2, batch_col)
            if country_col and country_col in fdf.columns:
                show_cols.append(country_col)
            if payment_col and payment_col in fdf.columns:
                show_cols.append(payment_col)
            if used_outcome and used_outcome in fdf.columns:
                show_cols.append(used_outcome)
            if "__sheet__" in fdf.columns:
                show_cols.append("__sheet__")

            st.markdown("#### Participants list")
            st.dataframe(participants[show_cols].head(500), use_container_width=True, height=360)

        st.divider()

        st.subheader("Overall competition performance: UG vs PG, Batchwise, Type-wise")
        if comp_events:
            comp_rows = []
            for ev in comp_events:
                attendees = fdf[fdf[ev] == 1]
                n_att = len(attendees)
                if n_att == 0:
                    continue
                paid_rate = float((attendees["conversion_category"] == "Paid / Admitted").mean() * 100)
                comp_rows.append({
                    "Competition/Event": ev,
                    "Type": classify_competition_type(ev),
                    "Participants": n_att,
                    "Paid Rate % (participants)": round(paid_rate, 1),
                    "UG%": round((attendees["Program"] == "UG").mean() * 100, 1),
                    "PG%": round((attendees["Program"] == "PG").mean() * 100, 1),
                })
            comp_df = pd.DataFrame(comp_rows).sort_values(["Participants"], ascending=False)
            st.dataframe(comp_df, use_container_width=True, height=360)

            # Type performance chart
            type_perf = comp_df.groupby("Type", as_index=False).agg(
                Participants=("Participants", "sum"),
                AvgPaidRate=("Paid Rate % (participants)", "mean"),
            )
            type_perf["AvgPaidRate"] = type_perf["AvgPaidRate"].round(1)
            type_perf = type_perf.sort_values("Participants", ascending=False)

            fig_type = px.bar(type_perf, x="Type", y="Participants", hover_data=["AvgPaidRate"], title="Competition type performance (participants)")
            fig_type.update_traces(marker_color=GREEN)
            st.plotly_chart(plotly_green_layout(fig_type, height=380, x_tickangle=-15), use_container_width=True)

            # UG vs PG total competition participations (counts, not unique students)
            ug_part = int(fdf[fdf["Program"] == "UG"][comp_events].sum().sum()) if comp_events else 0
            pg_part = int(fdf[fdf["Program"] == "PG"][comp_events].sum().sum()) if comp_events else 0
            fig_up = px.bar(pd.DataFrame({"Program": ["UG", "PG"], "Competition participations": [ug_part, pg_part]}),
                            x="Program", y="Competition participations",
                            title="Overall competition participations: UG vs PG (filtered)")
            fig_up.update_traces(marker_color=DARK_GREEN)
            st.plotly_chart(plotly_green_layout(fig_up, height=320), use_container_width=True)

            # Batchwise UG & PG breakdown
            if batch_col and batch_col in fdf.columns:
                btmp = fdf.copy()
                btmp[batch_col] = btmp[batch_col].astype(str).map(clean_text)
                btmp["comp_participations"] = btmp[comp_events].sum(axis=1) if comp_events else 0

                bsum = btmp.groupby(["Program", batch_col], as_index=False).agg(
                    Students=(name_col, "count"),
                    CompParticipants=("comp_participations", lambda s: int((s > 0).sum())),
                    TotalCompParticipations=("comp_participations", "sum"),
                    AvgCompParticipations=("comp_participations", "mean"),
                )
                bsum["AvgCompParticipations"] = bsum["AvgCompParticipations"].round(2)
                bsum = bsum.sort_values(["Program", batch_col])

                st.markdown("#### Batchwise competition performance (filtered)")
                st.dataframe(bsum, use_container_width=True, height=360)

                fig_b = px.bar(
                    bsum,
                    x=batch_col,
                    y="AvgCompParticipations",
                    color="Program",
                    barmode="group",
                    title="Average competition participations per student by batch",
                    color_discrete_map={"UG": GREEN, "PG": DARK_GREEN, "Unknown": LIGHT_GREEN},
                )
                st.plotly_chart(plotly_green_layout(fig_b, height=420, x_tickangle=-30, bottom_margin=120), use_container_width=True)
            else:
                st.info("Batch column not available in this view (batchwise analysis skipped).")
        else:
            st.info("No competition events detected.")


# ============================================================
# 🗓️ TIMELINE (trend + per-student timeline)
# ============================================================
with tab_timeline:
    st.subheader("Participation trend over time (dated events)")
    if event_dates and event_cols and len(fdf) > 0:
        dated_events = [(ev, event_dates.get(ev, pd.NaT)) for ev in event_cols]
        dated_events = [(ev, dt) for ev, dt in dated_events if pd.notna(dt)]
        if dated_events:
            trend = [(dt, safe_event_sum(fdf, ev)) for ev, dt in dated_events]
            trend_df = pd.DataFrame(trend, columns=["Event Date", "Participants"]).groupby("Event Date", as_index=False).sum()
            trend_df = trend_df.sort_values("Event Date")

            fig_tr = px.line(trend_df, x="Event Date", y="Participants", markers=True, title="Participants over time (filtered)")
            fig_tr.update_traces(line_color=DARK_GREEN)
            rng = safe_datetime_range_with_padding(trend_df["Event Date"].tolist(), pad_days=3)
            if rng:
                fig_tr.update_xaxes(range=rng)
            st.plotly_chart(plotly_green_layout(fig_tr, height=380), use_container_width=True)
        else:
            st.info("No dated events mapped.")
    else:
        st.info("No event dates detected/inferred for this view.")

    st.divider()

    st.subheader("Per-student participation timeline")
    students = fdf[name_col].dropna().astype(str).unique().tolist()
    if not students:
        st.info("No students in filtered view.")
    else:
        selected_student = st.selectbox("Select student", students)
        row = fdf[fdf[name_col].astype(str) == str(selected_student)].iloc[0]

        use_dates = bool(event_dates)
        timeline_events = event_cols[:]
        if use_dates:
            timeline_events = sorted(timeline_events, key=lambda ev: (pd.isna(event_dates.get(ev, pd.NaT)), event_dates.get(ev, pd.NaT)))

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

        attended_x, attended_y, missed_x, missed_y = [], [], [], []
        for xv, a in zip(x_vals, attended):
            if a == 1:
                attended_x.append(xv); attended_y.append(1)
            else:
                missed_x.append(xv); missed_y.append(0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=x_vals, y=[0.5]*len(x_vals), mode="lines",
            hoverinfo="skip", showlegend=False,
            line=dict(width=1, dash="dot", color=LIGHT_GREEN), opacity=0.35
        ))
        fig.add_trace(go.Scatter(
            x=attended_x, y=attended_y, mode="lines+markers",
            name="Attended", connectgaps=False,
            line=dict(color=DARK_GREEN), marker=dict(color=DARK_GREEN, size=8)
        ))
        fig.add_trace(go.Scatter(
            x=missed_x, y=missed_y, mode="markers",
            name="Missed", marker=dict(symbol="x", size=10, color=GREEN)
        ))

        # Payment marker (kept)
        if payment_col and payment_col in fdf.columns and pd.notna(row.get(payment_col, pd.NaT)) and use_dates:
            pay_dt = row[payment_col]
            fig.add_trace(go.Scatter(
                x=[pay_dt], y=[1.18], mode="markers+text",
                text=["<b>✔ Payment</b>"], textposition="top center",
                name="Payment", marker=dict(symbol="star", size=18, color=GREEN),
                textfont=dict(size=14, color=GREEN),
            ))

        fig.update_yaxes(range=[-0.2, 1.3], tickvals=[0, 1], title="Participation (1=Attended, 0=Missed)")
        fig.update_layout(
            title=f"Timeline — {selected_student}",
            height=560,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color=DARK_GREEN),
            margin=dict(l=10, r=10, t=60, b=90),
            xaxis_title="Event Date" if use_dates else "Event Sequence",
        )
        fig.update_xaxes(showgrid=True, gridcolor=GRID_GREEN)
        fig.update_yaxes(showgrid=True, gridcolor=GRID_GREEN)

        if use_dates and event_dt_list:
            rng = safe_datetime_range_with_padding(event_dt_list, pad_days=4)
            if rng:
                fig.update_xaxes(range=rng, autorange=False)

        if not use_dates:
            fig.update_xaxes(tickangle=-45)

        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 🧪 DATA QUALITY (diagnostics)
# ============================================================
with tab_quality:
    st.subheader("Parsing diagnostics (per selected sheet)")
    diag = []
    for sname, ctxi in contexts:
        diag.append({
            "Sheet": sname,
            "Header row": ctxi["meta"]["header_row"],
            "Date row": ctxi["meta"]["date_row"],
            "Event row": ctxi["meta"]["event_row"],
            "Detected events": len(ctxi["event_cols"]),
            "Has Batch col": bool(ctxi["batch_col"]),
            "Has Country col": bool(ctxi["country_col"]),
            "Has Payment col": bool(ctxi["payment_col"]),
            "Program inferred": infer_program_from_sheet_name(sname),
            "Batch inferred (if applied)": infer_batch_from_sheet_name(sname) if sname in BATCH_ALIGNED_SHEETS else "",
        })
    st.dataframe(pd.DataFrame(diag), use_container_width=True, height=320)

    st.subheader("Event date coverage (union)")
    if event_cols:
        cover = pd.DataFrame({
            "Event": event_cols,
            "Category": [classify_event_category(e) for e in event_cols],
            "Has date?": [pd.notna(event_dates.get(e, pd.NaT)) for e in event_cols],
            "Parsed date": [event_dates.get(e, pd.NaT) for e in event_cols],
            "Participants (filtered)": [safe_event_sum(fdf, e) for e in event_cols],
        }).sort_values(["Has date?", "Participants (filtered)"], ascending=[False, False])
        st.dataframe(cover, use_container_width=True, height=420)
    else:
        st.info("No event columns detected.")

    st.subheader("Notes")
    st.info(
        "Unique Attendees = students who attended ONLY ONE event category/type (AMA-only, Masterclass-only, Competition-only, etc). "
        "If you want to rename categories or adjust keywords, edit classify_event_category()."
    )
