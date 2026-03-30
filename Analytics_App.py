
import json
import re
from io import BytesIO
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except Exception:
    GSPREAD_AVAILABLE = False

st.set_page_config(page_title="Tetr Analytics Dashboard", layout="wide")

# -----------------------------
# Constants
# -----------------------------
MASTER_SHEETS = ["Master UG", "Master PG"]
UG_BATCH_SHEETS = ["UG - B1 to B4", "UG B5", "UG B6", "UG B7", "UG B8", "UG B9"]
PG_BATCH_SHEETS = ["PG - B1 & B2", "PG - B3 & B4", "PG B5"]
TETRX_SHEETS = ["Tetr-X-UG", "Tetr-X-PG"]
DETAIL_SHEETS = UG_BATCH_SHEETS + PG_BATCH_SHEETS + TETRX_SHEETS
ALL_REQUIRED = MASTER_SHEETS + DETAIL_SHEETS

GREEN = "#0b3d2e"
GREEN_2 = "#1f7a56"
GREEN_3 = "#56a77b"
GREEN_4 = "#9cd4b5"
GREEN_5 = "#dff3e7"
DARK = "#12372a"
LIGHT_BG = "#f7fbf8"
RED = "#d9534f"
AMBER = "#c17d11"

GSHEETS_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

# -----------------------------
# Styling
# -----------------------------
def inject_css():
    st.markdown(f"""
    <style>
    .stApp {{
        background: linear-gradient(180deg, #ffffff 0%, {LIGHT_BG} 100%);
    }}
    section[data-testid="stSidebar"] {{
        background: #eef8f2;
        border-right: 1px solid #d8eadf;
    }}
    .hero-card {{
        background: linear-gradient(135deg, #ffffff 0%, #eef8f2 100%);
        border: 1px solid #d8eadf;
        border-radius: 22px;
        padding: 22px 24px;
        box-shadow: 0 8px 24px rgba(11, 61, 46, 0.06);
        margin-bottom: 8px;
    }}
    .section-card {{
        background: #ffffff;
        border: 1px solid #e0eee5;
        border-radius: 18px;
        padding: 14px 16px;
        box-shadow: 0 4px 14px rgba(11, 61, 46, 0.04);
    }}
    .live-pill {{
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        border-radius: 999px;
        font-weight: 800;
        border: 1px solid #cfe8d9;
        color: {GREEN};
        background: #e8f6ed;
        white-space: nowrap;
    }}
    .live-pill.offline {{
        color: #7a1f1b;
        background: #fdeceb;
        border-color: #f3cdca;
    }}
    .heartbeat-wrap {{
        position: relative;
        width: 12px;
        height: 12px;
    }}
    .heartbeat-dot {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: #1bb55c;
        position: absolute;
        top: 0;
        left: 0;
        z-index: 2;
    }}
    .heartbeat-ping {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: rgba(27,181,92,0.30);
        position: absolute;
        top: 0;
        left: 0;
        animation: heartbeatPing 1.5s ease-out infinite;
        z-index: 1;
    }}
    .offline-dot {{
        width: 12px;
        height: 12px;
        border-radius: 50%;
        background: {RED};
    }}
    @keyframes heartbeatPing {{
        0% {{ transform: scale(0.9); opacity: 0.9; }}
        70% {{ transform: scale(2.2); opacity: 0; }}
        100% {{ transform: scale(2.2); opacity: 0; }}
    }}
    div[data-testid="stMetric"] {{
        background: #ffffff;
        border: 1px solid #dbeee0;
        border-radius: 16px;
        padding: 10px 12px;
        box-shadow: 0 2px 10px rgba(11, 61, 46, 0.05);
    }}
    div[data-testid="stMetric"] label {{
        color: {GREEN_2} !important;
        font-weight: 700 !important;
    }}
    h1, h2, h3, h4 {{ color: {DARK} !important; }}

    /* Full-width radio "tab bars" */
    section[data-testid="stSidebar"] div[role="radiogroup"] > label {{
        display: flex !important;
        width: 100% !important;
        margin: 0 0 10px 0 !important;
        padding: 11px 14px !important;
        border-radius: 12px !important;
        border: 1px solid #cfe5d7 !important;
        background: #dff3e7 !important;
        color: {GREEN} !important;
        font-weight: 700 !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        background: #cfeedd !important;
        border-color: #b8dcc7 !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label[data-selected="true"] {{
        background: linear-gradient(90deg, {GREEN} 0%, {GREEN_2} 100%) !important;
        color: white !important;
        border-color: {GREEN} !important;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label p {{
        color: inherit !important;
        font-weight: 700 !important;
        margin: 0 !important;
    }}
    </style>
    """, unsafe_allow_html=True)

inject_css()

# -----------------------------
# Utilities
# -----------------------------
def clean_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").replace("\xa0", " ").strip()

def normalize_name(x: str) -> str:
    s = clean_text(x).lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def make_unique(cols):
    seen = {}
    out = []
    for c in cols:
        c = clean_text(c) or "Unnamed"
        if c not in seen:
            seen[c] = 0
            out.append(c)
        else:
            seen[c] += 1
            out.append(f"{c}_{seen[c]}")
    return out

def parse_date_safe(x):
    if x is None:
        return pd.NaT
    if isinstance(x, pd.Timestamp):
        return x
    try:
        if isinstance(x, (int, float)) and not pd.isna(x):
            if 30000 <= float(x) <= 60000:
                return pd.Timestamp("1899-12-30") + pd.to_timedelta(float(x), unit="D")
        s = clean_text(x)
        if not s:
            return pd.NaT
        # ranges like 28-01 to 30.01.2026 -> use start date
        m = re.search(r"(\d{{1,2}})[\-.](\d{{1,2}}).*?(\d{{4}})", s)
        if m:
            dd, mm, yyyy = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return pd.Timestamp(year=yyyy, month=mm, day=dd)
        return pd.to_datetime(s, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT

def normalize_yes_no(x) -> int:
    s = clean_text(x).lower()
    return 1 if s in {"yes", "y", "1", "true", "present", "attended", "done"} else 0

def percent_series(s):
    out = pd.to_numeric(s, errors="coerce").fillna(0)
    if len(out) and out.max() <= 1.05:
        out = out * 100
    return out

def best_matching_col(df: pd.DataFrame, candidates):
    lowered = {c: clean_text(c).lower() for c in df.columns}
    for cand in candidates:
        for col, low in lowered.items():
            if cand in low:
                return col
    return None

def infer_program(sheet_name: str) -> str:
    s = clean_text(sheet_name).lower()
    return "UG" if "ug" in s else ("PG" if "pg" in s else "")

def infer_batch_label(sheet_name: str) -> str:
    s = clean_text(sheet_name).lower().replace("–", "-").replace("—", "-")
    m = re.search(r"b\s*(\d+)\s*to\s*b\s*(\d+)", s)
    if m:
        return f"B{m.group(1)}–B{m.group(2)}"
    m = re.search(r"b\s*(\d+)\s*&\s*b\s*(\d+)", s)
    if m:
        return f"B{m.group(1)}–B{m.group(2)}"
    m = re.search(r"\bb\s*(\d+)\b", s)
    if m:
        return f"B{m.group(1)}"
    return ""

def live_status_html(is_connected: bool, mode_label: str):
    if is_connected:
        return f"""
        <div class="live-pill">
            <span class="heartbeat-wrap"><span class="heartbeat-ping"></span><span class="heartbeat-dot"></span></span>
            LIVE · {mode_label}
        </div>"""
    return f"""
    <div class="live-pill offline">
        <span class="offline-dot"></span>
        OFFLINE · {mode_label}
    </div>"""

def nice_layout(fig, height=360, x_tickangle=None):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=DARK),
        title_font=dict(color=DARK),
        margin=dict(l=20, r=20, t=60, b=40),
        height=height,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e7f2eb", tickangle=x_tickangle)
    fig.update_yaxes(showgrid=True, gridcolor="#e7f2eb")
    return fig

def chart_key(page, name):
    return f"{page}__{name}"

# -----------------------------
# Data loading
# -----------------------------
def get_service_account_info():
    if "GOOGLE_SERVICE_ACCOUNT" not in st.secrets:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT in Streamlit secrets.")
    return dict(st.secrets["GOOGLE_SERVICE_ACCOUNT"])

def get_spreadsheet_id():
    return st.secrets.get("GSHEET_SPREADSHEET_ID", "")

def _get_gsheets_client():
    info = get_service_account_info()
    creds = Credentials.from_service_account_info(info, scopes=GSHEETS_SCOPES)
    return gspread.authorize(creds)

@st.cache_data(show_spinner=False, ttl=180)
def gsheets_get_sheet_names(spreadsheet_id: str):
    gc = _get_gsheets_client()
    sh = gc.open_by_key(spreadsheet_id)
    return [ws.title for ws in sh.worksheets()]

@st.cache_data(show_spinner=False, ttl=180)
def gsheets_read_raw_sheet(spreadsheet_id: str, sheet_name: str):
    gc = _get_gsheets_client()
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(sheet_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    max_len = max(len(r) for r in values)
    values = [r + [""] * (max_len - len(r)) for r in values]
    df = pd.DataFrame(values)
    df.replace("", np.nan, inplace=True)
    return df

# -----------------------------
# Sheet parsers
# -----------------------------
def parse_master_sheet(raw: pd.DataFrame, sheet_name: str):
    # Actual structure from workbook:
    # row 0 = header, rows 1-2 = summary, rows 3+ = students
    headers = make_unique(raw.iloc[0].tolist())
    df = raw.iloc[3:].copy().reset_index(drop=True)
    df.columns = headers
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["name"])
    email_col = best_matching_col(df, ["email"])
    batch_col = best_matching_col(df, ["batch"])
    country_col = best_matching_col(df, ["country"])
    income_col = best_matching_col(df, ["income"])
    status_col = best_matching_col(df, ["status"])
    payment_col = best_matching_col(df, ["payment"])
    admitted_group_col = best_matching_col(df, ["admitted group"])
    term_zero_col = best_matching_col(df, ["term zero"])
    contact_col = best_matching_col(df, ["contact"])

    if not name_col:
        raise ValueError(f"Name column not found in {sheet_name}")

    df = df[df[name_col].astype(str).str.strip().ne("")].copy()
    df["Program"] = infer_program(sheet_name)
    df["Sheet"] = sheet_name
    df["Batch Label"] = df[batch_col].astype(str).str.strip() if batch_col else ""
    df["_name_norm"] = df[name_col].apply(normalize_name)
    df["_email_norm"] = df[email_col].astype(str).str.strip().str.lower() if email_col else ""

    # aggregate engagement columns from master
    score_col = best_matching_col(df, ["engagement", "score"])
    pct_col = best_matching_col(df, ["engagement"])
    overall_score_col = best_matching_col(df, ["engagement"])
    if "Overall Engagement Score" in df.columns:
        df["engagement_score"] = pd.to_numeric(df["Overall Engagement Score"], errors="coerce").fillna(0)
    elif score_col:
        df["engagement_score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0)
    else:
        df["engagement_score"] = 0

    if "Overall Engagement %" in df.columns:
        df["engagement_pct"] = percent_series(df["Overall Engagement %"])
    elif pct_col:
        df["engagement_pct"] = percent_series(df[pct_col])
    else:
        df["engagement_pct"] = 0

    raw_payment = df[payment_col].astype(str).str.lower() if payment_col else pd.Series("", index=df.index)
    raw_status = df[status_col].astype(str).str.lower() if status_col else pd.Series("", index=df.index)
    raw_group = df[admitted_group_col].astype(str).str.lower() if admitted_group_col else pd.Series("", index=df.index)

    df["raw_status_text"] = (
        raw_payment.fillna("") + " | " + raw_status.fillna("") + " | " + raw_group.fillna("")
    ).str.strip(" |")
    df["raw_paid_flag"] = df["raw_status_text"].str.contains("paid|admitted", case=False, na=False)
    df["raw_refunded_flag"] = df["raw_status_text"].str.contains("refund", case=False, na=False)
    df["is_active"] = df["engagement_pct"] > 0

    ctx = {
        "name_col": name_col,
        "email_col": email_col,
        "batch_col": batch_col,
        "country_col": country_col,
        "income_col": income_col,
        "status_col": status_col,
        "payment_col": payment_col,
        "admitted_group_col": admitted_group_col,
        "term_zero_col": term_zero_col,
        "contact_col": contact_col,
    }
    return df, ctx

def parse_activity_sheet(raw: pd.DataFrame, sheet_name: str):
    # Actual structure from workbook:
    # row 0 type | row 1 event name | row 2 event date | row 3 day | row 4 done | row 5 header | row 6+ data
    header_row = 5
    data_start = 6
    headers = [clean_text(x) for x in raw.iloc[header_row].tolist()]
    headers = make_unique(headers)

    # determine where metadata columns end
    non_empty_header_idxs = [i for i, h in enumerate(headers) if clean_text(h)]
    last_meta_idx = max(non_empty_header_idxs) if non_empty_header_idxs else 0

    event_names = []
    event_types = []
    event_dates = []
    for idx in range(last_meta_idx + 1, raw.shape[1]):
        type_val = clean_text(raw.iloc[0, idx]) if idx < raw.shape[1] else ""
        name_val = clean_text(raw.iloc[1, idx]) if idx < raw.shape[1] else ""
        date_val = parse_date_safe(raw.iloc[2, idx]) if idx < raw.shape[1] else pd.NaT
        done_val = clean_text(raw.iloc[4, idx]) if idx < raw.shape[1] else ""
        if not (type_val or name_val or pd.notna(date_val) or done_val):
            continue
        event_name = name_val or f"Event {idx - last_meta_idx}"
        event_names.append((idx, event_name))
        event_types.append((idx, type_val or "Other"))
        event_dates.append((idx, date_val))

    col_names = []
    event_info_rows = []
    event_idx_to_name = {}
    for idx in range(raw.shape[1]):
        if idx <= last_meta_idx:
            col_names.append(headers[idx] if idx < len(headers) else f"Col_{idx}")
        else:
            found = next((nm for j, nm in event_names if j == idx), None)
            if found:
                dt = next((d for j, d in event_dates if j == idx), pd.NaT)
                unique_name = found
                if sum(1 for _, n in event_names if n == found) > 1 and pd.notna(dt):
                    unique_name = f"{found} ({pd.to_datetime(dt).strftime('%d %b %Y')})"
                col_names.append(unique_name)
                event_idx_to_name[idx] = unique_name
                event_info_rows.append({
                    "column_name": unique_name,
                    "event_name": found,
                    "event_type": next((t for j, t in event_types if j == idx), "Other"),
                    "event_date": next((d for j, d in event_dates if j == idx), pd.NaT),
                    "sheet": sheet_name,
                })
            else:
                col_names.append(f"Unused_{idx}")

    col_names = make_unique(col_names)
    # sync unique names back to event info
    for i, idx in enumerate(sorted(event_idx_to_name)):
        event_info_rows[i]["column_name"] = col_names[idx]

    df = raw.iloc[data_start:].copy().reset_index(drop=True)
    df.columns = col_names
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["student name", "name"])
    email_col = best_matching_col(df, ["email"])
    batch_col = best_matching_col(df, ["batch"])
    country_col = best_matching_col(df, ["country"])
    income_col = best_matching_col(df, ["income"])
    status_col = best_matching_col(df, ["status"])
    payment_status_col = best_matching_col(df, ["payment status"])
    payment_date_col = best_matching_col(df, ["payment date"])
    comments_col = best_matching_col(df, ["comments"])

    if not name_col:
        raise ValueError(f"Name column not found in {sheet_name}")

    df = df[df[name_col].astype(str).str.strip().ne("")].copy()
    df["Program"] = infer_program(sheet_name)
    df["Sheet"] = sheet_name
    if batch_col:
        df["Batch Label"] = df[batch_col].astype(str).str.strip()
    else:
        df["Batch Label"] = infer_batch_label(sheet_name)
    df["_name_norm"] = df[name_col].apply(normalize_name)
    df["_email_norm"] = df[email_col].astype(str).str.strip().str.lower() if email_col else ""

    # Event columns
    event_info = pd.DataFrame(
        event_info_rows,
        columns=["column_name", "event_name", "event_type", "event_date", "sheet"],
    )
    event_cols = event_info["column_name"].tolist() if not event_info.empty else []
    event_cols = [c for c in event_cols if c in df.columns]
    for c in event_cols:
        df[c] = df[c].apply(normalize_yes_no).astype(int)

    if "Overall Engagement Score" in df.columns:
        df["engagement_score"] = pd.to_numeric(df["Overall Engagement Score"], errors="coerce").fillna(0)
    else:
        df["engagement_score"] = df[event_cols].sum(axis=1) if event_cols else 0

    if "Overall Engagement %" in df.columns:
        df["engagement_pct"] = percent_series(df["Overall Engagement %"])
    else:
        total_events = max(len(event_cols), 1)
        df["engagement_pct"] = (df[event_cols].sum(axis=1) / total_events) * 100 if event_cols else 0

    raw_status_parts = []
    for c in [status_col, payment_status_col, comments_col]:
        if c and c in df.columns:
            raw_status_parts.append(df[c].astype(str).str.lower())
    df["raw_status_text"] = " | ".join([]) if not raw_status_parts else raw_status_parts[0]
    if len(raw_status_parts) > 1:
        tmp = raw_status_parts[0]
        for s in raw_status_parts[1:]:
            tmp = tmp.fillna("") + " | " + s.fillna("")
        df["raw_status_text"] = tmp
    df["raw_paid_flag"] = df["raw_status_text"].str.contains("paid|admitted", case=False, na=False)
    df["raw_refunded_flag"] = df["raw_status_text"].str.contains("refund", case=False, na=False)
    df["is_active"] = df["engagement_pct"] > 0

    if payment_date_col and payment_date_col in df.columns:
        df["payment_date_local"] = df[payment_date_col].apply(parse_date_safe)
    else:
        df["payment_date_local"] = pd.NaT

    df["participation_count"] = df[event_cols].sum(axis=1) if event_cols else 0

    ctx = {
        "name_col": name_col,
        "email_col": email_col,
        "batch_col": batch_col,
        "country_col": country_col,
        "income_col": income_col,
        "status_col": status_col,
        "payment_status_col": payment_status_col,
        "payment_date_col": payment_date_col,
        "comments_col": comments_col,
        "event_cols": event_cols,
        "event_info": event_info,
    }
    return df, ctx

# -----------------------------
# Reconciliation with Tetr-X
# -----------------------------
def build_lookup(df, name_col="Student Name", email_col="Email"):
    email_map = {}
    name_map = {}
    for idx, row in df.iterrows():
        e = clean_text(row.get("_email_norm", ""))
        n = clean_text(row.get("_name_norm", ""))
        if e and e not in email_map:
            email_map[e] = idx
        if n and n not in name_map:
            name_map[n] = idx
    return email_map, name_map

def get_final_status_fields(row, matched_tetrx_row=None):
    raw = clean_text(row.get("raw_status_text", "")).lower()
    if matched_tetrx_row is not None:
        tx_status = clean_text(matched_tetrx_row.get("Status", matched_tetrx_row.get("raw_status_text", ""))).lower()
        tx_term = clean_text(matched_tetrx_row.get("Tetr X/Term 0 Status", "")).lower()
        combined = " | ".join([t for t in [tx_status, tx_term, raw] if t]).strip(" |")
        pay_dt = matched_tetrx_row.get("payment_date_final", pd.NaT)
    else:
        combined = raw
        pay_dt = row.get("payment_date_local", pd.NaT)

    refunded = "refund" in combined
    paid = (("admitted" in combined) or ("paid" in combined) or pd.notna(pay_dt)) and not refunded

    if refunded:
        label = "Refunded"
    elif paid:
        label = "Paid / Admitted"
    else:
        label = "Not Paid"
    return paid, refunded, label, pay_dt, combined

def reconcile_with_tetrx(df, tetrx_df):
    if df.empty:
        return df
    df = df.copy()
    tx_email_map, tx_name_map = build_lookup(tetrx_df)
    matched_indices = []
    paid_flags = []
    refunded_flags = []
    labels = []
    pay_dates = []
    combined_status = []

    for _, row in df.iterrows():
        e = clean_text(row.get("_email_norm", ""))
        n = clean_text(row.get("_name_norm", ""))
        tx_idx = None
        if e and e in tx_email_map:
            tx_idx = tx_email_map[e]
        elif n and n in tx_name_map:
            tx_idx = tx_name_map[n]
        matched_indices.append(tx_idx)
        tx_row = tetrx_df.loc[tx_idx] if tx_idx is not None else None
        paid, refunded, label, pay_dt, combo = get_final_status_fields(row, tx_row)
        paid_flags.append(paid)
        refunded_flags.append(refunded)
        labels.append(label)
        pay_dates.append(pay_dt)
        combined_status.append(combo)

    df["matched_tetrx_idx"] = matched_indices
    df["is_paid"] = paid_flags
    df["is_refunded"] = refunded_flags
    df["paid_label"] = labels
    df["payment_date_final"] = pay_dates
    df["status_final"] = combined_status
    return df

# -----------------------------
# Dashboard loading
# -----------------------------
@st.cache_data(show_spinner=False, ttl=180)
def load_dashboard_data(spreadsheet_id: str):
    sheet_names = gsheets_get_sheet_names(spreadsheet_id)
    missing = [s for s in ALL_REQUIRED if s not in sheet_names]

    masters = {}
    master_contexts = {}
    details = {}
    detail_contexts = {}

    for sheet in MASTER_SHEETS:
        if sheet in sheet_names:
            raw = gsheets_read_raw_sheet(spreadsheet_id, sheet)
            masters[sheet], master_contexts[sheet] = parse_master_sheet(raw, sheet)

    for sheet in DETAIL_SHEETS:
        if sheet in sheet_names:
            raw = gsheets_read_raw_sheet(spreadsheet_id, sheet)
            details[sheet], detail_contexts[sheet] = parse_activity_sheet(raw, sheet)

    # Tetr-X as source of truth for paid/admitted and payment date
    tetrx_lookup = {
        "UG": details.get("Tetr-X-UG", pd.DataFrame()),
        "PG": details.get("Tetr-X-PG", pd.DataFrame()),
    }

    # payment date final on tetrx
    for prog, tx in tetrx_lookup.items():
        if not tx.empty:
            if "payment_date_local" in tx.columns:
                tx["payment_date_final"] = tx["payment_date_local"]
            else:
                tx["payment_date_final"] = pd.NaT
            details[f"Tetr-X-{prog}"] = tx

    # Reconcile
    for sheet, df in list(masters.items()):
        prog = "UG" if "UG" in sheet else "PG"
        masters[sheet] = reconcile_with_tetrx(df, tetrx_lookup.get(prog, pd.DataFrame()))
        # fallback if still unmatched
        if "is_paid" not in masters[sheet].columns:
            masters[sheet]["is_paid"] = masters[sheet]["raw_paid_flag"]
            masters[sheet]["is_refunded"] = masters[sheet]["raw_refunded_flag"]
            masters[sheet]["paid_label"] = np.where(masters[sheet]["is_refunded"], "Refunded", np.where(masters[sheet]["is_paid"], "Paid / Admitted", "Not Paid"))
            masters[sheet]["payment_date_final"] = pd.NaT
            masters[sheet]["status_final"] = masters[sheet]["raw_status_text"]

    for sheet, df in list(details.items()):
        prog = "UG" if "UG" in sheet else "PG"
        details[sheet] = reconcile_with_tetrx(df, tetrx_lookup.get(prog, pd.DataFrame()))

    overview_df = pd.concat([masters[s] for s in MASTER_SHEETS if s in masters], ignore_index=True) if masters else pd.DataFrame()

    # unified student universe for profile
    student_rows = []
    for sheet, df in masters.items():
        ctx = master_contexts[sheet]
        for _, row in df.iterrows():
            student_rows.append({
                "name": row[ctx["name_col"]],
                "email": row.get("_email_norm", ""),
                "program": row["Program"],
                "sheet": sheet,
                "source_type": "master",
                "name_norm": row["_name_norm"],
            })
    all_students = pd.DataFrame(student_rows).drop_duplicates(subset=["email", "name_norm", "program"])

    return {
        "sheet_names": sheet_names,
        "missing": missing,
        "masters": masters,
        "master_contexts": master_contexts,
        "details": details,
        "detail_contexts": detail_contexts,
        "overview_df": overview_df,
        "all_students": all_students,
    }

# -----------------------------
# Charts
# -----------------------------
def gauge_chart(value, title, maximum=None, suffix=""):
    maximum = maximum or max(value, 1)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"suffix": suffix},
        title={"text": title},
        gauge={
            "axis": {"range": [0, maximum]},
            "bar": {"color": GREEN},
            "bgcolor": "white",
            "steps": [
                {"range": [0, maximum * 0.5], "color": GREEN_5},
                {"range": [maximum * 0.5, maximum * 0.8], "color": GREEN_4},
                {"range": [maximum * 0.8, maximum], "color": GREEN_3},
            ],
        },
    ))
    return nice_layout(fig, height=300)

def donut_chart(labels, values, title, color_map=None):
    colors = None
    if color_map:
        colors = [color_map.get(lbl, GREEN_3) for lbl in labels]
    else:
        colors = [GREEN, GREEN_2, GREEN_3, GREEN_4, GREEN_5][:len(labels)]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, marker=dict(colors=colors), textinfo="label+percent"))
    fig.update_layout(title=title)
    return nice_layout(fig, height=340)

# -----------------------------
# Rendering
# -----------------------------
def render_header(is_connected: bool):
    col1, col2 = st.columns([8, 2])
    with col1:
        logo_path = None
        for candidate in ["logo", "logo.png", "logo.jpg", "logo.jpeg", "logo.webp"]:
            p = Path(candidate)
            if p.exists():
                logo_path = str(p)
                break
        logo_html = ""
        if logo_path and Path(logo_path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            st.markdown('<div class="hero-card">', unsafe_allow_html=True)
            c1, c2 = st.columns([1, 9])
            with c1:
                st.image(logo_path, width=70)
            with c2:
                st.markdown("## Tetr Analytics Dashboard")
                st.caption("Live business-school engagement, admissions, payment and student profile analytics")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="hero-card">
                <div style="font-size:28px;font-weight:900;color:{GREEN};">Tetr Analytics Dashboard</div>
                <div style="margin-top:4px;color:{GREEN_2};font-weight:600;">
                    Live business-school engagement, admissions, payment and student profile analytics
                </div>
            </div>
            """, unsafe_allow_html=True)
    with col2:
        st.markdown(live_status_html(is_connected, "Google Sheets"), unsafe_allow_html=True)

def render_overview(data):
    st.subheader("Overview")
    overview_df = data["overview_df"]
    ctx = data["master_contexts"]["Master UG"] if "Master UG" in data["master_contexts"] else list(data["master_contexts"].values())[0]
    if overview_df.empty:
        st.warning("Overview data unavailable.")
        return

    name_col = ctx["name_col"]
    country_col = ctx["country_col"]
    batch_col = "Batch Label"
    income_col = ctx["income_col"]

    total_students = int(overview_df[name_col].count())
    total_active = int(overview_df["is_active"].sum())
    total_paid = int((overview_df["paid_label"] == "Paid / Admitted").sum())
    total_refunded = int((overview_df["paid_label"] == "Refunded").sum())
    ug_students = int((overview_df["Program"] == "UG").sum())
    pg_students = int((overview_df["Program"] == "PG").sum())
    ug_paid = int(((overview_df["Program"] == "UG") & (overview_df["paid_label"] == "Paid / Admitted")).sum())
    pg_paid = int(((overview_df["Program"] == "PG") & (overview_df["paid_label"] == "Paid / Admitted")).sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Students", f"{total_students:,}")
    c2.metric("Active Students", f"{total_active:,}", delta=f"{(total_active/total_students*100 if total_students else 0):.1f}%")
    c3.metric("Paid / Admitted", f"{total_paid:,}", delta=f"{(total_paid/total_students*100 if total_students else 0):.1f}%")
    c4.metric("Refunded", f"{total_refunded:,}")
    c5.metric("Avg Engagement", f"{overview_df['engagement_pct'].mean():.1f}%")

    g1, g2, g3 = st.columns([1.2, 1, 1])
    with g1:
        st.plotly_chart(gauge_chart(total_students, "Total Students", max(total_students, 1)), use_container_width=True, key=chart_key("overview", "gauge_total"))
    with g2:
        st.plotly_chart(donut_chart(["UG", "PG"], [ug_students, pg_students], "UG / PG Distribution"), use_container_width=True, key=chart_key("overview", "ugpg_donut"))
    with g3:
        st.plotly_chart(donut_chart(["UG Paid", "PG Paid"], [ug_paid, pg_paid], "Paid Distribution"), use_container_width=True, key=chart_key("overview", "paid_donut"))

    a1, a2 = st.columns(2)
    with a1:
        batch_plot = (overview_df.groupby(batch_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False))
        fig = px.bar(batch_plot, x=batch_col, y="Students", title="Students by Batch")
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-25), use_container_width=True, key=chart_key("overview", "batch_bar"))
    with a2:
        status_plot = overview_df.groupby(["Program", "paid_label"])[name_col].count().reset_index(name="Students")
        fig = px.bar(
            status_plot, x="Program", y="Students", color="paid_label", barmode="group", title="Paid vs Refunded vs Not Paid",
            color_discrete_map={"Paid / Admitted": GREEN, "Refunded": AMBER, "Not Paid": GREEN_4}
        )
        st.plotly_chart(nice_layout(fig, height=380), use_container_width=True, key=chart_key("overview", "status_bar"))

    b1, b2 = st.columns(2)
    with b1:
        if country_col and country_col in overview_df.columns:
            country_plot = overview_df.groupby(country_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False).head(12)
            fig = px.bar(country_plot, x=country_col, y="Students", title="Top Countries")
            fig.update_traces(marker_color=GREEN_3)
            st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-30), use_container_width=True, key=chart_key("overview", "country_bar"))
    with b2:
        if income_col and income_col in overview_df.columns:
            income_plot = overview_df.groupby(income_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False)
            fig = px.bar(income_plot, x=income_col, y="Students", title="Income Distribution")
            fig.update_traces(marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-25), use_container_width=True, key=chart_key("overview", "income_bar"))

    st.markdown("#### Student Table")
    preview_cols = [c for c in [name_col, "Program", batch_col, country_col, income_col, "engagement_pct", "engagement_score", "paid_label", "payment_date_final"] if c in overview_df.columns]
    st.dataframe(overview_df[preview_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False), use_container_width=True, height=420, key=chart_key("overview", "df"))

def render_sheet_page(sheet_name, df, ctx):
    st.subheader(sheet_name)
    if df.empty:
        st.warning("No data available.")
        return

    name_col = ctx["name_col"]
    country_col = ctx["country_col"]
    event_cols = ctx["event_cols"] if "event_cols" in ctx else []
    event_info = ctx.get("event_info", pd.DataFrame())

    total_students = int(df[name_col].count())
    active_students = int(df["is_active"].sum())
    paid_students = int((df["paid_label"] == "Paid / Admitted").sum())
    refunded_students = int((df["paid_label"] == "Refunded").sum())
    avg_engagement = round(float(df["engagement_pct"].mean()), 1) if len(df) else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Students", f"{total_students:,}")
    k2.metric("Active", f"{active_students:,}", delta=f"{(active_students/total_students*100 if total_students else 0):.1f}%")
    k3.metric("Paid / Admitted", f"{paid_students:,}")
    k4.metric("Refunded", f"{refunded_students:,}")
    k5.metric("Avg Engagement", f"{avg_engagement:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="engagement_pct", nbins=12, title="Engagement Distribution")
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=chart_key(sheet_name, "engagement_hist"))
    with c2:
        status = df["paid_label"].value_counts().reset_index()
        status.columns = ["Status", "Students"]
        fig = px.pie(
            status, names="Status", values="Students", hole=0.58, title="Student Status",
            color="Status", color_discrete_map={"Paid / Admitted": GREEN, "Refunded": AMBER, "Not Paid": GREEN_4}
        )
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=chart_key(sheet_name, "status_pie"))

    d1, d2 = st.columns(2)
    with d1:
        if event_cols:
            event_counts = []
            for c in event_cols:
                count = int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum())
                meta = event_info[event_info["column_name"] == c]
                ev_type = meta["event_type"].iloc[0] if len(meta) else "Other"
                event_counts.append({"Event": c, "Participants": count, "Type": ev_type})
            event_counts = pd.DataFrame(event_counts).sort_values("Participants", ascending=False).head(12)
            fig = px.bar(event_counts, x="Participants", y="Event", orientation="h", color="Type", title="Top Events by Participation")
            st.plotly_chart(nice_layout(fig, height=460), use_container_width=True, key=chart_key(sheet_name, "event_bar"))
    with d2:
        if country_col and country_col in df.columns:
            country_plot = df.groupby(country_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False).head(10)
            fig = px.bar(country_plot, x=country_col, y="Students", title="Country Split")
            fig.update_traces(marker_color=GREEN_3)
            st.plotly_chart(nice_layout(fig, height=430, x_tickangle=-30), use_container_width=True, key=chart_key(sheet_name, "country_bar"))

    e1, e2 = st.columns(2)
    with e1:
        if event_cols and not event_info.empty:
            type_counts = []
            for ev_type, sub in event_info.groupby("event_type"):
                cols = [c for c in sub["column_name"].tolist() if c in df.columns]
                type_counts.append({"Event Type": ev_type or "Other", "Participations": int(df[cols].sum().sum()) if cols else 0})
            type_df = pd.DataFrame(type_counts).sort_values("Participations", ascending=False)
            fig = px.bar(type_df, x="Event Type", y="Participations", title="Participation by Event Type")
            fig.update_traces(marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-20), use_container_width=True, key=chart_key(sheet_name, "type_bar"))
    with e2:
        if event_cols and not event_info.empty:
            timeline_rows = []
            for _, meta in event_info.iterrows():
                c = meta["column_name"]
                if c in df.columns and pd.notna(meta["event_date"]):
                    timeline_rows.append({"Date": pd.to_datetime(meta["event_date"]), "Participants": int(df[c].sum())})
            timeline = pd.DataFrame(timeline_rows).groupby("Date", as_index=False)["Participants"].sum().sort_values("Date") if timeline_rows else pd.DataFrame()
            if not timeline.empty:
                fig = px.line(timeline, x="Date", y="Participants", markers=True, title="Participation Timeline")
                fig.update_traces(line_color=GREEN, marker_color=GREEN)
                st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=chart_key(sheet_name, "timeline"))

    st.markdown("#### Top Students")
    top_cols = [c for c in [name_col, "Batch Label", "engagement_pct", "engagement_score", "paid_label", "payment_date_final"] if c in df.columns]
    st.dataframe(df[top_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False).head(30), use_container_width=True, height=360, key=chart_key(sheet_name, "top_df"))

    st.markdown("#### Full Student Table")
    display_cols = [c for c in [name_col, "Batch Label", country_col, "engagement_pct", "engagement_score", "paid_label", "payment_date_final"] if c in df.columns]
    preview_events = event_cols[:8] if event_cols else []
    display_cols += [c for c in preview_events if c in df.columns]
    st.dataframe(df[display_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False), use_container_width=True, height=420, key=chart_key(sheet_name, "full_df"))

def gather_student_matches(data, query_names, program_filter="All"):
    wanted = {normalize_name(n) for n in query_names if normalize_name(n)}
    if not wanted:
        return []
    out = []
    seen = set()
    for sheet, df in data["masters"].items():
        ctx = data["master_contexts"][sheet]
        for _, row in df.iterrows():
            nm = row["_name_norm"]
            if nm in wanted and (program_filter == "All" or row["Program"] == program_filter):
                key = (row.get("_email_norm", ""), row["_name_norm"], row["Program"])
                if key not in seen:
                    out.append(key)
                    seen.add(key)
    return out

def render_profile_card(title, info_dict):
    st.markdown(f"""
    <div class="section-card">
      <div style="font-size:18px;font-weight:800;color:{GREEN};margin-bottom:8px;">{title}</div>
    """, unsafe_allow_html=True)
    cols = st.columns(3)
    items = list(info_dict.items())
    for i, (k, v) in enumerate(items):
        with cols[i % 3]:
            st.markdown(f"**{k}**  \n{v if clean_text(v) else '-'}")
    st.markdown("</div>", unsafe_allow_html=True)

def render_student_profile(data):
    st.subheader("Student Profile")
    all_students = data["all_students"]
    master_names = sorted(all_students["name"].dropna().astype(str).unique().tolist())

    st.caption("Search one or more students. The dashboard will combine Master, Batch and Tetr-X records, then show participation, event types and timeline.")

    c1, c2 = st.columns([2, 1])
    with c1:
        text_input = st.text_area("Paste student names", placeholder="One name per line or comma-separated", height=100)
    with c2:
        chosen_names = st.multiselect("Or select students", options=master_names)

    program_filter = st.selectbox("Program", ["All", "UG", "PG"], index=0)

    parsed_text_names = []
    for part in re.split(r"[\n,;]+", text_input or ""):
        if clean_text(part):
            parsed_text_names.append(clean_text(part))
    queries = list(dict.fromkeys(chosen_names + parsed_text_names))

    if not queries:
        st.info("Select or paste one or more student names to view profiles.")
        return

    matched_keys = gather_student_matches(data, queries, program_filter)
    if not matched_keys:
        st.warning("No matching students found in Master sheets.")
        return

    for email_norm, name_norm, prog in matched_keys:
        master_df = data["masters"]["Master UG" if prog == "UG" else "Master PG"]
        master_row = master_df[(master_df["_name_norm"] == name_norm)]
        if email_norm:
            master_row = master_row[(master_row["_email_norm"] == email_norm)] if not master_row.empty else master_df[(master_df["_email_norm"] == email_norm)]
        if master_row.empty:
            continue
        master_row = master_row.iloc[0]
        master_ctx = data["master_contexts"]["Master UG" if prog == "UG" else "Master PG"]

        display_name = master_row[master_ctx["name_col"]]
        st.markdown(f"### {display_name}")

        # profile summary
        summary = {
            "Program": prog,
            "Email": master_row.get(master_ctx["email_col"], ""),
            "Batch": master_row.get("Batch Label", ""),
            "Country": master_row.get(master_ctx["country_col"], ""),
            "Income": master_row.get(master_ctx["income_col"], ""),
            "Paid Status": master_row.get("paid_label", ""),
            "Payment Date": master_row.get("payment_date_final", ""),
            "Engagement %": f"{master_row.get('engagement_pct', 0):.1f}%",
            "Engagement Score": int(master_row.get("engagement_score", 0)),
        }
        render_profile_card("Master Profile", summary)

        # collect matches across detail sheets
        detail_rows = []
        event_records = []
        type_counter = Counter()
        sheet_summary = []
        total_events = 0

        for sheet in DETAIL_SHEETS:
            if sheet not in data["details"]:
                continue
            df = data["details"][sheet]
            ctx = data["detail_contexts"][sheet]
            sub = df[df["_name_norm"] == name_norm]
            if email_norm:
                sub = sub[(sub["_email_norm"] == email_norm)] if not sub.empty else df[df["_email_norm"] == email_norm]
            if sub.empty:
                continue
            row = sub.iloc[0]
            detail_rows.append((sheet, row, ctx))
            sheet_summary.append({
                "Sheet": sheet,
                "Batch": row.get("Batch Label", ""),
                "Paid Status": row.get("paid_label", ""),
                "Engagement %": row.get("engagement_pct", 0),
                "Events Attended": int(row.get("participation_count", 0)),
                "Payment Date": row.get("payment_date_final", pd.NaT),
            })

            event_info = ctx.get("event_info", pd.DataFrame())
            for _, meta in event_info.iterrows():
                col = meta["column_name"]
                if col in sub.columns and int(row.get(col, 0)) == 1:
                    total_events += 1
                    ev_type = clean_text(meta["event_type"]) or "Other"
                    type_counter[ev_type] += 1
                    event_records.append({
                        "Date": pd.to_datetime(meta["event_date"]) if pd.notna(meta["event_date"]) else pd.NaT,
                        "Event": clean_text(meta["event_name"]) or col,
                        "Type": ev_type,
                        "Sheet": sheet,
                    })

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Participations", total_events)
        k2.metric("Matched Sheets", len(detail_rows))
        k3.metric("Current Paid Status", master_row.get("paid_label", ""))
        k4.metric("Master Engagement", f"{master_row.get('engagement_pct', 0):.1f}%")

        if sheet_summary:
            st.markdown("#### Sheet Matches")
            st.dataframe(pd.DataFrame(sheet_summary).sort_values(["Sheet"]), use_container_width=True, height=220, key=chart_key(display_name, "sheet_matches"))

        if event_records:
            event_df = pd.DataFrame(event_records).sort_values(["Date", "Event"])
            type_df = pd.DataFrame([
                {"Event Type": t, "Count": c, "Percent": round(c / total_events * 100, 1) if total_events else 0}
                for t, c in type_counter.items()
            ]).sort_values("Count", ascending=False)

            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(type_df, x="Event Type", y="Count", text="Percent", title="Event Types Attended")
                fig.update_traces(marker_color=GREEN)
                st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-20), use_container_width=True, key=chart_key(display_name, "type_counts"))
            with c2:
                event_date_counts = event_df.dropna(subset=["Date"]).groupby("Date", as_index=False).size().rename(columns={"size": "Participations"})
                fig = px.line(event_date_counts, x="Date", y="Participations", markers=True, title="Engagement Timeline")
                fig.update_traces(line_color=GREEN, marker_color=GREEN)
                pay_dt = master_row.get("payment_date_final", pd.NaT)
                if pd.notna(pay_dt):
                    xdt = pd.to_datetime(pay_dt)
                    fig.add_shape(
                        type="line",
                        x0=xdt, x1=xdt, y0=0, y1=1, yref="paper",
                        line=dict(color=RED, dash="dash", width=2)
                    )
                    fig.add_annotation(x=xdt, y=1, yref="paper", text="Payment Date", showarrow=False, yshift=10, font=dict(color=RED))
                st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=chart_key(display_name, "timeline"))

            st.markdown("#### Event Type Breakdown")
            st.dataframe(type_df, use_container_width=True, height=220, key=chart_key(display_name, "type_df"))

            st.markdown("#### Event History")
            st.dataframe(event_df, use_container_width=True, height=320, key=chart_key(display_name, "event_df"))
        else:
            st.info("No batch or Tetr-X participation records found for this student.")

        st.markdown("---")

# -----------------------------
# Sidebar / main
# -----------------------------
def sidebar_navigation():
    nav_options = ["Overview", "Student Profile"] + UG_BATCH_SHEETS + PG_BATCH_SHEETS + TETRX_SHEETS
    with st.sidebar:
        st.markdown("### Navigation")
        page = st.radio("Go to", nav_options, label_visibility="collapsed")
        st.markdown("---")
        st.caption("Live Google Sheets dashboard")
    return page

def main():
    if not GSPREAD_AVAILABLE:
        st.error("Missing gspread/google-auth in environment.")
        st.stop()

    spreadsheet_id = get_spreadsheet_id()
    if not spreadsheet_id:
        st.error("Missing GSHEET_SPREADSHEET_ID in Streamlit secrets.")
        st.stop()

    connected_ok = False
    try:
        data = load_dashboard_data(spreadsheet_id)
        connected_ok = True
    except Exception as e:
        render_header(False)
        st.error(f"Google Sheets connection failed: {e}")
        st.stop()

    render_header(connected_ok)
    page = sidebar_navigation()

    if page == "Overview":
        render_overview(data)
    elif page == "Student Profile":
        render_student_profile(data)
    elif page in data["details"]:
        render_sheet_page(page, data["details"][page], data["detail_contexts"][page])
    else:
        st.warning(f"{page} not found in live sheet.")

if __name__ == "__main__":
    main()
