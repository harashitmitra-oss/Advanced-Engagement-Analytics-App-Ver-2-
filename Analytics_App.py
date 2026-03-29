
import json
import re
from io import BytesIO
from collections import defaultdict

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


st.set_page_config(page_title="Tetr Business School Analytics Dashboard", layout="wide")

MASTER_SHEETS = ["Master UG", "Master PG"]
DETAIL_SHEETS = ["PG B5", "UG B9", "UG B8", "UG B7", "UG B6"]
TETRX_SHEETS = ["Tetr-X-UG", "Tetr-X-PG"]
ALL_REQUIRED = MASTER_SHEETS + DETAIL_SHEETS + TETRX_SHEETS

GREEN = "#0b3d2e"
GREEN_2 = "#1f7a56"
GREEN_3 = "#56a77b"
GREEN_4 = "#9cd4b5"
GREEN_5 = "#dff3e7"
DARK = "#12372a"
LIGHT_BG = "#f7fbf8"
RED = "#d9534f"

GSHEETS_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


# -----------------------------
# Styling
# -----------------------------
def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, #ffffff 0%, {LIGHT_BG} 100%);
        }}
        section[data-testid="stSidebar"] {{
            background: #f3faf5;
            border-right: 1px solid #d9eee1;
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
            padding: 12px 14px;
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
        .profile-card {{
            background: white;
            border: 1px solid #dfeee5;
            border-radius: 20px;
            padding: 18px 18px 12px 18px;
            box-shadow: 0 6px 18px rgba(11,61,46,0.05);
            margin-bottom: 14px;
        }}
        div[data-testid="stSidebar"] .stRadio > div {{
            gap: 0.45rem;
        }}
        div[data-testid="stSidebar"] .stRadio label {{
            background: #dff3e7;
            border: 1px solid #cde9d9;
            border-radius: 12px;
            padding: 10px 12px;
            width: 100%;
            margin: 0 0 4px 0;
            display: block;
        }}
        div[data-testid="stSidebar"] .stRadio label:hover {{
            background: #cfeedd;
            border-color: #bfe2cf;
        }}
        div[data-testid="stSidebar"] .stRadio label[data-checked="true"] {{
            background: linear-gradient(135deg, #0b3d2e 0%, #1f7a56 100%);
            color: white !important;
            border-color: #0b3d2e;
        }}
        h1, h2, h3, h4 {{ color: {DARK} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# -----------------------------
# Helpers
# -----------------------------
def clean_text(x):
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).replace("\n", " ").replace("\r", " ").replace("\xa0", " ").strip()


def slugify_name(x):
    s = clean_text(x).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s


def normalize_yes_no(x):
    s = clean_text(x).lower()
    return 1 if s in {"yes", "y", "1", "true", "present", "attended", "done"} else 0


def parse_date_safe(x):
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT


def numeric_percent_series(series):
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if len(s) and s.max() <= 1.05:
        s = s * 100
    return s


def best_matching_col(df: pd.DataFrame, candidates):
    lowered = {c: clean_text(c).lower() for c in df.columns}
    for cand in candidates:
        for col, low in lowered.items():
            if cand in low:
                return col
    return None


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


def parse_event_date(val):
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
        try:
            return pd.Timestamp(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        except Exception:
            return pd.NaT
    return pd.NaT


def infer_program_from_sheet(sheet_name: str) -> str:
    s = clean_text(sheet_name).lower()
    return "UG" if "ug" in s else ("PG" if "pg" in s else "")


def infer_batch_from_sheet_name(sheet_name: str):
    m = re.search(r"\b(b\d+)\b", clean_text(sheet_name).lower())
    return m.group(1).upper() if m else ""


def payment_label_from_text(text):
    s = clean_text(text).lower()
    if "refund" in s:
        return "Refunded"
    if "admitted" in s or s == "paid" or s.startswith("paid"):
        return "Paid / Admitted"
    if s in {"not paid", ""}:
        return "Not Paid"
    return "Not Paid"


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


def donut_chart(labels, values, title):
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.62,
        marker=dict(colors=[GREEN, GREEN_2, GREEN_3, GREEN_4, GREEN_5][:len(labels)]),
        textinfo="label+percent",
    ))
    fig.update_layout(title=title)
    return nice_layout(fig, height=340)


def live_status_html(is_connected: bool, mode_label: str):
    if is_connected:
        return f"""
        <div class="live-pill">
            <span class="heartbeat-wrap"><span class="heartbeat-ping"></span><span class="heartbeat-dot"></span></span>
            LIVE · {mode_label}
        </div>
        """
    return f"""
    <div class="live-pill offline">
        <span class="offline-dot"></span>
        OFFLINE · {mode_label}
    </div>
    """


# -----------------------------
# Data loading
# -----------------------------
def get_secrets_credentials():
    if "GOOGLE_SERVICE_ACCOUNT" not in st.secrets:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT in Streamlit secrets.")
    return dict(st.secrets["GOOGLE_SERVICE_ACCOUNT"])


def get_spreadsheet_id(default_value=""):
    if "GSHEET_SPREADSHEET_ID" in st.secrets:
        return st.secrets["GSHEET_SPREADSHEET_ID"]
    return default_value


def _get_gsheets_client():
    info = get_secrets_credentials()
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
    max_cols = max(len(r) for r in values)
    values = [r + [""] * (max_cols - len(r)) for r in values]
    df = pd.DataFrame(values)
    df.replace("", np.nan, inplace=True)
    return df


@st.cache_data(show_spinner=False)
def excel_get_sheet_names(file_bytes: bytes):
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def excel_read_raw_sheet(file_bytes: bytes, sheet_name: str):
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return pd.read_excel(xls, sheet_name=sheet_name, header=None)


def load_raw_sheet(source_mode: str, sheet_name: str, spreadsheet_id=None, file_bytes=None):
    if source_mode == "gsheets":
        return gsheets_read_raw_sheet(spreadsheet_id, sheet_name)
    return excel_read_raw_sheet(file_bytes, sheet_name)


def get_sheet_names(source_mode: str, spreadsheet_id=None, file_bytes=None):
    if source_mode == "gsheets":
        return gsheets_get_sheet_names(spreadsheet_id)
    return excel_get_sheet_names(file_bytes)


# -----------------------------
# Parsing
# -----------------------------
def load_master_sheet(raw: pd.DataFrame, program: str):
    # Real header at row 1, real students start at row 4
    header_idx = 0
    start_idx = 3
    df = raw.iloc[start_idx:].copy().reset_index(drop=True)
    df.columns = make_unique(raw.iloc[header_idx].tolist())
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["name"])
    email_col = best_matching_col(df, ["email"])
    contact_col = best_matching_col(df, ["contact"])
    batch_col = best_matching_col(df, ["batch"])
    country_col = best_matching_col(df, ["country"])
    income_col = best_matching_col(df, ["income"])
    admitted_group_col = best_matching_col(df, ["admitted group"])
    status_col = best_matching_col(df, ["status"])
    payment_col = best_matching_col(df, ["payment"])
    term0_col = best_matching_col(df, ["term zero group"])
    unofficial_group_col = best_matching_col(df, ["unofficial group"])

    if name_col is None:
        raise ValueError(f"Could not detect Name column in master sheet for {program}.")

    df = df[df[name_col].astype(str).str.strip().ne("")].copy()
    df["Program"] = program
    df["Batch"] = df[batch_col].astype(str).str.strip() if batch_col else ""
    if df["Batch"].str.fullmatch(r"\d+(\.0)?").any():
        df["Batch"] = df["Batch"].str.replace(".0", "", regex=False).map(lambda x: f"B{x}" if clean_text(x) else "")
    df["student_key"] = df[email_col].astype(str).str.strip().str.lower() if email_col else ""
    df["name_key"] = df[name_col].map(slugify_name)

    # derive event columns as explicit yes/no columns after payment-ish columns
    excluded = {c for c in [name_col, email_col, contact_col, batch_col, country_col, income_col, admitted_group_col, status_col, payment_col, term0_col, unofficial_group_col, "Program", "Batch", "student_key", "name_key"] if c}
    event_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        s = df[col].astype(str).str.strip().str.lower()
        if len(s) and s.isin({"yes","no","","nan"}).mean() > 0.75:
            event_cols.append(col)

    for col in event_cols:
        df[col] = df[col].apply(normalize_yes_no)

    df["participation_count"] = df[event_cols].sum(axis=1) if event_cols else 0
    total_events = max(len(event_cols), 1)
    df["engagement_pct"] = (df["participation_count"] / total_events) * 100
    df["engagement_score"] = df["participation_count"]

    pay_series = df[payment_col].astype(str) if payment_col else pd.Series("", index=df.index)
    df["master_payment_text"] = pay_series
    df["paid_label"] = pay_series.map(payment_label_from_text)
    df["is_paid"] = df["paid_label"].eq("Paid / Admitted")
    df["is_active"] = df["engagement_pct"] > 0

    ctx = {
        "name_col": name_col,
        "email_col": email_col,
        "contact_col": contact_col,
        "country_col": country_col,
        "batch_col": "Batch",
        "income_col": income_col,
        "status_col": status_col,
        "payment_col": payment_col,
        "admitted_group_col": admitted_group_col,
        "term0_col": term0_col,
        "unofficial_group_col": unofficial_group_col,
        "event_cols": event_cols,
    }
    return df, ctx


def build_event_label(category, name):
    cat = clean_text(category)
    nm = clean_text(name)
    if nm and cat:
        return f"{nm} [{cat}]"
    return nm or cat or "Event"


def load_event_sheet(raw: pd.DataFrame, sheet_name: str):
    # rows 1-5 metadata, row 6 header, row 7+ students
    category_idx = 0
    name_idx = 1
    date_idx = 2
    header_idx = 5
    start_idx = 6

    header_vals = list(raw.iloc[header_idx].tolist())
    categories = list(raw.iloc[category_idx].tolist()) if len(raw) > category_idx else []
    names = list(raw.iloc[name_idx].tolist()) if len(raw) > name_idx else []
    dates = list(raw.iloc[date_idx].tolist()) if len(raw) > date_idx else []

    final_headers = []
    event_dates = {}
    event_types = {}
    for i, hv in enumerate(header_vals):
        hdr = clean_text(hv)
        cat = clean_text(categories[i]) if i < len(categories) else ""
        nm = clean_text(names[i]) if i < len(names) else ""
        if hdr:
            final_headers.append(hdr)
        elif nm or cat:
            final_headers.append(build_event_label(cat, nm))
        else:
            final_headers.append(f"Unnamed_{i}")

    final_headers = make_unique(final_headers)
    df = raw.iloc[start_idx:].copy().reset_index(drop=True)
    df.columns = final_headers
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["student name", "name"])
    email_col = best_matching_col(df, ["email"])
    country_col = best_matching_col(df, ["country"])
    income_col = best_matching_col(df, ["income"])
    payment_status_col = best_matching_col(df, ["payment status", "status"])
    payment_date_col = best_matching_col(df, ["payment date"])
    engagement_pct_col = best_matching_col(df, ["overall engagement %"])
    if engagement_pct_col is None:
        engagement_pct_col = best_matching_col(df, ["engagement %"])
    engagement_score_col = best_matching_col(df, ["overall engagement score"])
    if engagement_score_col is None:
        engagement_score_col = best_matching_col(df, ["engagement score"])
    batch_col = best_matching_col(df, ["batch"])

    if name_col is None:
        raise ValueError(f"Could not detect Name column in {sheet_name}.")

    excluded = {c for c in [name_col, email_col, country_col, income_col, payment_status_col, payment_date_col, engagement_pct_col, engagement_score_col, batch_col] if c}
    event_cols = []
    for i, col in enumerate(final_headers):
        if col in excluded:
            continue
        s = df[col].astype(str).str.strip().str.lower()
        if len(s) and s.isin({"yes", "no", "y", "n", "true", "false", "1", "0", "attended", "present", "absent", "done", "", "nan"}).mean() > 0.55:
            event_cols.append(col)
            dt = parse_date_safe(dates[i] if i < len(dates) else None)
            if pd.notna(dt):
                event_dates[col] = dt
            event_types[col] = clean_text(categories[i]) if i < len(categories) else ""

    df = df[df[name_col].astype(str).str.strip().ne("")].copy()
    df["Program"] = infer_program_from_sheet(sheet_name)
    if batch_col:
        df["Batch"] = df[batch_col].astype(str).str.strip()
    else:
        df["Batch"] = infer_batch_from_sheet_name(sheet_name)
    df["Batch"] = df["Batch"].replace({"": infer_batch_from_sheet_name(sheet_name)})
    df["Batch"] = df["Batch"].astype(str).str.replace(".0", "", regex=False).map(lambda x: f"B{x}" if x.isdigit() else x)

    for col in event_cols:
        df[col] = df[col].apply(normalize_yes_no).astype(int)

    df["student_key"] = df[email_col].astype(str).str.strip().str.lower() if email_col else ""
    df["name_key"] = df[name_col].map(slugify_name)
    df["participation_count"] = df[event_cols].sum(axis=1) if event_cols else 0

    if engagement_score_col:
        df["engagement_score"] = pd.to_numeric(df[engagement_score_col], errors="coerce").fillna(df["participation_count"])
    else:
        df["engagement_score"] = df["participation_count"]

    if engagement_pct_col:
        df["engagement_pct"] = numeric_percent_series(df[engagement_pct_col])
    else:
        total_events = max(len(event_cols), 1)
        df["engagement_pct"] = (df["participation_count"] / total_events) * 100

    payment_series = df[payment_status_col].astype(str) if payment_status_col else pd.Series("", index=df.index)
    df["payment_status_text"] = payment_series
    df["paid_label"] = payment_series.map(payment_label_from_text)
    df["is_paid"] = df["paid_label"].eq("Paid / Admitted")
    df["is_active"] = df["engagement_pct"] > 0
    if payment_date_col:
        df["payment_date_parsed"] = df[payment_date_col].apply(parse_date_safe)
    else:
        df["payment_date_parsed"] = pd.NaT

    ctx = {
        "name_col": name_col,
        "email_col": email_col,
        "country_col": country_col,
        "income_col": income_col,
        "payment_status_col": payment_status_col,
        "payment_date_col": payment_date_col,
        "engagement_pct_col": engagement_pct_col,
        "engagement_score_col": engagement_score_col,
        "event_cols": event_cols,
        "event_dates": event_dates,
        "event_types": event_types,
        "batch_col": "Batch",
    }
    return df, ctx


def merge_truth(overview_df, detail_frames, tetrx_frames):
    # Tetr-X is source of truth for admitted/refunded + payment date
    truth_rows = []
    for sheet, df in tetrx_frames.items():
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            status = clean_text(r.get("payment_status_text", "")) or clean_text(r.get("master_payment_text", "")) or clean_text(r.get("Status", ""))
            paid_label = payment_label_from_text(status)
            payment_dt = r.get("payment_date_parsed", pd.NaT)
            truth_rows.append({
                "student_key": clean_text(r.get("student_key", "")),
                "name_key": clean_text(r.get("name_key", "")),
                "truth_paid_label": paid_label,
                "truth_is_paid": paid_label == "Paid / Admitted",
                "truth_payment_date": payment_dt,
                "truth_program": clean_text(r.get("Program", "")),
                "truth_batch": clean_text(r.get("Batch", "")),
                "truth_status_text": status,
                "source_sheet": sheet,
            })
    truth = pd.DataFrame(truth_rows)

    def build_lookup(df):
        email_map, name_map = {}, {}
        for _, row in df.iterrows():
            sk = clean_text(row["student_key"])
            nk = clean_text(row["name_key"])
            if sk and sk not in email_map:
                email_map[sk] = row
            if nk and nk not in name_map:
                name_map[nk] = row
        return email_map, name_map

    email_map, name_map = build_lookup(truth)

    def apply_truth(df):
        if df is None or df.empty:
            return df
        df = df.copy()
        final_labels, final_paid, final_dates = [], [], []
        for _, row in df.iterrows():
            candidate = None
            sk = clean_text(row.get("student_key", ""))
            nk = clean_text(row.get("name_key", ""))
            if sk and sk in email_map:
                candidate = email_map[sk]
            elif nk and nk in name_map:
                candidate = name_map[nk]
            if candidate is not None:
                final_labels.append(candidate["truth_paid_label"])
                final_paid.append(bool(candidate["truth_is_paid"]))
                dt = candidate["truth_payment_date"]
                final_dates.append(dt if pd.notna(dt) else row.get("payment_date_parsed", pd.NaT))
            else:
                lbl = clean_text(row.get("paid_label", "Not Paid")) or "Not Paid"
                final_labels.append(lbl)
                final_paid.append(lbl == "Paid / Admitted")
                final_dates.append(row.get("payment_date_parsed", pd.NaT))
        df["paid_label"] = final_labels
        df["is_paid"] = final_paid
        df["payment_date_final"] = final_dates
        return df

    overview_df = apply_truth(overview_df)
    detail_frames = {k: apply_truth(v) for k, v in detail_frames.items()}
    tetrx_frames = {k: apply_truth(v) for k, v in tetrx_frames.items()}
    return overview_df, detail_frames, tetrx_frames, truth


@st.cache_data(show_spinner=False, ttl=180)
def load_dashboard_data(source_mode: str, spreadsheet_id=None, file_bytes=None):
    sheet_names = get_sheet_names(source_mode, spreadsheet_id, file_bytes)
    missing = [s for s in ALL_REQUIRED if s not in sheet_names]

    masters, details, tetrx = {}, {}, {}
    master_contexts, detail_contexts, tetrx_contexts = {}, {}, {}

    for sheet in MASTER_SHEETS:
        if sheet in sheet_names:
            raw = load_raw_sheet(source_mode, sheet, spreadsheet_id, file_bytes)
            masters[sheet], master_contexts[sheet] = load_master_sheet(raw, "UG" if sheet.endswith("UG") else "PG")

    for sheet in DETAIL_SHEETS:
        if sheet in sheet_names:
            raw = load_raw_sheet(source_mode, sheet, spreadsheet_id, file_bytes)
            details[sheet], detail_contexts[sheet] = load_event_sheet(raw, sheet)

    for sheet in TETRX_SHEETS:
        if sheet in sheet_names:
            raw = load_raw_sheet(source_mode, sheet, spreadsheet_id, file_bytes)
            tetrx[sheet], tetrx_contexts[sheet] = load_event_sheet(raw, sheet)

    overview_df = pd.concat(list(masters.values()), ignore_index=True) if masters else pd.DataFrame()
    overview_df, details, tetrx, truth = merge_truth(overview_df, details, tetrx)

    return {
        "sheet_names": sheet_names,
        "missing": missing,
        "masters": masters,
        "details": details,
        "tetrx": tetrx,
        "master_contexts": master_contexts,
        "detail_contexts": detail_contexts,
        "tetrx_contexts": tetrx_contexts,
        "overview_df": overview_df,
        "truth_df": truth,
    }


# -----------------------------
# Analytics helpers
# -----------------------------
def event_summary_for_student(row, ctx):
    event_cols = ctx.get("event_cols", [])
    event_dates = ctx.get("event_dates", {})
    event_types = ctx.get("event_types", {})
    events = []
    for col in event_cols:
        val = pd.to_numeric(pd.Series([row.get(col, 0)]), errors="coerce").fillna(0).iloc[0]
        if val > 0:
            events.append({
                "Event": col,
                "Event Type": event_types.get(col, "Other") or "Other",
                "Date": event_dates.get(col, pd.NaT),
                "Attended": int(val),
            })
    return pd.DataFrame(events)


def event_type_counts(events_df):
    if events_df.empty:
        return pd.DataFrame(columns=["Event Type","Count","Share %"])
    out = events_df.groupby("Event Type").size().reset_index(name="Count")
    total = out["Count"].sum()
    out["Share %"] = np.where(total > 0, out["Count"] / total * 100, 0)
    return out.sort_values("Count", ascending=False)


def find_student_matches(query_names, data):
    overview = data["overview_df"].copy()
    matches = []

    # Build combined index across overview + details + tetrx
    for q in query_names:
        q_clean = clean_text(q)
        if not q_clean:
            continue
        q_key = slugify_name(q_clean)
        exact = overview[overview["name_key"].eq(q_key)]
        if exact.empty:
            exact = overview[overview[overview.columns[0]].astype(str).str.lower().str.contains(q_clean.lower(), na=False)]
        if exact.empty:
            continue
        for _, base in exact.iterrows():
            matches.append(build_student_profile(base, data))
    # dedupe by email or name
    unique = []
    seen = set()
    for m in matches:
        key = (clean_text(m["master"].get("student_key", "")), clean_text(m["master"].get("name_key", "")))
        if key not in seen:
            seen.add(key)
            unique.append(m)
    return unique


def build_student_profile(master_row, data):
    overview = data["overview_df"]
    detail_frames = data["details"]
    tetrx_frames = data["tetrx"]

    sk = clean_text(master_row.get("student_key", ""))
    nk = clean_text(master_row.get("name_key", ""))

    linked_detail_rows = []
    for sheet, df in detail_frames.items():
        hit = pd.DataFrame()
        if sk:
            hit = df[df["student_key"].eq(sk)]
        if hit.empty and nk:
            hit = df[df["name_key"].eq(nk)]
        if not hit.empty:
            row = hit.iloc[0].copy()
            row["source_sheet"] = sheet
            linked_detail_rows.append(row)

    linked_tetrx_rows = []
    for sheet, df in tetrx_frames.items():
        hit = pd.DataFrame()
        if sk:
            hit = df[df["student_key"].eq(sk)]
        if hit.empty and nk:
            hit = df[df["name_key"].eq(nk)]
        if not hit.empty:
            row = hit.iloc[0].copy()
            row["source_sheet"] = sheet
            linked_tetrx_rows.append(row)

    # choose best activity row
    combined_rows = linked_detail_rows + linked_tetrx_rows
    activity_row = None
    activity_ctx = None
    if combined_rows:
        combined_rows = sorted(combined_rows, key=lambda r: float(r.get("engagement_score", 0)), reverse=True)
        activity_row = combined_rows[0]
        activity_ctx = data["detail_contexts"].get(activity_row["source_sheet"], data["tetrx_contexts"].get(activity_row["source_sheet"], {}))

    events_df = event_summary_for_student(activity_row, activity_ctx) if activity_row is not None else pd.DataFrame(columns=["Event","Event Type","Date","Attended"])
    event_type_df = event_type_counts(events_df)

    timeline_frames = []
    for row in linked_detail_rows + linked_tetrx_rows:
        src = row["source_sheet"]
        ctx = data["detail_contexts"].get(src, data["tetrx_contexts"].get(src, {}))
        local = event_summary_for_student(row, ctx)
        if not local.empty:
            local["Sheet"] = src
            timeline_frames.append(local)
    timeline = pd.concat(timeline_frames, ignore_index=True) if timeline_frames else pd.DataFrame(columns=["Event","Event Type","Date","Attended","Sheet"])
    if not timeline.empty:
        timeline = timeline.sort_values("Date")

    return {
        "master": master_row,
        "detail_rows": linked_detail_rows,
        "tetrx_rows": linked_tetrx_rows,
        "events_df": events_df,
        "event_type_df": event_type_df,
        "timeline_df": timeline,
    }


# -----------------------------
# Rendering
# -----------------------------
def render_header(connected_ok, source_label):
    left, right = st.columns([4, 1.2])
    with left:
        st.markdown(
            """
            <div class="hero-card">
              <div style="font-size:30px;font-weight:900;color:#0b3d2e;">Tetr Business School Analytics Dashboard</div>
              <div style="margin-top:6px;color:#2e6b57;font-weight:600;">
                Live overview, batch analytics, student profiles, Tetr-X tracking, and payment-linked participation insights.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(live_status_html(connected_ok, source_label), unsafe_allow_html=True)


def render_overview(overview_df, ctx, truth_df):
    st.subheader("Overview")
    if overview_df.empty:
        st.warning("Master UG / Master PG data could not be loaded.")
        return

    name_col = ctx["name_col"]
    country_col = ctx["country_col"]
    batch_col = ctx["batch_col"]
    income_col = ctx["income_col"]

    total_students = int(overview_df[name_col].count())
    total_active = int(overview_df["is_active"].sum())
    total_paid = int(overview_df["is_paid"].sum())
    total_refunded = int(overview_df["paid_label"].eq("Refunded").sum())
    ug_students = int((overview_df["Program"] == "UG").sum())
    pg_students = int((overview_df["Program"] == "PG").sum())
    ug_paid = int(((overview_df["Program"] == "UG") & (overview_df["is_paid"])).sum())
    pg_paid = int(((overview_df["Program"] == "PG") & (overview_df["is_paid"])).sum())
    active_rate = round((total_active / total_students * 100), 1) if total_students else 0
    paid_rate = round((total_paid / total_students * 100), 1) if total_students else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Students", f"{total_students:,}")
    m2.metric("Active Students", f"{total_active:,}", delta=f"{active_rate}% active")
    m3.metric("Paid / Admitted", f"{total_paid:,}", delta=f"{paid_rate}% paid")
    m4.metric("Refunded", f"{total_refunded:,}")
    m5.metric("UG vs PG", f"{ug_students:,} / {pg_students:,}", delta="UG / PG")

    g1, g2, g3 = st.columns([1.2, 1, 1])
    with g1:
        st.plotly_chart(gauge_chart(total_students, "Total Students", maximum=max(total_students, 1)),
                        use_container_width=True, key="overview_gauge")
    with g2:
        st.plotly_chart(donut_chart(["UG", "PG"], [ug_students, pg_students], "UG / PG Distribution"),
                        use_container_width=True, key="overview_prog_donut")
    with g3:
        st.plotly_chart(donut_chart(["UG Paid", "PG Paid"], [ug_paid, pg_paid], "Paid Distribution"),
                        use_container_width=True, key="overview_paid_donut")

    a1, a2 = st.columns(2)
    with a1:
        batch_plot = (
            overview_df.groupby(batch_col, dropna=False)[name_col]
            .count().reset_index(name="Students")
            .sort_values("Students", ascending=False)
        )
        batch_plot[batch_col] = batch_plot[batch_col].replace("", "Unknown")
        fig = px.bar(batch_plot, x=batch_col, y="Students", title="Students by Batch")
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-25),
                        use_container_width=True, key="overview_batch_bar")

    with a2:
        status_plot = overview_df.groupby(["Program", "paid_label"])[name_col].count().reset_index(name="Students")
        fig = px.bar(
            status_plot, x="Program", y="Students", color="paid_label", barmode="group",
            title="Paid vs Not Paid by Program",
            color_discrete_map={"Paid / Admitted": GREEN, "Refunded": RED, "Not Paid": GREEN_4},
        )
        st.plotly_chart(nice_layout(fig, height=380), use_container_width=True, key="overview_status_bar")

    b1, b2 = st.columns(2)
    with b1:
        country_plot = overview_df.groupby(country_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False).head(12)
        fig = px.bar(country_plot, x=country_col, y="Students", title="Top Countries")
        fig.update_traces(marker_color=GREEN_3)
        st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-30), use_container_width=True, key="overview_country_bar")
    with b2:
        income_plot = overview_df.groupby(income_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False)
        fig = px.bar(income_plot, x=income_col, y="Students", title="Income Distribution")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-25), use_container_width=True, key="overview_income_bar")

    c1, c2 = st.columns(2)
    with c1:
        paid_by_batch = overview_df.groupby(batch_col)["is_paid"].sum().reset_index(name="Paid / Admitted")
        total_by_batch = overview_df.groupby(batch_col)[name_col].count().reset_index(name="Total")
        merged = paid_by_batch.merge(total_by_batch, on=batch_col, how="left")
        merged["Paid %"] = np.where(merged["Total"] > 0, merged["Paid / Admitted"] / merged["Total"] * 100, 0)
        fig = px.bar(merged.sort_values("Paid %", ascending=False), x=batch_col, y="Paid %", title="Paid % by Batch")
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-25), use_container_width=True, key="overview_paidpct_bar")
    with c2:
        active_by_batch = overview_df.groupby(batch_col)["is_active"].sum().reset_index(name="Active")
        total_by_batch = overview_df.groupby(batch_col)[name_col].count().reset_index(name="Total")
        merged = active_by_batch.merge(total_by_batch, on=batch_col, how="left")
        merged["Active %"] = np.where(merged["Total"] > 0, merged["Active"] / merged["Total"] * 100, 0)
        fig = px.bar(merged.sort_values("Active %", ascending=False), x=batch_col, y="Active %", title="Active % by Batch")
        fig.update_traces(marker_color=GREEN_3)
        st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-25), use_container_width=True, key="overview_activepct_bar")

    st.markdown("#### Master Student Table")
    preview_cols = [c for c in [name_col, "Program", batch_col, country_col, income_col, "engagement_pct", "engagement_score", "paid_label"] if c in overview_df.columns]
    st.dataframe(overview_df[preview_cols].sort_values(["engagement_pct","engagement_score"], ascending=False),
                 use_container_width=True, height=420, key="overview_table")


def render_sheet_tab(sheet_name, df, ctx, tab_prefix):
    st.subheader(sheet_name)
    if df.empty:
        st.warning(f"No data available for {sheet_name}.")
        return

    name_col = ctx["name_col"]
    country_col = ctx["country_col"]
    event_cols = ctx["event_cols"]
    event_dates = ctx["event_dates"]
    event_types = ctx.get("event_types", {})

    total_students = int(df[name_col].count())
    active_students = int(df["is_active"].sum())
    paid_students = int(df["is_paid"].sum())
    avg_engagement = round(float(df["engagement_pct"].mean()), 1) if len(df) else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students", f"{total_students:,}")
    k2.metric("Active", f"{active_students:,}", delta=f"{(active_students/total_students*100 if total_students else 0):.1f}%")
    k3.metric("Paid / Admitted", f"{paid_students:,}", delta=f"{(paid_students/total_students*100 if total_students else 0):.1f}%")
    k4.metric("Avg Engagement", f"{avg_engagement:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="engagement_pct", nbins=12, title="Engagement Distribution")
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{tab_prefix}_hist")
    with c2:
        status = df["paid_label"].value_counts().reset_index()
        status.columns = ["Status", "Students"]
        fig = px.pie(status, names="Status", values="Students", hole=0.58, color="Status",
                     color_discrete_map={"Paid / Admitted": GREEN, "Refunded": RED, "Not Paid": GREEN_4})
        fig.update_layout(title="Payment Status")
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{tab_prefix}_pie")

    d1, d2 = st.columns(2)
    with d1:
        if event_cols:
            event_counts = pd.DataFrame({
                "Event": event_cols,
                "Participants": [int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in event_cols],
                "Type": [event_types.get(c, "Other") or "Other" for c in event_cols],
            }).sort_values("Participants", ascending=False).head(15)
            fig = px.bar(event_counts, x="Participants", y="Event", color="Type", orientation="h", title="Top Events by Participation")
            st.plotly_chart(nice_layout(fig, height=500), use_container_width=True, key=f"{tab_prefix}_events")
    with d2:
        top_country = df.groupby(country_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False).head(10)
        fig = px.bar(top_country, x=country_col, y="Students", title="Country Split")
        fig.update_traces(marker_color=GREEN_3)
        st.plotly_chart(nice_layout(fig, height=430, x_tickangle=-30), use_container_width=True, key=f"{tab_prefix}_country")

    e1, e2 = st.columns(2)
    with e1:
        type_df = pd.DataFrame({
            "Event Type": [event_types.get(c, "Other") or "Other" for c in event_cols],
            "Participants": [int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in event_cols],
        }).groupby("Event Type", as_index=False)["Participants"].sum().sort_values("Participants", ascending=False)
        if not type_df.empty:
            fig = px.bar(type_df, x="Event Type", y="Participants", title="Participation by Event Type")
            fig.update_traces(marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-20), use_container_width=True, key=f"{tab_prefix}_types")
    with e2:
        payment_timeline = df.dropna(subset=["payment_date_final"]).copy()
        if not payment_timeline.empty:
            payment_timeline["pay_day"] = pd.to_datetime(payment_timeline["payment_date_final"]).dt.date
            day_plot = payment_timeline.groupby("pay_day")[name_col].count().reset_index(name="Paid Students")
            fig = px.line(day_plot, x="pay_day", y="Paid Students", markers=True, title="Payment Timeline")
            fig.update_traces(line_color=GREEN, marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{tab_prefix}_payline")

    if event_cols:
        timeline = pd.DataFrame({
            "Event": event_cols,
            "Date": [event_dates.get(c, pd.NaT) for c in event_cols],
            "Participants": [int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in event_cols],
        }).dropna(subset=["Date"]).sort_values("Date")
        if not timeline.empty:
            fig = px.line(timeline, x="Date", y="Participants", markers=True, title="Participation Timeline")
            fig.update_traces(line_color=GREEN_2, marker_color=GREEN_2)
            st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{tab_prefix}_timeline")

    f1, f2 = st.columns(2)
    with f1:
        students = df[[name_col, "engagement_pct", "engagement_score", "paid_label"]].sort_values(["engagement_pct","engagement_score"], ascending=False).head(20)
        st.markdown("#### Top Students")
        st.dataframe(students, use_container_width=True, height=390, key=f"{tab_prefix}_top_students")
    with f2:
        targets = df[(~df["is_paid"]) & (df["engagement_pct"] > 0)][[name_col, "engagement_pct", "engagement_score", "paid_label"]].sort_values(["engagement_pct","engagement_score"], ascending=False).head(20)
        st.markdown("#### Best Upgrade Targets")
        st.dataframe(targets, use_container_width=True, height=390, key=f"{tab_prefix}_targets")

    st.markdown("#### Full Student Table")
    display_cols = [c for c in [name_col, country_col, "Batch", "engagement_pct", "engagement_score", "paid_label", "payment_date_final"] if c in df.columns]
    display_cols += [c for c in event_cols[:8] if c in df.columns]
    st.dataframe(df[display_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False),
                 use_container_width=True, height=440, key=f"{tab_prefix}_full_table")


def render_student_profile(data):
    st.subheader("Student Profile")
    overview = data["overview_df"]
    if overview.empty:
        st.warning("No master student data available.")
        return

    name_col = data["master_contexts"]["Master UG"]["name_col"] if "Master UG" in data["master_contexts"] else list(overview.columns)[0]
    all_names = sorted(overview[name_col].dropna().astype(str).unique().tolist())

    st.markdown("Search one or more students by exact name, partial name, comma-separated names, or pasted lines.")
    c1, c2 = st.columns([2, 1.3])
    with c1:
        free_text = st.text_area("Search names", placeholder="Type or paste names, separated by comma or new lines", height=100, key="student_profile_text")
    with c2:
        selected = st.multiselect("Quick select", options=all_names, key="student_profile_multi")

    query_names = []
    if free_text:
        for part in re.split(r"[,\\n]+", free_text):
            if clean_text(part):
                query_names.append(clean_text(part))
    query_names.extend(selected)
    query_names = list(dict.fromkeys(query_names))

    if not query_names:
        st.info("Enter or select a student name to view profiles.")
        return

    profiles = find_student_matches(query_names, data)
    if not profiles:
        st.warning("No matching students found.")
        return

    for i, profile in enumerate(profiles, start=1):
        master = profile["master"]
        display_name = clean_text(master.get(name_col, "Student"))
        st.markdown(f'<div class="profile-card">', unsafe_allow_html=True)
        st.markdown(f"### {display_name}")

        top1, top2, top3, top4 = st.columns(4)
        top1.metric("Program", clean_text(master.get("Program", "")))
        top2.metric("Batch", clean_text(master.get("Batch", "")))
        top3.metric("Paid Status", clean_text(master.get("paid_label", "")))
        top4.metric("Total Events", f"{len(profile['events_df']):,}")

        info_cols = st.columns(3)
        master_items = {
            "Email": master.get("Email", master.get("email", master.get("student_key", ""))),
            "Country": master.get("Country", ""),
            "Income": master.get("Income", ""),
            "Community / Group": master.get("Admitted Group (Batch onwards)", master.get("Admitted Group (Batch Onwards) (working)", "")),
            "Status": master.get("Status", ""),
            "Payment": master.get("master_payment_text", master.get("paid_label", "")),
            "Payment Date": master.get("payment_date_final", pd.NaT),
        }
        items = list(master_items.items())
        for idx, (k, v) in enumerate(items):
            with info_cols[idx % 3]:
                st.markdown(f"**{k}**")
                if pd.isna(v):
                    st.write("—")
                else:
                    st.write(v)

        lower1, lower2 = st.columns([1.2, 1])
        with lower1:
            st.markdown("#### Participation Timeline")
            timeline = profile["timeline_df"]
            if not timeline.empty:
                fig = px.scatter(
                    timeline,
                    x="Date",
                    y="Event Type",
                    color="Sheet",
                    hover_data=["Event"],
                    title="Events Attended Over Time",
                )
                fig.update_traces(marker=dict(size=10))
                pay_dt = master.get("payment_date_final", pd.NaT)
                if pd.notna(pay_dt):
                    fig.add_vline(x=pd.to_datetime(pay_dt), line_dash="dash", line_color=RED, annotation_text="Payment Date")
                st.plotly_chart(nice_layout(fig, height=420), use_container_width=True, key=f"profile_timeline_{i}")
            else:
                st.info("No participation events found for this student in the linked batch/Tetr-X sheets.")

        with lower2:
            st.markdown("#### Event Type Mix")
            type_df = profile["event_type_df"]
            if not type_df.empty:
                fig = px.bar(type_df, x="Event Type", y="Count", text=type_df["Share %"].round(1).astype(str) + "%", title="Event Type Counts & Shares")
                fig.update_traces(marker_color=GREEN_2)
                st.plotly_chart(nice_layout(fig, height=420, x_tickangle=-20), use_container_width=True, key=f"profile_types_{i}")
            else:
                st.info("No event type data available.")

        t1, t2 = st.columns(2)
        with t1:
            st.markdown("#### Events Attended")
            events_df = profile["events_df"].copy()
            if not events_df.empty:
                events_df["Date"] = pd.to_datetime(events_df["Date"]).dt.date
                st.dataframe(events_df, use_container_width=True, height=260, key=f"profile_events_{i}")
            else:
                st.info("No event attendance found.")
        with t2:
            st.markdown("#### Linked Sheet Records")
            records = []
            for row in profile["detail_rows"] + profile["tetrx_rows"]:
                records.append({
                    "Sheet": row["source_sheet"],
                    "Engagement %": row.get("engagement_pct", 0),
                    "Engagement Score": row.get("engagement_score", 0),
                    "Paid Status": row.get("paid_label", ""),
                    "Payment Date": row.get("payment_date_final", pd.NaT),
                })
            if records:
                st.dataframe(pd.DataFrame(records), use_container_width=True, height=260, key=f"profile_records_{i}")
            else:
                st.info("No linked batch/Tetr-X records found.")

        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------
# App / Sidebar
# -----------------------------
def resolve_source():
    spreadsheet_id = ""
    file_bytes = None
    connected_ok = False
    note = ""
    if GSPREAD_AVAILABLE:
        with st.sidebar:
            st.markdown("## Data Source")
            source_choice = st.radio("Source", ["Google Sheets (live)", "Upload Excel (manual)"], index=0)
    else:
        source_choice = "Upload Excel (manual)"
        st.sidebar.warning("Install gspread and google-auth for live Google Sheets.")

    if source_choice == "Google Sheets (live)":
        spreadsheet_id = get_spreadsheet_id("1By2Zb8vKQnTIQn72JRgyEuuRgO6ZZARCZ1JNklmf25U")
        with st.sidebar:
            spreadsheet_id = st.text_input("Spreadsheet ID", value=spreadsheet_id)
        try:
            names = gsheets_get_sheet_names(spreadsheet_id)
            connected_ok = True
            note = f"Connected to {len(names)} sheets"
        except Exception as e:
            connected_ok = False
            note = f"Connection failed: {e}"
        return {
            "source_mode": "gsheets",
            "spreadsheet_id": spreadsheet_id,
            "file_bytes": None,
            "connected_ok": connected_ok,
            "note": note,
        }

    with st.sidebar:
        uploaded = st.file_uploader("Workbook", type=["xlsx"])
    if uploaded is not None:
        file_bytes = uploaded.getvalue()
        try:
            names = excel_get_sheet_names(file_bytes)
            connected_ok = True
            note = f"Loaded {len(names)} sheets"
        except Exception as e:
            connected_ok = False
            note = f"Workbook load failed: {e}"
    return {
        "source_mode": "excel",
        "spreadsheet_id": "",
        "file_bytes": file_bytes,
        "connected_ok": connected_ok,
        "note": note,
    }


def main():
    cfg = resolve_source()
    render_header(cfg["connected_ok"], "Google Sheets" if cfg["source_mode"] == "gsheets" else "Excel")
    if cfg["note"]:
        if cfg["connected_ok"]:
            st.caption(cfg["note"])
        else:
            st.error(cfg["note"])

    if cfg["source_mode"] == "excel" and cfg["file_bytes"] is None:
        st.info("Connect the Google Sheet or upload the workbook to load the dashboard.")
        return

    try:
        data = load_dashboard_data(cfg["source_mode"], spreadsheet_id=cfg["spreadsheet_id"], file_bytes=cfg["file_bytes"])
    except Exception as e:
        st.error(f"Failed to load dashboard data: {e}")
        return

    if data["missing"]:
        st.warning("Missing expected sheets: " + ", ".join(data["missing"]))

    with st.sidebar:
        st.markdown("## Navigate")
        pages = ["Overview", "Student Profile"] + DETAIL_SHEETS + TETRX_SHEETS
        page = st.radio("Page", pages, index=0, label_visibility="collapsed")

    if page == "Overview":
        master_ctx = next(iter(data["master_contexts"].values()))
        render_overview(data["overview_df"], master_ctx, data["truth_df"])
    elif page == "Student Profile":
        render_student_profile(data)
    elif page in data["details"]:
        render_sheet_tab(page, data["details"][page], data["detail_contexts"][page], f"detail_{slugify_name(page)}")
    elif page in data["tetrx"]:
        render_sheet_tab(page, data["tetrx"][page], data["tetrx_contexts"][page], f"tetrx_{slugify_name(page)}")
    else:
        st.warning("Page data not available.")


if __name__ == "__main__":
    main()
