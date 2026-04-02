import json
import re
from datetime import datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except Exception:
    GSPREAD_AVAILABLE = False

st.set_page_config(page_title="Tetr Analytics Dashboard", layout="wide")

MASTER_SHEETS = ["Master UG", "Master PG"]
UG_BATCH_SHEETS = ["UG - B1 to B4", "UG B5", "UG B6", "UG B7", "UG B8", "UG B9"]
PG_BATCH_SHEETS = ["PG - B1 & B2", "PG - B3 & B4", "PG B5"]
TX_SHEETS = ["Tetr-X-UG", "Tetr-X-PG"]
ALL_REQUIRED = MASTER_SHEETS + UG_BATCH_SHEETS + PG_BATCH_SHEETS + TX_SHEETS

GREEN = "#0b3d2e"
GREEN_2 = "#1f7a56"
GREEN_3 = "#56a77b"
GREEN_4 = "#9cd4b5"
GREEN_5 = "#dff3e7"
DARK = "#12372a"
LIGHT_BG = "#f7fbf8"
RED = "#d9534f"
AMBER = "#ffb000"

GSHEETS_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]


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
            padding: 18px 22px;
            box-shadow: 0 8px 24px rgba(11, 61, 46, 0.06);
            margin-bottom: 12px;
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
        .heartbeat-wrap {{ position: relative; width: 12px; height: 12px; }}
        .heartbeat-dot {{
            width: 12px; height: 12px; border-radius: 50%; background: #1bb55c;
            position: absolute; top: 0; left: 0; z-index: 2;
        }}
        .heartbeat-ping {{
            width: 12px; height: 12px; border-radius: 50%; background: rgba(27,181,92,0.30);
            position: absolute; top: 0; left: 0; animation: heartbeatPing 1.5s ease-out infinite; z-index: 1;
        }}
        .offline-dot {{ width: 12px; height: 12px; border-radius: 50%; background: {RED}; }}
        @keyframes heartbeatPing {{ 0% {{ transform: scale(0.9); opacity: 0.9; }} 70% {{ transform: scale(2.2); opacity: 0; }} 100% {{ transform: scale(2.2); opacity: 0; }} }}
        div[data-testid="stMetric"] {{
            background: #ffffff;
            border: 1px solid #dbeee0;
            border-radius: 16px;
            padding: 10px 12px;
            box-shadow: 0 2px 10px rgba(11, 61, 46, 0.05);
        }}
        div[data-testid="stMetric"] label {{ color: {GREEN_2} !important; font-weight: 700 !important; }}
        h1, h2, h3 {{ color: {DARK} !important; }}
        .stRadio [role="radiogroup"] label {{
            background: #eaf7ee !important;
            border: 1px solid #cfe8d9 !important;
            border-radius: 12px !important;
            padding: 10px 12px !important;
            margin-bottom: 8px !important;
            width: 100% !important;
        }}
        .stRadio [role="radiogroup"] label:hover {{ background: #def2e6 !important; }}
        .stRadio [role="radiogroup"] label p {{ color: #0b3d2e !important; font-weight: 700 !important; width:100% !important; }}
        .stRadio [role="radiogroup"] > label, .stRadio [role="radiogroup"] div[role="radiogroup"] > label {{ width:100% !important; display:flex !important; }}
        .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
        .stTabs [data-baseweb="tab"] {{
            background: #edf8f1;
            border: 1px solid #d6eadc;
            border-radius: 12px;
            padding: 10px 14px;
        }}
        .stTabs [aria-selected="true"] {{ background: #dff3e7; border-color: #8fcaab; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


def clean_text(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").replace("\xa0", " ").strip()


def normalize_name(x):
    s = clean_text(x).lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_email(x):
    return clean_text(x).lower()


def is_numeric_or_percent_text(x):
    s = clean_text(x).replace(",", "")
    if not s:
        return False
    return bool(re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", s))


def is_valid_student_name(x):
    s = clean_text(x)
    if not s:
        return False
    lower = s.lower().strip()
    if is_numeric_or_percent_text(s):
        return False
    if lower in {"total", "totals", "average", "avg", "mean", "median", "sum", "count", "percentage", "%"}:
        return False
    return bool(re.search(r"[A-Za-z]", s))


def normalize_yes_no(x):
    s = clean_text(x).lower()
    return 1 if s in {"yes", "y", "1", "true", "present", "attended", "done"} else 0


def normalize_community_status(x):
    s = clean_text(x).strip().lower()
    if s in {"tetr x", "tetrx", "added to term 0"}:
        return "Tetr X"
    if s == "in":
        return "In"
    return "Out"


def parse_date_safe(x):
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT


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


def best_matching_col(df: pd.DataFrame, candidates):
    lowered = {c: clean_text(c).lower() for c in df.columns}
    for cand in candidates:
        for col, low in lowered.items():
            if cand in low:
                return col
    return None


def infer_program_from_sheet(sheet_name):
    s = sheet_name.lower()
    if "ug" in s:
        return "UG"
    if "pg" in s:
        return "PG"
    return ""


def infer_batch_group_from_sheet_name(sheet_name: str) -> str:
    s = clean_text(sheet_name)
    return s


def live_status_html(is_connected: bool, mode_label: str):
    if is_connected:
        return f'''<div class="live-pill"><span class="heartbeat-wrap"><span class="heartbeat-ping"></span><span class="heartbeat-dot"></span></span>LIVE · {mode_label}</div>'''
    return f'''<div class="live-pill offline"><span class="offline-dot"></span>OFFLINE · {mode_label}</div>'''


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


def find_logo_path():
    base = Path(__file__).resolve().parent
    for pat in ["logo.png", "logo.jpg", "logo.jpeg", "logo.webp", "logo.svg"]:
        p = base / pat
        if p.exists():
            return p
    return None


# ---------------- Data source ----------------

def get_secret_service_account():
    if "GOOGLE_SERVICE_ACCOUNT" not in st.secrets:
        raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT in Streamlit secrets")
    return dict(st.secrets["GOOGLE_SERVICE_ACCOUNT"])


def _get_gsheets_client():
    key_dict = get_secret_service_account()
    creds = Credentials.from_service_account_info(key_dict, scopes=GSHEETS_SCOPES)
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
    df = pd.DataFrame(values)
    df.replace("", np.nan, inplace=True)
    return df.dropna(how="all")


@st.cache_data(show_spinner=False)
def excel_get_sheet_names(file_bytes: bytes):
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def excel_read_raw_sheet(file_bytes: bytes, sheet_name: str):
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return pd.read_excel(xls, sheet_name=sheet_name, header=None).dropna(how="all")


def resolve_source():
    spreadsheet_id = st.secrets.get("GSHEET_SPREADSHEET_ID", "") if hasattr(st, "secrets") else ""
    file_bytes = None
    source_mode = "excel"
    connected_ok = False
    connection_note = ""

    with st.sidebar:
        st.markdown("## 📡 Data Source")
        options = ["Upload Excel (manual)"]
        if GSPREAD_AVAILABLE and spreadsheet_id:
            options.insert(0, "Google Sheets (live)")
        source_choice = st.radio("Source", options, index=0, key="source_choice")

        uploaded = st.file_uploader("Manual workbook (.xlsx)", type=["xlsx"], key="manual_upload")
        if uploaded is not None:
            file_bytes = uploaded.getvalue()

    if source_choice == "Google Sheets (live)":
        source_mode = "gsheets"
        try:
            _ = gsheets_get_sheet_names(spreadsheet_id)
            connected_ok = True
            connection_note = "Google Sheets"
        except Exception as e:
            connected_ok = False
            connection_note = f"Google Sheets connection failed: {e}"
            if file_bytes is not None:
                source_mode = "excel"
    else:
        source_mode = "excel"
        connected_ok = file_bytes is not None
        connection_note = "Manual Workbook" if file_bytes is not None else "No workbook uploaded"

    return {
        "source_mode": source_mode,
        "spreadsheet_id": spreadsheet_id,
        "file_bytes": file_bytes,
        "connected_ok": connected_ok,
        "connection_note": connection_note,
    }


def get_sheet_names(source_mode: str, spreadsheet_id=None, file_bytes=None):
    if source_mode == "gsheets":
        return gsheets_get_sheet_names(spreadsheet_id)
    if file_bytes is None:
        return []
    return excel_get_sheet_names(file_bytes)


def load_raw_sheet(source_mode: str, sheet_name: str, spreadsheet_id=None, file_bytes=None):
    if source_mode == "gsheets":
        return gsheets_read_raw_sheet(spreadsheet_id, sheet_name)
    return excel_read_raw_sheet(file_bytes, sheet_name)


# ---------------- Parsing ----------------


def parse_master_sheet(raw: pd.DataFrame, program: str, sheet_name: str):
    header_row = 0
    data_start = 3
    header = make_unique(raw.iloc[header_row].tolist())
    df = raw.iloc[data_start:].copy().reset_index(drop=True)
    df.columns = header
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["name"])
    email_col = best_matching_col(df, ["email"])
    batch_col = best_matching_col(df, ["batch"])
    country_col = best_matching_col(df, ["country"])
    income_col = best_matching_col(df, ["income"])
    status_col = best_matching_col(df, ["status"])
    payment_col = best_matching_col(df, ["payment"])
    community_status_col = best_matching_col(df, ["community status", "admitted group"])
    term_zero_col = best_matching_col(df, ["term zero group"])

    if not name_col:
        raise ValueError(f"Name column not found in {sheet_name}")

    df = df[df[name_col].apply(is_valid_student_name)].copy()
    df["Program"] = program
    df["Batch"] = df[batch_col].astype(str).str.strip() if batch_col else ""
    df["source_sheet"] = sheet_name
    df["student_name"] = df[name_col].map(clean_text)
    df["student_key"] = df["student_name"].map(normalize_name)
    df["email_key"] = df[email_col].map(normalize_email) if email_col else ""
    community_base = df[community_status_col].map(clean_text) if community_status_col else pd.Series("", index=df.index)
    if term_zero_col:
        term_zero_series = df[term_zero_col].map(clean_text)
        community_base = community_base.where(term_zero_series.eq(""), term_zero_series)
    df["community_status_value"] = community_base.map(normalize_community_status)

    pay_series = df[payment_col].astype(str).str.lower().str.strip() if payment_col else pd.Series("", index=df.index)
    stat_series = df[status_col].astype(str).str.lower().str.strip() if status_col else pd.Series("", index=df.index)
    df["master_is_paid"] = stat_series.eq("admitted")
    df["master_is_refunded"] = pay_series.str.contains("refund", na=False) | stat_series.str.contains("refund", na=False)
    df["master_status_value"] = df[status_col].map(clean_text) if status_col else ""
    df["master_payment_value"] = df[payment_col].map(clean_text) if payment_col else ""

    event_cols = []
    protected = {name_col, email_col, batch_col, country_col, income_col, status_col, payment_col, community_status_col,
                 "Program", "Batch", "source_sheet", "student_name", "student_key", "email_key", "community_status_value",
                 "master_is_paid", "master_is_refunded", "master_status_value", "master_payment_value"}
    for col in df.columns:
        if col in protected:
            continue
        s = df[col].fillna("").astype(str).str.strip().str.lower()
        if len(s) and (s.isin({"yes", "no", "", "nan"}).mean() > 0.6):
            event_cols.append(col)
            df[col] = s.map(normalize_yes_no)

    df["participation_count_master"] = df[event_cols].sum(axis=1) if event_cols else 0
    df["active_master"] = df["participation_count_master"] > 0

    ctx = {
        "name_col": name_col,
        "email_col": email_col,
        "batch_col": batch_col,
        "country_col": country_col,
        "income_col": income_col,
        "status_col": status_col,
        "payment_col": payment_col,
        "community_status_col": community_status_col,
        "event_cols": event_cols,
    }
    return df, ctx



def parse_activity_sheet(raw: pd.DataFrame, sheet_name: str):
    header_row = 5
    data_start = 6
    if raw.shape[0] <= header_row:
        raise ValueError(f"Sheet too short: {sheet_name}")

    type_row = raw.iloc[0].tolist() if raw.shape[0] > 0 else []
    event_row = raw.iloc[1].tolist() if raw.shape[0] > 1 else []
    date_row = raw.iloc[2].tolist() if raw.shape[0] > 2 else []
    header_cells = raw.iloc[header_row].tolist()

    cols = []
    event_rows = []
    for idx, h in enumerate(header_cells):
        header_name = clean_text(h)
        event_name = clean_text(event_row[idx]) if idx < len(event_row) else ""
        event_type = clean_text(type_row[idx]) if idx < len(type_row) else ""
        event_date = parse_event_date(date_row[idx]) if idx < len(date_row) else pd.NaT
        if header_name:
            cols.append(header_name)
            if idx >= 19 and (event_name or event_type or pd.notna(event_date)):
                event_rows.append({
                    "column_name": header_name,
                    "event_name": event_name or header_name,
                    "event_type": event_type or "Other",
                    "event_date": event_date,
                    "sheet": sheet_name,
                })
        elif event_name or event_type or pd.notna(event_date):
            synthetic = f"EVENT_{idx}"
            cols.append(synthetic)
            event_rows.append({
                "column_name": synthetic,
                "event_name": event_name or synthetic,
                "event_type": event_type or "Other",
                "event_date": event_date,
                "sheet": sheet_name,
            })
        else:
            cols.append(f"Unnamed_{idx}")

    cols = make_unique(cols)
    for row in event_rows:
        if row["column_name"].startswith("EVENT_"):
            i = int(row["column_name"].split("_")[-1])
            row["column_name"] = cols[i]

    df = raw.iloc[data_start:].copy().reset_index(drop=True)
    df.columns = cols
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["student name", "name"])
    email_col = best_matching_col(df, ["email"])
    batch_col = best_matching_col(df, ["batch"])
    country_col = best_matching_col(df, ["country"])
    income_col = best_matching_col(df, ["income"])
    mobile_col = best_matching_col(df, ["mobile"])
    community_status_col = best_matching_col(df, ["community status", "admitted group"])
    term_zero_col = best_matching_col(df, ["term zero group"])
    payment_status_col = best_matching_col(df, ["payment status", "status"])
    payment_date_col = best_matching_col(df, ["payment date"])
    engagement_pct_col = best_matching_col(df, ["overall engagement %", "engagement %"])
    engagement_score_col = best_matching_col(df, ["overall engagement score", "engagement score"])

    if not name_col:
        raise ValueError(f"Name column not found in {sheet_name}")

    df = df[df[name_col].apply(is_valid_student_name)].copy()
    df["Program"] = infer_program_from_sheet(sheet_name)
    df["source_sheet"] = sheet_name
    df["student_name"] = df[name_col].map(clean_text)
    df["student_key"] = df["student_name"].map(normalize_name)
    df["email_key"] = df[email_col].map(normalize_email) if email_col else ""
    df["Batch"] = df[batch_col].map(clean_text) if batch_col else infer_batch_group_from_sheet_name(sheet_name)
    community_base = df[community_status_col].map(clean_text) if community_status_col else pd.Series("", index=df.index)
    if term_zero_col:
        term_zero_series = df[term_zero_col].map(clean_text)
        community_base = community_base.where(term_zero_series.eq(""), term_zero_series)
    df["community_status_value"] = community_base.map(normalize_community_status)

    event_info = pd.DataFrame(event_rows, columns=["column_name", "event_name", "event_type", "event_date", "sheet"])
    event_cols = [c for c in event_info["column_name"].tolist() if c in df.columns] if not event_info.empty else []

    for c in event_cols:
        df[c] = df[c].apply(normalize_yes_no).astype(int)

    df["participation_count"] = df[event_cols].sum(axis=1) if event_cols else 0
    if engagement_score_col:
        df["engagement_score"] = pd.to_numeric(df[engagement_score_col], errors="coerce").fillna(0)
    else:
        first_col = df.columns[0]
        df["engagement_score"] = pd.to_numeric(df[first_col], errors="coerce").fillna(df["participation_count"])
    if engagement_pct_col:
        df["engagement_pct"] = pd.to_numeric(df[engagement_pct_col], errors="coerce").fillna(0)
        if df["engagement_pct"].max() <= 1.05:
            df["engagement_pct"] = df["engagement_pct"] * 100
    else:
        total_events = max(len(event_cols), 1)
        df["engagement_pct"] = (df["participation_count"] / total_events) * 100

    if payment_date_col:
        df[payment_date_col] = df[payment_date_col].apply(parse_date_safe)
        df["payment_date_parsed"] = df[payment_date_col]
    else:
        df["payment_date_parsed"] = pd.NaT

    stat_series = df[payment_status_col].astype(str).str.lower().str.strip() if payment_status_col else pd.Series("", index=df.index)
    df["sheet_status_raw"] = df[payment_status_col].map(clean_text) if payment_status_col else ""
    df["sheet_is_refunded"] = stat_series.str.contains("refund", na=False)
    df["sheet_is_paid"] = stat_series.eq("admitted")
    df["is_active"] = df["participation_count"] > 0

    ctx = {
        "name_col": name_col,
        "email_col": email_col,
        "batch_col": batch_col,
        "country_col": country_col,
        "income_col": income_col,
        "mobile_col": mobile_col,
        "community_status_col": community_status_col,
        "payment_status_col": payment_status_col,
        "payment_date_col": payment_date_col,
        "engagement_score_col": engagement_score_col,
        "engagement_pct_col": engagement_pct_col,
        "event_info": event_info,
        "event_cols": event_cols,
    }
    return df, ctx



def reconcile_master_with_tx(master_df, tx_df):
    tx_by_email = {}
    tx_by_name = {}
    for _, row in tx_df.iterrows():
        email = row.get("email_key", "")
        name = row.get("student_key", "")
        if email and email not in tx_by_email:
            tx_by_email[email] = row
        if name and name not in tx_by_name:
            tx_by_name[name] = row

    resolved_status, resolved_payment, resolved_source = [], [], []
    for _, row in master_df.iterrows():
        match = None
        email = row.get("email_key", "")
        name = row.get("student_key", "")
        if email and email in tx_by_email:
            match = tx_by_email[email]
        elif name and name in tx_by_name:
            match = tx_by_name[name]

        if match is not None:
            status = clean_text(match.get("tx_status", "")) or clean_text(match.get("sheet_status_raw", ""))
            pay_dt = match.get("tx_payment_date", pd.NaT)
            resolved_status.append(status)
            resolved_payment.append(pay_dt if pd.notna(pay_dt) else pd.NaT)
            resolved_source.append(match.get("source_sheet", ""))
        else:
            if row.get("master_is_refunded", False):
                status = "Refunded"
            elif row.get("master_is_paid", False):
                status = "Admitted"
            else:
                status = clean_text(row.get("master_status_value", ""))
            resolved_status.append(status)
            resolved_payment.append(pd.NaT)
            resolved_source.append("")

    out = master_df.copy()
    out["resolved_status"] = resolved_status
    out["resolved_payment_date"] = resolved_payment
    out["resolved_tx_source"] = resolved_source

    status_lower = out["resolved_status"].astype(str).str.lower().str.strip()
    out["is_refunded"] = status_lower.str.contains("refund", na=False)
    out["is_paid"] = status_lower.eq("admitted")
    out["status_bucket"] = np.select(
        [out["is_refunded"], out["is_paid"]],
        ["Refunded", "Paid / Admitted"],
        default="Not Paid",
    )
    out["paid_label"] = out["status_bucket"]
    out["is_active"] = out["active_master"]
    return out


@st.cache_data(show_spinner=False, ttl=180)
def load_dashboard_data(source_mode: str, spreadsheet_id=None, file_bytes=None):
    sheet_names = get_sheet_names(source_mode, spreadsheet_id, file_bytes)
    missing = [s for s in ALL_REQUIRED if s not in sheet_names]

    masters, master_ctx = {}, {}
    activities, activity_ctx = {}, {}

    for sheet in MASTER_SHEETS:
        if sheet in sheet_names:
            raw = load_raw_sheet(source_mode, sheet, spreadsheet_id, file_bytes)
            masters[sheet], master_ctx[sheet] = parse_master_sheet(raw, "UG" if sheet.endswith("UG") else "PG", sheet)

    for sheet in UG_BATCH_SHEETS + PG_BATCH_SHEETS + TX_SHEETS:
        if sheet in sheet_names:
            raw = load_raw_sheet(source_mode, sheet, spreadsheet_id, file_bytes)
            activities[sheet], activity_ctx[sheet] = parse_activity_sheet(raw, sheet)

    tx_df = pd.concat([activities[s] for s in TX_SHEETS if s in activities], ignore_index=True) if any(s in activities for s in TX_SHEETS) else pd.DataFrame()
    if not tx_df.empty:
        tx_df = tx_df.copy()
        tx_df["tx_status"] = tx_df.get("sheet_status_raw", "")
        tx_df["tx_payment_date"] = tx_df.get("payment_date_parsed", pd.NaT)

    overview_frames = []
    for sheet in MASTER_SHEETS:
        if sheet in masters:
            prog = "UG" if sheet.endswith("UG") else "PG"
            tx_prog = tx_df[tx_df["Program"] == prog] if not tx_df.empty else pd.DataFrame()
            overview_frames.append(reconcile_master_with_tx(masters[sheet], tx_prog))

    overview_df = pd.concat(overview_frames, ignore_index=True) if overview_frames else pd.DataFrame()

    combined_profiles = []
    if not overview_df.empty:
        combined_profiles.append(overview_df.assign(profile_source="master"))
    for s, df in activities.items():
        combined_profiles.append(df.assign(profile_source=s))
    profile_df = pd.concat(combined_profiles, ignore_index=True) if combined_profiles else pd.DataFrame()

    return {
        "sheet_names": sheet_names,
        "missing": missing,
        "masters": masters,
        "master_ctx": master_ctx,
        "activities": activities,
        "activity_ctx": activity_ctx,
        "overview_df": overview_df,
        "profile_df": profile_df,
        "tx_df": tx_df,
    }


# ---------------- Rendering ----------------



def compute_tx_prepayment_event_type_summary(tx_df, tx_program, data):
    if tx_df is None or tx_df.empty:
        return pd.DataFrame(columns=["event_type", "Students Attended", "Attended %"])

    batch_sheets = UG_BATCH_SHEETS if tx_program == "UG" else PG_BATCH_SHEETS
    available_batch_sheets = [s for s in batch_sheets if s in data.get("activities", {})]
    if not available_batch_sheets:
        return pd.DataFrame(columns=["event_type", "Students Attended", "Attended %"])

    type_to_students = {}
    total_students = int(len(tx_df))

    for _, tx_row in tx_df.iterrows():
        email_key = clean_text(tx_row.get("email_key", ""))
        student_key = clean_text(tx_row.get("student_key", ""))
        pay_dt = tx_row.get("payment_date_parsed", pd.NaT)
        pay_dt = pd.to_datetime(pay_dt, errors="coerce")

        matched_batch_row = None
        matched_ctx = None
        for sheet in available_batch_sheets:
            batch_df = data["activities"].get(sheet, pd.DataFrame())
            if batch_df.empty:
                continue
            part = batch_df[(batch_df.get("email_key", "") == email_key) | (batch_df.get("student_key", "") == student_key)]
            if not part.empty:
                matched_batch_row = part.iloc[0]
                matched_ctx = data["activity_ctx"].get(sheet, {})
                break

        if matched_batch_row is None or matched_ctx is None:
            continue

        event_info = matched_ctx.get("event_info", pd.DataFrame())
        if event_info is None or event_info.empty:
            continue

        student_types = set()
        for _, ev in event_info.iterrows():
            col = ev.get("column_name")
            if not col or col not in matched_batch_row.index:
                continue
            attended = pd.to_numeric(pd.Series([matched_batch_row.get(col, 0)]), errors="coerce").fillna(0).iloc[0]
            if attended <= 0:
                continue
            ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
            if pd.notna(pay_dt) and pd.notna(ev_date) and ev_date >= pay_dt:
                continue
            ev_type = clean_text(ev.get("event_type", "Other")) or "Other"
            student_types.add(ev_type)

        for ev_type in student_types:
            type_to_students.setdefault(ev_type, set()).add(student_key or email_key or clean_text(tx_row.get("student_name", "")))

    rows = []
    for ev_type, students in type_to_students.items():
        cnt = len([s for s in students if clean_text(s)])
        rows.append({
            "event_type": ev_type,
            "Students Attended": cnt,
            "Attended %": round((cnt / total_students * 100), 2) if total_students else 0.0,
        })
    return pd.DataFrame(rows).sort_values(["Attended %", "Students Attended", "event_type"], ascending=[False, False, True]) if rows else pd.DataFrame(columns=["event_type", "Students Attended", "Attended %"])


def render_live_ist_clock(connected_ok: bool, connection_note: str):
    status_html = live_status_html(connected_ok, connection_note or "Google Sheets")
    html = f"""
    <div style="display:flex; justify-content:flex-end; align-items:center; gap:14px; margin-bottom:10px; font-family: Arial, sans-serif;">
        {status_html}
        <div id="ist-live-clock" style="padding:10px 14px; border-radius:999px; border:1px solid #dbeee0; background:#ffffff; color:#0b3d2e; font-weight:700; min-width:300px; text-align:center; box-shadow:0 2px 10px rgba(11, 61, 46, 0.05);">IST · --</div>
    </div>
    <script>
    const pad=(n)=>String(n).padStart(2,'0');
    function formatIST() {{
        const now = new Date();
        const parts = new Intl.DateTimeFormat('en-IN', {{
            timeZone: 'Asia/Kolkata',
            weekday: 'short',
            day: '2-digit',
            month: 'short',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit',
            hour12: true
        }}).format(now);
        return 'IST · ' + parts;
    }}
    function updateISTClock() {{
        const el = document.getElementById('ist-live-clock');
        if (el) el.textContent = formatIST();
    }}
    updateISTClock();
    setInterval(updateISTClock, 1000);
    </script>
    """
    components.html(html, height=70)


def render_header(cfg):
    c1, c2 = st.columns([5.5, 1.9])
    with c1:
        logo = find_logo_path()
        hero_html = '<div class="hero-card"><div style="font-size:30px; font-weight:900; color:#0b3d2e;">Tetr Analytics Dashboard</div><div style="margin-top:6px; color:#2e6b57; font-weight:600;">Live overview, batch analytics, and student-level tracking across Master, Batch, and Tetr-X sheets.</div><div style="margin-top:10px; color:#5b7f6e; font-size:13px; font-weight:600;">Developed by <span style="color:#0b3d2e; font-weight:800;">Harashit Mitra</span></div></div>'
        if logo is not None:
            a, b = st.columns([0.12, 0.88])
            with a:
                st.image(str(logo), width=72)
            with b:
                st.markdown(hero_html, unsafe_allow_html=True)
        else:
            st.markdown(hero_html, unsafe_allow_html=True)
    with c2:
        render_live_ist_clock(cfg["connected_ok"], cfg["connection_note"])


def overview_metrics(overview_df):
    total_students = int(len(overview_df))
    total_active = int(overview_df["is_active"].sum()) if not overview_df.empty else 0
    total_paid = int(overview_df["is_paid"].sum()) if not overview_df.empty else 0
    total_refunded = int(overview_df["is_refunded"].sum()) if not overview_df.empty else 0
    ug_students = int((overview_df["Program"] == "UG").sum()) if not overview_df.empty else 0
    pg_students = int((overview_df["Program"] == "PG").sum()) if not overview_df.empty else 0
    ug_paid = int(((overview_df["Program"] == "UG") & (overview_df["is_paid"])).sum()) if not overview_df.empty else 0
    pg_paid = int(((overview_df["Program"] == "PG") & (overview_df["is_paid"])).sum()) if not overview_df.empty else 0
    ug_refunded = int(((overview_df["Program"] == "UG") & (overview_df["is_refunded"])).sum()) if not overview_df.empty else 0
    pg_refunded = int(((overview_df["Program"] == "PG") & (overview_df["is_refunded"])).sum()) if not overview_df.empty else 0
    return total_students, total_active, total_paid, total_refunded, ug_students, pg_students, ug_paid, pg_paid, ug_refunded, pg_refunded



def build_status_breakdown(df, status_col="sheet_status_raw"):
    if status_col not in df.columns:
        return pd.DataFrame(columns=["Status", "Students"])
    s = df[status_col].map(clean_text)
    s = s[~s.map(is_numeric_or_percent_text)]
    s = s.replace("", "Unspecified")
    if s.empty:
        return pd.DataFrame(columns=["Status", "Students"])
    out = s.value_counts(dropna=False).reset_index()
    out.columns = ["Status", "Students"]
    return out


def payment_percentage_by_country(overview_df, country_col):
    if not country_col or country_col not in overview_df.columns:
        return pd.DataFrame(columns=[country_col or "Country", "Paid Students", "Paid Student %"])
    paid_df = overview_df[overview_df["is_paid"]].copy()
    if paid_df.empty:
        return pd.DataFrame(columns=[country_col or "Country", "Paid Students", "Paid Student %"])
    grp = paid_df.groupby(country_col, dropna=False).agg(**{"Paid Students": ("student_name", "count")}).reset_index()
    grp[country_col] = grp[country_col].replace("", "Unknown")
    total_paid = grp["Paid Students"].sum()
    grp["Paid Student %"] = np.where(total_paid > 0, grp["Paid Students"] / total_paid * 100, 0.0)
    return grp.sort_values(["Paid Student %", "Paid Students"], ascending=[False, False])


def render_overview(data):
    st.subheader("Overview")
    overview_df = data["overview_df"]
    if overview_df.empty:
        st.warning("Master UG / Master PG could not be loaded.")
        return

    name_col = "student_name"
    master_ug_ctx = data["master_ctx"].get("Master UG", {})
    master_pg_ctx = data["master_ctx"].get("Master PG", {})
    country_col = master_ug_ctx.get("country_col") or master_pg_ctx.get("country_col")
    income_col = master_ug_ctx.get("income_col") or master_pg_ctx.get("income_col")

    total_students, total_active, total_paid, total_refunded, ug_students, pg_students, ug_paid, pg_paid, ug_refunded, pg_refunded = overview_metrics(overview_df)
    active_rate = round((total_active / total_students * 100), 1) if total_students else 0
    paid_rate = round((total_paid / total_students * 100), 1) if total_students else 0
    refunded_rate = round((total_refunded / total_students * 100), 1) if total_students else 0

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Students", f"{total_students:,}")
    m2.metric("Active Students", f"{total_active:,}", delta=f"{active_rate}% active")
    m3.metric("Paid / Admitted", f"{total_paid:,}", delta=f"{paid_rate}% paid")
    m4.metric("Refunded", f"{total_refunded:,}", delta=f"{refunded_rate}% refunded")
    m5.metric("UG vs PG", f"{ug_students:,} / {pg_students:,}")

    g1, g2, g3 = st.columns([1.2, 1, 1])
    with g1:
        st.plotly_chart(gauge_chart(total_students, "Total Students", maximum=max(total_students, 1)), use_container_width=True, key="overview_gauge")
    with g2:
        st.plotly_chart(donut_chart(["UG", "PG"], [ug_students, pg_students], "UG / PG Distribution"), use_container_width=True, key="overview_program_donut")
    with g3:
        st.plotly_chart(donut_chart(["Paid / Admitted", "Refunded", "Not Paid"], [total_paid, total_refunded, max(total_students - total_paid - total_refunded, 0)], "Overall Status Distribution"), use_container_width=True, key="overview_paid_donut")

    a1, a2 = st.columns(2)
    with a1:
        batch_plot = overview_df.groupby(["Program", "Batch"], dropna=False)[name_col].count().reset_index(name="Students")
        batch_plot["Batch"] = batch_plot["Batch"].replace("", "Unknown")
        fig = px.bar(batch_plot, x="Batch", y="Students", color="Program", barmode="group", title="Students by Batch", color_discrete_map={"UG": GREEN, "PG": GREEN_3})
        st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-25), use_container_width=True, key="overview_batch_bar")
    with a2:
        status_plot = overview_df.groupby(["Program", "status_bucket"])[name_col].count().reset_index(name="Students")
        fig = px.bar(status_plot, x="Program", y="Students", color="status_bucket", barmode="group", title="Status Distribution by Program",
                     color_discrete_map={"Paid / Admitted": GREEN, "Refunded": RED, "Not Paid": GREEN_4})
        st.plotly_chart(nice_layout(fig, height=380), use_container_width=True, key="overview_status_bar")

    b1, b2 = st.columns(2)
    with b1:
        if country_col and country_col in overview_df.columns:
            country_students = overview_df.groupby(country_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False).head(15)
            country_students[country_col] = country_students[country_col].replace("", "Unknown")
            fig = px.bar(country_students, x=country_col, y="Students", title="Country Distribution of Students")
            fig.update_traces(marker_color=GREEN_3)
            st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-30), use_container_width=True, key="overview_country_students_bar")
    with b2:
        comm = overview_df["community_status_value"].replace("", np.nan).dropna()
        if not comm.empty:
            community_plot = comm.value_counts().reset_index()
            community_plot.columns = ["Community Status", "Students"]
            fig = px.pie(community_plot, names="Community Status", values="Students", hole=0.6, title="Community Status Overview",
                         color="Community Status", color_discrete_map={"Tetr X": GREEN, "In": GREEN_3, "Out": GREEN_4})
            st.plotly_chart(nice_layout(fig, height=400), use_container_width=True, key="overview_comm_donut")

    c1, c2 = st.columns(2)
    with c1:
        if income_col and income_col in overview_df.columns:
            income_plot = overview_df.groupby(income_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False)
            fig = px.bar(income_plot, x=income_col, y="Students", title="Income Distribution")
            fig.update_traces(marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-25), use_container_width=True, key="overview_income_bar")
    with c2:
        country_pay = payment_percentage_by_country(overview_df, country_col)
        if not country_pay.empty:
            fig = px.bar(country_pay.head(15), x=country_col, y="Paid Student %", hover_data=["Paid Students"], title="Paid Students Country-wise % Distribution")
            fig.update_traces(marker_color=GREEN_2)
            st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-30), use_container_width=True, key="overview_country_payment_bar")

    d1, d2 = st.columns(2)
    with d1:
        status_circle = pd.DataFrame({
            "Metric": ["Active", "Paid", "Refunded"],
            "Count": [total_active, total_paid, total_refunded]
        })
        fig = px.line_polar(status_circle, r="Count", theta="Metric", line_close=True, title="Overview Activity Circle")
        fig.update_traces(fill='toself', line_color=GREEN, marker_color=GREEN)
        st.plotly_chart(nice_layout(fig, height=400), use_container_width=True, key="overview_circle_bar")
    with d2:
        active_circle = pd.DataFrame({"Status": ["Active", "Non-Active"], "Students": [total_active, max(total_students - total_active, 0)]})
        fig = px.pie(active_circle, names="Status", values="Students", hole=0.58, title="Active vs Non-Active",
                     color="Status", color_discrete_map={"Active": GREEN, "Non-Active": GREEN_4})
        st.plotly_chart(nice_layout(fig, height=400), use_container_width=True, key="overview_active_circle")

    display_cols = [c for c in ["student_name", "Program", "Batch", country_col, income_col, "community_status_value", "resolved_status", "resolved_payment_date", "status_bucket"] if c and c in overview_df.columns]
    st.markdown("#### Overview Table")
    st.dataframe(overview_df[display_cols].sort_values(["Program", "Batch", "student_name"]), use_container_width=True, height=420, key="overview_table")



def render_sheet_detail(sheet_name, df, ctx, prefix, data=None):
    st.markdown(f"#### {sheet_name}")
    if df.empty:
        st.warning(f"No data available for {sheet_name}.")
        return

    total_students = int(len(df))
    active_students = int(df["is_active"].sum()) if "is_active" in df else int((pd.to_numeric(df["engagement_score"], errors="coerce").fillna(0) > 0).sum())
    paid_students = int(df["sheet_is_paid"].sum()) if "sheet_is_paid" in df else 0
    refunded_students = int(df["sheet_is_refunded"].sum()) if "sheet_is_refunded" in df else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students", f"{total_students:,}")
    k2.metric("Active", f"{active_students:,}", delta=f"{(active_students/total_students*100 if total_students else 0):.1f}%")
    k3.metric("Admitted / Paid", f"{paid_students:,}", delta=f"{(paid_students/total_students*100 if total_students else 0):.1f}%")
    k4.metric("Refunded", f"{refunded_students:,}", delta=f"{(refunded_students/total_students*100 if total_students else 0):.1f}%")

    event_info = ctx["event_info"]
    c1, c2, c3 = st.columns(3)
    with c1:
        fig = px.histogram(df, x="engagement_pct", nbins=12, title="Engagement Distribution")
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=340), use_container_width=True, key=f"{prefix}_hist")
    with c2:
        status = build_status_breakdown(df)
        fig = px.pie(status, names="Status", values="Students", hole=0.58, title="Status Breakdown")
        st.plotly_chart(nice_layout(fig, height=340), use_container_width=True, key=f"{prefix}_pie")
    with c3:
        active_circle = pd.DataFrame({"Status": ["Active", "Non-Active"], "Students": [active_students, max(total_students - active_students, 0)]})
        fig = px.pie(active_circle, names="Status", values="Students", hole=0.58, title="Active vs Non-Active",
                     color="Status", color_discrete_map={"Active": GREEN, "Non-Active": GREEN_4})
        st.plotly_chart(nice_layout(fig, height=340), use_container_width=True, key=f"{prefix}_active_circle")

    comm = df["community_status_value"].replace("", np.nan).dropna() if "community_status_value" in df.columns else pd.Series(dtype=object)
    if not comm.empty:
        community_plot = comm.value_counts().reset_index()
        community_plot.columns = ["Community Status", "Students"]
        fig = px.pie(community_plot, names="Community Status", values="Students", hole=0.58, title="Community Status",
                     color="Community Status", color_discrete_map={"Tetr X": GREEN, "In": GREEN_3, "Out": GREEN_4})
        st.plotly_chart(nice_layout(fig, height=340), use_container_width=True, key=f"{prefix}_community")

    d1, d2 = st.columns(2)
    with d1:
        event_info = ctx["event_info"]
        if not event_info.empty:
            participants = []
            for _, r in event_info.iterrows():
                col = r["column_name"]
                participants.append(int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum()))
            event_counts = event_info.assign(Participants=participants).sort_values("Participants", ascending=False).head(12)
            fig = px.bar(event_counts, x="Participants", y="event_name", orientation="h", color="event_type", title="Top Events by Participation")
            st.plotly_chart(nice_layout(fig, height=460), use_container_width=True, key=f"{prefix}_events")
    with d2:
        country_col = ctx.get("country_col")
        if country_col and country_col in df.columns:
            top_country = df.groupby(country_col)["student_name"].count().reset_index(name="Students").sort_values("Students", ascending=False).head(10)
            fig = px.bar(top_country, x=country_col, y="Students", title="Country Split")
            fig.update_traces(marker_color=GREEN_3)
            st.plotly_chart(nice_layout(fig, height=430, x_tickangle=-30), use_container_width=True, key=f"{prefix}_country")

    t1, t2 = st.columns(2)
    with t1:
        students = df[["student_name", "engagement_pct", "engagement_score", "community_status_value"]].sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
        st.markdown("#### Top Students")
        st.dataframe(students, use_container_width=True, height=390, key=f"{prefix}_top_df")
    with t2:
        if prefix.startswith("tx_") and data is not None:
            tx_program = infer_program_from_sheet(sheet_name)
            type_counts = compute_tx_prepayment_event_type_summary(df, tx_program, data)
            st.markdown("#### Event Type Attendance Summary")
            st.caption("Based on each Tetr-X student's attended events in their respective batch sheet before their payment date.")
            if not type_counts.empty:
                fig = px.bar(type_counts, x="event_type", y="Attended %", text="Students Attended", title="Pre-Payment Batch Attendance by Event Type", hover_data=["Students Attended"], color="event_type")
                fig.update_traces(textposition="outside")
                st.plotly_chart(nice_layout(fig, height=390, x_tickangle=-25), use_container_width=True, key=f"{prefix}_event_type_attendance")
                st.dataframe(type_counts.rename(columns={"event_type": "Event Type"}), use_container_width=True, height=190, key=f"{prefix}_event_type_df")
            else:
                st.info("No pre-payment batch attendance was found for the students in this Tetr-X sheet.")
        else:
            target = df[(~df["sheet_is_paid"]) & (~df["sheet_is_refunded"]) & (df["is_active"])][["student_name", "engagement_pct", "engagement_score", "community_status_value"]].sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
            st.markdown("#### Best Upgrade Targets")
            st.dataframe(target, use_container_width=True, height=390, key=f"{prefix}_upgrade_df")

    if not event_info.empty and event_info["event_date"].notna().any():
        participants = []
        for _, r in event_info.iterrows():
            col = r["column_name"]
            participants.append(int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum()))
        timeline = event_info.assign(Participants=participants).dropna(subset=["event_date"]).sort_values("event_date")
        fig = px.line(timeline, x="event_date", y="Participants", markers=True, title="Participation Timeline")
        fig.update_traces(line_color=GREEN, marker_color=GREEN)
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{prefix}_timeline")


def render_student_profile(data):
    st.subheader("Student Profile")
    overview_df = data["overview_df"]
    if overview_df.empty:
        st.warning("Master sheets not available.")
        return

    students = overview_df[["student_name", "email_key", "student_key", "Program"]].drop_duplicates().sort_values("student_name")
    search = st.text_input("Search student names", value="", placeholder="Type a name...")
    options = students[students["student_name"].str.contains(search, case=False, na=False)]["student_name"].tolist() if search else students["student_name"].tolist()
    selected = st.multiselect("Select one or more students", options=options, default=[])
    pasted = st.text_area("Or paste multiple student names (one per line)")
    pasted_names = [clean_text(x) for x in pasted.splitlines() if clean_text(x)]
    final_names = list(dict.fromkeys(selected + pasted_names))

    if not final_names:
        st.info("Search and select a student to view the profile.")
        return

    for i, student_name in enumerate(final_names):
        matches = overview_df[overview_df["student_name"].str.lower() == student_name.lower()]
        if matches.empty:
            matches = overview_df[overview_df["student_key"] == normalize_name(student_name)]
        if matches.empty:
            st.warning(f"No master profile found for {student_name}")
            continue

        master = matches.iloc[0]
        email_key = master.get("email_key", "")
        name_key = master.get("student_key", "")

        related = []
        for sheet, df in data["activities"].items():
            part = df[(df["email_key"] == email_key) | (df["student_key"] == name_key)].copy()
            if not part.empty:
                related.append(part)
        related_df = pd.concat(related, ignore_index=True) if related else pd.DataFrame()

        st.markdown("---")
        st.markdown(f"### {master['student_name']}")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Program", clean_text(master.get("Program", "")))
        p2.metric("Batch", clean_text(master.get("Batch", "")))
        p3.metric("Paid Status", clean_text(master.get("resolved_status", "Not Paid")))
        pay_dt = master.get("resolved_payment_date", pd.NaT)
        p4.metric("Payment Date", pay_dt.strftime("%Y-%m-%d") if pd.notna(pay_dt) else "—")

        info_cols = [c for c in ["student_name", "Email", "email_key"] if c in overview_df.columns]
        c1, c2 = st.columns([1.2, 1])
        with c1:
            master_display = {
                "Name": master.get("student_name", ""),
                "Email": clean_text(master.get("email_key", "")),
                "Program": clean_text(master.get("Program", "")),
                "Batch": clean_text(master.get("Batch", "")),
                "Country": clean_text(master.get(data["master_ctx"]["Master UG"]["country_col"] if master.get("Program") == "UG" and "Master UG" in data["master_ctx"] else data["master_ctx"].get("Master PG", {}).get("country_col", ""), "")),
                "Income": clean_text(master.get(data["master_ctx"]["Master UG"]["income_col"] if master.get("Program") == "UG" and "Master UG" in data["master_ctx"] else data["master_ctx"].get("Master PG", {}).get("income_col", ""), "")),
                "Status": clean_text(master.get("resolved_status", "")),
                "Payment": clean_text(master.get("master_payment_value", "")),
            }
            st.dataframe(pd.DataFrame(master_display.items(), columns=["Field", "Value"]), use_container_width=True, hide_index=True, key=f"profile_info_{i}")
        with c2:
            if related_df.empty:
                st.info("No matching batch / Tetr-X records found.")
            else:
                total_events = int(related_df["participation_count"].sum()) if "participation_count" in related_df else 0
                distinct_sheets = related_df["source_sheet"].nunique()
                active_records = int((related_df["engagement_pct"] > 0).sum()) if "engagement_pct" in related_df else 0
                st.metric("Total Event Participations", total_events)
                st.metric("Matched Sheets", distinct_sheets)
                st.metric("Active Records", active_records)

        if not related_df.empty:
            # event type summary
            event_rows = []
            for sheet, ctx in data["activity_ctx"].items():
                part = data["activities"][sheet][(data["activities"][sheet]["email_key"] == email_key) | (data["activities"][sheet]["student_key"] == name_key)]
                if part.empty or ctx["event_info"].empty:
                    continue
                row = part.iloc[0]
                for _, ev in ctx["event_info"].iterrows():
                    count = int(row.get(ev["column_name"], 0))
                    if count > 0:
                        event_rows.append({
                            "sheet": sheet,
                            "event_name": ev["event_name"],
                            "event_type": ev["event_type"],
                            "event_date": ev["event_date"],
                            "count": count,
                        })
            ev_df = pd.DataFrame(event_rows)
            if not ev_df.empty:
                type_df = ev_df.groupby("event_type")["count"].sum().reset_index().sort_values("count", ascending=False)
                total = type_df["count"].sum()
                type_df["percentage"] = np.where(total > 0, type_df["count"] / total * 100, 0)

                x1, x2 = st.columns(2)
                with x1:
                    fig = px.bar(type_df, x="event_type", y="count", title="Event Type Participation")
                    fig.update_traces(marker_color=GREEN_2)
                    st.plotly_chart(nice_layout(fig, height=340, x_tickangle=-25), use_container_width=True, key=f"profile_type_bar_{i}")
                with x2:
                    fig = px.pie(type_df, names="event_type", values="count", hole=0.58, title="Event Type % Share")
                    st.plotly_chart(nice_layout(fig, height=340), use_container_width=True, key=f"profile_type_pie_{i}")

                timeline = ev_df.dropna(subset=["event_date"]).sort_values("event_date")
                if not timeline.empty:
                    timeline = timeline.groupby(["event_date", "event_name"], as_index=False)["count"].sum()
                    fig = px.line(timeline, x="event_date", y="count", markers=True, title="Engagement Timeline")
                    fig.update_traces(line_color=GREEN, marker_color=GREEN)
                    if pd.notna(pay_dt):
                        x = pd.Timestamp(pay_dt)
                        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper", line=dict(color=RED, width=2, dash="dash"))
                        fig.add_annotation(x=x, y=1, yref="paper", text="Payment Date", showarrow=False, font=dict(color=RED), bgcolor="white")
                    st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"profile_timeline_{i}")

                st.markdown("#### Event Details")
                show = ev_df.sort_values(["event_date", "event_type", "event_name"], ascending=[True, True, True]).copy()
                st.dataframe(show, use_container_width=True, height=320, key=f"profile_events_{i}")
            else:
                st.info("Matched records found, but no attended events were recorded for this student.")

            st.markdown("#### Matched Batch / Tetr-X Records")
            record_cols = [c for c in ["source_sheet", "Batch", "engagement_pct", "engagement_score", "sheet_status_raw", "payment_date_parsed"] if c in related_df.columns]
            st.dataframe(related_df[record_cols].sort_values(["source_sheet", "Batch"]), use_container_width=True, height=250, key=f"profile_records_{i}")


def render_program_page(title, sheets, data, page_prefix):
    st.subheader(title)
    available = [s for s in sheets if s in data["activities"]]
    if not available:
        st.warning("No sheets available for this section.")
        return
    tabs = st.tabs(available)
    for tab, sheet in zip(tabs, available):
        with tab:
            render_sheet_detail(sheet, data["activities"][sheet], data["activity_ctx"][sheet], f"{page_prefix}_{sheet}", data=data)



def render_ug_vs_pg_page(data):
    st.subheader("UG vs PG")
    ug_frames = [data["activities"][s] for s in UG_BATCH_SHEETS if s in data["activities"]]
    pg_frames = [data["activities"][s] for s in PG_BATCH_SHEETS if s in data["activities"]]
    if not ug_frames or not pg_frames:
        st.warning("UG or PG batch sheets are missing.")
        return

    ug = pd.concat(ug_frames, ignore_index=True)
    pg = pd.concat(pg_frames, ignore_index=True)

    def pct(n, d):
        return round((float(n) / float(d) * 100), 1) if d else 0.0

    comp = pd.DataFrame({
        "Program": ["UG", "PG"],
        "Students": [len(ug), len(pg)],
        "Active %": [pct(ug["is_active"].sum(), len(ug)), pct(pg["is_active"].sum(), len(pg))],
        "Paid %": [pct(ug["sheet_is_paid"].sum(), len(ug)), pct(pg["sheet_is_paid"].sum(), len(pg))],
        "Refunded %": [pct(ug["sheet_is_refunded"].sum(), len(ug)), pct(pg["sheet_is_refunded"].sum(), len(pg))],
        "Tetr X %": [pct((ug.get("community_status_value", pd.Series(dtype=object)) == "Tetr X").sum(), len(ug)), pct((pg.get("community_status_value", pd.Series(dtype=object)) == "Tetr X").sum(), len(pg))],
        "In %": [pct((ug.get("community_status_value", pd.Series(dtype=object)) == "In").sum(), len(ug)), pct((pg.get("community_status_value", pd.Series(dtype=object)) == "In").sum(), len(pg))],
        "Out %": [pct((ug.get("community_status_value", pd.Series(dtype=object)) == "Out").sum(), len(ug)), pct((pg.get("community_status_value", pd.Series(dtype=object)) == "Out").sum(), len(pg))],
    })

    c1, c2, c3 = st.columns(3)
    with c1:
        melt = comp.melt(id_vars=["Program", "Students"], value_vars=["Active %", "Paid %", "Refunded %"], var_name="Metric", value_name="Percentage")
        fig = px.bar(melt, x="Metric", y="Percentage", color="Program", barmode="group", title="Core Percentage Comparison", color_discrete_map={"UG": GREEN, "PG": GREEN_3})
        st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-20), use_container_width=True, key="uvspg_core")
    with c2:
        comm = comp.melt(id_vars=["Program"], value_vars=["Tetr X %", "In %", "Out %"], var_name="Metric", value_name="Percentage")
        fig = px.bar(comm, x="Metric", y="Percentage", color="Program", barmode="group", title="Community Status % Comparison", color_discrete_map={"UG": GREEN, "PG": GREEN_3})
        st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-20), use_container_width=True, key="uvspg_commpct")
    with c3:
        circle_df = pd.DataFrame({
            "Program": ["UG"]*3 + ["PG"]*3,
            "Metric": ["Active %","Paid %","Refunded %"]*2,
            "Value": [comp.loc[0, "Active %"], comp.loc[0, "Paid %"], comp.loc[0, "Refunded %"], comp.loc[1, "Active %"], comp.loc[1, "Paid %"], comp.loc[1, "Refunded %"]]
        })
        fig = px.line_polar(circle_df, r="Value", theta="Metric", color="Program", line_close=True, title="Percentage Circle Comparison", color_discrete_map={"UG": GREEN, "PG": GREEN_3})
        fig.update_traces(fill='toself')
        st.plotly_chart(nice_layout(fig, height=380), use_container_width=True, key="uvspg_circle")

    t1, t2 = st.columns(2)
    with t1:
        pay_country_rows = []
        for label, df in [("UG", ug), ("PG", pg)]:
            country_col = next((c for c in ["Country", "country", "Country of Residence"] if c in df.columns), None)
            if country_col:
                grp = df.groupby(country_col, dropna=False).agg(Students=("student_name", "count"), Paid=("sheet_is_paid", "sum")).reset_index()
                grp["Payment %"] = np.where(grp["Students"] > 0, grp["Paid"] / grp["Students"] * 100, 0.0)
                grp["Program"] = label
                grp[country_col] = grp[country_col].replace("", "Unknown")
                pay_country_rows.append(grp.head(10))
        if pay_country_rows:
            pc = pd.concat(pay_country_rows, ignore_index=True)
            country_name_col = [c for c in pc.columns if c not in ["Students","Paid","Payment %","Program"]][0]
            fig = px.bar(pc, x=country_name_col, y="Payment %", color="Program", barmode="group", title="Country-wise Payment %", color_discrete_map={"UG": GREEN, "PG": GREEN_3})
            st.plotly_chart(nice_layout(fig, height=420, x_tickangle=-25), use_container_width=True, key="uvspg_countrypay")
    with t2:
        rows=[]
        for label, sheets in [("UG", UG_BATCH_SHEETS), ("PG", PG_BATCH_SHEETS)]:
            for sheet in [s for s in sheets if s in data["activity_ctx"]]:
                info = data["activity_ctx"][sheet]["event_info"]
                sdf = data["activities"][sheet]
                if info.empty or sdf.empty:
                    continue
                for _, r in info.iterrows():
                    denom = len(sdf) if len(sdf) else 1
                    rows.append({"Program": label, "Event Type": r["event_type"], "Participation %": round(pd.to_numeric(sdf[r["column_name"]], errors="coerce").fillna(0).sum() / denom * 100, 1)})
        if rows:
            evt = pd.DataFrame(rows).groupby(["Program", "Event Type"], as_index=False)["Participation %"].mean()
            fig = px.bar(evt, x="Event Type", y="Participation %", color="Program", barmode="group", title="Average Event Participation % by Type", color_discrete_map={"UG": GREEN, "PG": GREEN_3})
            st.plotly_chart(nice_layout(fig, height=420, x_tickangle=-25), use_container_width=True, key="uvspg_evt")

    st.markdown("#### Percentage Comparison Table")
    st.dataframe(comp, use_container_width=True, hide_index=True, key="uvspg_table")


def render_tetrx_page(data):
    st.subheader("Tetr-X")
    available = [s for s in TX_SHEETS if s in data["activities"]]
    if not available:
        st.warning("Tetr-X sheets not available.")
        return
    tx_all = pd.concat([data["activities"][s] for s in available], ignore_index=True)
    tx_students = int(len(tx_all))
    tx_active = int(tx_all["is_active"].sum()) if "is_active" in tx_all else 0
    tx_paid = int(tx_all["sheet_is_paid"].sum()) if "sheet_is_paid" in tx_all else 0
    tx_refunded = int(tx_all["sheet_is_refunded"].sum()) if "sheet_is_refunded" in tx_all else 0
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tetr-X Students", f"{tx_students:,}")
    k2.metric("Active Students", f"{tx_active:,}", delta=f"{(tx_active/tx_students*100 if tx_students else 0):.1f}%")
    k3.metric("Admitted / Paid", f"{tx_paid:,}", delta=f"{(tx_paid/tx_students*100 if tx_students else 0):.1f}%")
    k4.metric("Refunded", f"{tx_refunded:,}", delta=f"{(tx_refunded/tx_students*100 if tx_students else 0):.1f}%")
    tabs = st.tabs(available)
    for tab, sheet in zip(tabs, available):
        with tab:
            render_sheet_detail(sheet, data["activities"][sheet], data["activity_ctx"][sheet], f"tx_{sheet}", data=data)


def main():
    cfg = resolve_source()
    render_header(cfg)

    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        default_pages = ["Overview", "Student Profile", "UG", "PG", "UG vs PG", "Tetr-X"]
        default_index = default_pages.index(st.session_state.get("nav_page", "Overview")) if st.session_state.get("nav_page", "Overview") in default_pages else 0
        page = st.radio("Go to", default_pages, index=default_index, label_visibility="collapsed", key="nav")
        st.session_state["nav_page"] = page
        if cfg["source_mode"] == "excel" and cfg["file_bytes"] is None:
            st.info("Upload the workbook to use manual mode.")
        if not cfg["connected_ok"] and cfg["source_mode"] == "gsheets":
            st.error(cfg["connection_note"])

    if cfg["source_mode"] == "excel" and cfg["file_bytes"] is None:
        st.warning("Connect the Google Sheet or upload the workbook to load the dashboard.")
        return

    try:
        data = load_dashboard_data(cfg["source_mode"], spreadsheet_id=cfg["spreadsheet_id"], file_bytes=cfg["file_bytes"])
    except Exception as e:
        st.error(f"Dashboard load failed: {e}")
        return

    if data["missing"]:
        st.warning("Missing sheets: " + ", ".join(data["missing"]))

    if page == "Overview":
        render_overview(data)
    elif page == "Student Profile":
        render_student_profile(data)
    elif page == "UG":
        render_program_page("UG Batch Sheets", UG_BATCH_SHEETS, data, "ug")
    elif page == "PG":
        render_program_page("PG Batch Sheets", PG_BATCH_SHEETS, data, "pg")
    elif page == "UG vs PG":
        render_ug_vs_pg_page(data)
    elif page == "Tetr-X":
        render_tetrx_page(data)


if __name__ == "__main__":
    main()
