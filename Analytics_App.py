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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except Exception:
    GSPREAD_AVAILABLE = False

st.set_page_config(page_title="Tetr Analytics Dashboard", layout="wide")

MASTER_SHEETS = ["Master UG", "Master PG"]
UG_BATCH_SHEETS = ["UG - B1 to B4", "UG B5", "UG B6", "UG B7", "UG B8", "UG B9", "UG B10", "UG B11"]
PG_BATCH_SHEETS = ["PG - B1 & B2", "PG - B3 & B4", "PG B5"]
TX_SHEETS = ["Tetr-X-UG", "Tetr-X-PG"]
DATES_SHEET = "Dates"
WINNER_SHEET = "Winner"
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


def normalize_batch_token(x):
    s = clean_text(x).strip().upper()
    if not s:
        return ""
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", "", s)
    m = re.search(r"B?(\d+)$", s)
    if m:
        return f"B{m.group(1)}"
    m = re.search(r"B?(\d+)TOB?(\d+)", s)
    if m:
        return f"B{m.group(1)}-B{m.group(2)}"
    m = re.search(r"B?(\d+)&B?(\d+)", s)
    if m:
        return f"B{m.group(1)}-B{m.group(2)}"
    return s


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


def map_profile_plot_event_type(event_type: str) -> str:
    s = clean_text(event_type).strip().lower()
    if s in {"general", "poll", "fun", "fun task"}:
        return "General/Fun"
    if s in {"competition", "hackathon"}:
        return "Competitions"
    if s in {"masterclass", "skill bootcamp"}:
        return "Masterclasses"
    if s == "online event":
        return "Online AMAs"
    return clean_text(event_type) or "Other"


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



def parse_numeric_percent(x):
    s = clean_text(x)
    if not s:
        return np.nan
    s = s.replace(",", "").replace("%", "").strip()
    if s.lower() in {"nan", "none", "#div/0!", "inf", "-inf"}:
        return np.nan
    try:
        return float(s)
    except Exception:
        return np.nan


def format_date_display(x):
    dt = pd.to_datetime(x, errors="coerce")
    return dt.strftime("%Y-%m-%d") if pd.notna(dt) else "—"

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



def parse_dates_sheet(raw: pd.DataFrame):
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["student_name", "student_key", "email_key", "UG PG", "Batch", "Offered date", "Deadline", "offered_date_parsed", "deadline_parsed"])

    header_row = 0
    for i in range(min(10, len(raw))):
        vals = [clean_text(v).lower() for v in raw.iloc[i].tolist()]
        joined = " | ".join(vals)
        if "offered" in joined and "deadline" in joined and ("name" in joined or "email" in joined):
            header_row = i
            break

    df = raw.iloc[header_row + 1 :].copy().reset_index(drop=True)
    df.columns = make_unique(raw.iloc[header_row].tolist())
    df = df.dropna(how="all")

    name_col = next((c for c in df.columns if "name" in clean_text(c).lower()), None)
    email_col = next((c for c in df.columns if "email" in clean_text(c).lower()), None)
    program_col = next((c for c in df.columns if clean_text(c).lower() in {"ug/pg", "program", "ug pg"}), None)
    if program_col is None:
        program_col = next((c for c in df.columns if "ug/pg" in clean_text(c).lower()), None)
    batch_col = next((c for c in df.columns if "batch" in clean_text(c).lower()), None)
    offered_col = next((c for c in df.columns if "offered" in clean_text(c).lower()), None)
    deadline_col = next((c for c in df.columns if "deadline" in clean_text(c).lower()), None)

    if name_col is None and email_col is None:
        return pd.DataFrame(columns=["student_name", "student_key", "email_key", "UG PG", "Batch", "Offered date", "Deadline", "offered_date_parsed", "deadline_parsed"])

    df["student_name"] = df[name_col].map(clean_text) if name_col else ""
    df["student_key"] = df["student_name"].map(normalize_name)
    df["email_key"] = df[email_col].map(normalize_email) if email_col else ""
    df["UG PG"] = df[program_col].map(clean_text) if program_col else ""
    df["Batch"] = df[batch_col].map(normalize_batch_token) if batch_col else ""
    df["Offered date"] = df[offered_col].map(clean_text) if offered_col else ""
    df["Deadline"] = df[deadline_col].map(clean_text) if deadline_col else ""
    df["offered_date_parsed"] = df["Offered date"].apply(parse_date_safe)
    df["deadline_parsed"] = df["Deadline"].apply(parse_date_safe)

    keep = ["student_name", "student_key", "email_key", "UG PG", "Batch", "Offered date", "Deadline", "offered_date_parsed", "deadline_parsed"]
    out = df[keep].copy()
    out = out[(out["student_name"].apply(is_valid_student_name)) | (out["email_key"].astype(str).str.len() > 3)].copy()
    out = out.sort_values(["email_key", "student_key"]).drop_duplicates(subset=["email_key", "student_key", "UG PG", "Batch"], keep="first")
    return out.reset_index(drop=True)


def find_student_dates_row(dates_df: pd.DataFrame, student_name: str, email_key: str = "", student_key: str = "", program: str = "", batch: str = ""):
    if dates_df is None or dates_df.empty:
        return None

    email_key = clean_text(email_key)
    student_key = clean_text(student_key) or normalize_name(student_name)
    norm_program = clean_text(program).upper()
    norm_batch = normalize_batch_token(batch)

    mask = pd.Series(False, index=dates_df.index)
    if email_key and "email_key" in dates_df.columns:
        mask = mask | dates_df["email_key"].astype(str).eq(email_key)
    if student_key and "student_key" in dates_df.columns:
        mask = mask | dates_df["student_key"].astype(str).eq(student_key)

    cand = dates_df.loc[mask].copy()
    if cand.empty and student_name:
        cand = dates_df.loc[dates_df["student_key"].astype(str).eq(normalize_name(student_name))].copy()
    if cand.empty:
        return None

    if norm_program and "UG PG" in cand.columns:
        c2 = cand[cand["UG PG"].astype(str).str.upper().eq(norm_program)]
        if not c2.empty:
            cand = c2

    if norm_batch and "Batch" in cand.columns:
        c2 = cand[cand["Batch"].astype(str).map(normalize_batch_token).eq(norm_batch)]
        if not c2.empty:
            cand = c2

    cand = cand.sort_values(["offered_date_parsed", "deadline_parsed"], na_position="last")
    return cand.iloc[0] if not cand.empty else None

def parse_winner_sheet(raw: pd.DataFrame):
    if raw is None or raw.empty:
        return pd.DataFrame(columns=[
            "challenge_name", "winner_name", "student_key", "email_key", "batch_key",
            "amount_usd", "entry_type", "is_winner", "is_spotlight"
        ])

    header_row = 0
    df = raw.iloc[1:].copy().reset_index(drop=True)
    df.columns = make_unique(raw.iloc[header_row].tolist())
    df = df.dropna(how="all")

    challenge_col = next((c for c in df.columns if "challenge" in clean_text(c).lower()), None)
    winner_name_col = next((c for c in df.columns if "winner name" in clean_text(c).lower()), None)
    email_col = next((c for c in df.columns if "email" in clean_text(c).lower()), None)
    batch_col = next((c for c in df.columns if "batch" in clean_text(c).lower()), None)
    amount_col = next((c for c in df.columns if "amount" in clean_text(c).lower() and "usd" in clean_text(c).lower()), None)
    type_col = next((c for c in df.columns if "winner/spotlight" in clean_text(c).lower()), None)

    if winner_name_col is None and email_col is None:
        return pd.DataFrame(columns=[
            "challenge_name", "winner_name", "student_key", "email_key", "batch_key",
            "amount_usd", "entry_type", "is_winner", "is_spotlight"
        ])

    out = pd.DataFrame()
    out["challenge_name"] = df[challenge_col].map(clean_text) if challenge_col else ""
    out["winner_name"] = df[winner_name_col].map(clean_text) if winner_name_col else ""
    out["student_key"] = out["winner_name"].map(normalize_name)
    out["email_key"] = df[email_col].map(normalize_email) if email_col else ""
    out["batch_key"] = df[batch_col].map(normalize_batch_token) if batch_col else ""
    if amount_col:
        amt = df[amount_col].astype(str).str.replace(",", "", regex=False).str.extract(r'([-+]?\d*\.?\d+)')[0]
        out["amount_usd"] = pd.to_numeric(amt, errors="coerce").fillna(0.0)
    else:
        out["amount_usd"] = 0.0
    out["entry_type"] = df[type_col].map(clean_text) if type_col else ""
    out["is_winner"] = out["entry_type"].astype(str).str.lower().eq("winner")
    out["is_spotlight"] = out["entry_type"].astype(str).str.lower().eq("spotlight")

    out = out[(out["winner_name"].apply(is_valid_student_name)) | (out["email_key"].astype(str).str.len() > 3)].copy()
    out = out[out["challenge_name"].astype(str).str.strip().ne("") | out["is_winner"] | out["is_spotlight"]].copy()
    return out.reset_index(drop=True)


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
    payment_date_col = best_matching_col(df, ["payment date", "date of payment", "paid date"])
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
    if payment_date_col:
        df["master_payment_date_parsed"] = df[payment_date_col].apply(parse_date_safe)
    else:
        df["master_payment_date_parsed"] = df["master_payment_value"].apply(parse_date_safe)

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
    df["is_paid"] = df["master_is_paid"]
    df["is_refunded"] = df["master_is_refunded"]
    df["status_bucket"] = np.select(
        [df["is_refunded"], df["is_paid"]],
        ["Refunded", "Paid / Admitted"],
        default="Not Paid",
    )
    df["paid_label"] = df["status_bucket"]
    df["resolved_status"] = df["master_status_value"]
    df["resolved_payment_date"] = df["master_payment_date_parsed"]
    df["is_active"] = df["active_master"]

    ctx = {
        "name_col": name_col,
        "email_col": email_col,
        "batch_col": batch_col,
        "country_col": country_col,
        "income_col": income_col,
        "status_col": status_col,
        "payment_col": payment_col,
        "payment_date_col": payment_date_col,
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
    payment_date_col = best_matching_col(df, ["payment date", "community join date"])
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
    total_events = max(len(event_cols), 1)
    if engagement_pct_col:
        parsed_pct = df[engagement_pct_col].apply(parse_numeric_percent)
        if parsed_pct.dropna().empty:
            df["engagement_pct"] = (df["participation_count"] / total_events) * 100
        else:
            if parsed_pct.max(skipna=True) <= 1.05:
                parsed_pct = parsed_pct * 100
            fallback_pct = (df["participation_count"] / total_events) * 100
            zero_needs_fallback = parsed_pct.fillna(0).eq(0) & df["participation_count"].gt(0)
            parsed_pct = parsed_pct.where(~zero_needs_fallback, fallback_pct)
            df["engagement_pct"] = parsed_pct.fillna(fallback_pct).fillna(0)
    else:
        df["engagement_pct"] = (df["participation_count"] / total_events) * 100

    if payment_date_col:
        df[payment_date_col] = df[payment_date_col].apply(parse_date_safe)
        df["payment_date_parsed"] = df[payment_date_col]
    else:
        df["payment_date_parsed"] = pd.NaT

    if sheet_name in {"Tetr-X-UG", "Tetr-X-PG"} and "payment_date_parsed" in df.columns and df["payment_date_parsed"].isna().all():
        join_col = best_matching_col(df, ["community join date"])
        if join_col and join_col in df.columns:
            df["payment_date_parsed"] = df[join_col].apply(parse_date_safe)

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
            fallback_pay = pd.to_datetime(row.get("master_payment_date_parsed", pd.NaT), errors="coerce")
            resolved_payment.append(fallback_pay if pd.notna(fallback_pay) else pd.NaT)
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
    dates_df = pd.DataFrame()
    winner_df = pd.DataFrame()

    for sheet in MASTER_SHEETS:
        if sheet in sheet_names:
            raw = load_raw_sheet(source_mode, sheet, spreadsheet_id, file_bytes)
            masters[sheet], master_ctx[sheet] = parse_master_sheet(raw, "UG" if sheet.endswith("UG") else "PG", sheet)

    for sheet in UG_BATCH_SHEETS + PG_BATCH_SHEETS + TX_SHEETS:
        if sheet in sheet_names:
            raw = load_raw_sheet(source_mode, sheet, spreadsheet_id, file_bytes)
            activities[sheet], activity_ctx[sheet] = parse_activity_sheet(raw, sheet)

    if DATES_SHEET in sheet_names:
        raw = load_raw_sheet(source_mode, DATES_SHEET, spreadsheet_id, file_bytes)
        dates_df = parse_dates_sheet(raw)

    if WINNER_SHEET in sheet_names:
        raw = load_raw_sheet(source_mode, WINNER_SHEET, spreadsheet_id, file_bytes)
        winner_df = parse_winner_sheet(raw)

    tx_df = pd.concat([activities[s] for s in TX_SHEETS if s in activities], ignore_index=True) if any(s in activities for s in TX_SHEETS) else pd.DataFrame()
    if not tx_df.empty:
        tx_df = tx_df.copy()
        tx_df["tx_status"] = tx_df.get("sheet_status_raw", "")
        tx_df["tx_payment_date"] = tx_df.get("payment_date_parsed", pd.NaT)

    overview_frames = [masters[sheet].copy() for sheet in MASTER_SHEETS if sheet in masters]
    overview_df = pd.concat(overview_frames, ignore_index=True) if overview_frames else pd.DataFrame()

    if not overview_df.empty and not dates_df.empty:
        offered_vals, deadline_vals = [], []
        for _, row in overview_df.iterrows():
            drow = find_student_dates_row(dates_df, row.get("student_name", ""), row.get("email_key", ""), row.get("student_key", ""), row.get("Program", ""), row.get("Batch", ""))
            if drow is None:
                offered_vals.append(pd.NaT); deadline_vals.append(pd.NaT)
            else:
                offered_vals.append(pd.to_datetime(drow.get("offered_date_parsed", pd.NaT), errors="coerce"))
                deadline_vals.append(pd.to_datetime(drow.get("deadline_parsed", pd.NaT), errors="coerce"))
        overview_df["offered_date_parsed"] = offered_vals
        overview_df["deadline_parsed"] = deadline_vals

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
        "dates_df": dates_df,
        "winner_df": winner_df,
    }


# ---------------- Rendering ----------------



def compute_tx_prepayment_event_type_summary(tx_df, tx_program, data):
    columns = ["event_type", "Students Attended", "Attended %", "Event Occurrences", "Attendance Hits"]
    if tx_df is None or tx_df.empty:
        return pd.DataFrame(columns=columns)

    # Use only paid/admitted Tetr-X students for the pre-payment batch attendance summary.
    if "sheet_is_paid" in tx_df.columns:
        tx_df = tx_df[tx_df["sheet_is_paid"]].copy()
    if tx_df.empty:
        return pd.DataFrame(columns=columns)

    batch_sheets = UG_BATCH_SHEETS if tx_program == "UG" else PG_BATCH_SHEETS
    available_batch_sheets = [s for s in batch_sheets if s in data.get("activities", {})]
    if not available_batch_sheets:
        return pd.DataFrame(columns=columns)

    total_students = int(len(tx_df))
    # Build unique event occurrences by (event_type, event_date) across program sheets.
    unique_events = set()
    for sheet in available_batch_sheets:
        ctx = data.get("activity_ctx", {}).get(sheet, {})
        event_info = ctx.get("event_info", pd.DataFrame())
        if event_info is None or event_info.empty:
            continue
        for _, ev in event_info.iterrows():
            ev_type = clean_text(ev.get("event_type", "Other")) or "Other"
            ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
            ev_date_key = ev_date.normalize() if pd.notna(ev_date) else pd.NaT
            unique_events.add((ev_type, ev_date_key))

    occurrence_counts = {}
    for ev_type, ev_date in unique_events:
        occurrence_counts[ev_type] = occurrence_counts.get(ev_type, 0) + 1

    attendance_hits = {}
    unique_student_hits = {}

    for _, tx_row in tx_df.iterrows():
        email_key = clean_text(tx_row.get("email_key", ""))
        student_key = clean_text(tx_row.get("student_key", ""))
        pay_dt = pd.to_datetime(tx_row.get("payment_date_parsed", pd.NaT), errors="coerce")

        matched_batch_row = None
        matched_ctx = None
        for sheet in available_batch_sheets:
            batch_df = data["activities"].get(sheet, pd.DataFrame())
            if batch_df.empty:
                continue
            mask = pd.Series(False, index=batch_df.index)
            if email_key and "email_key" in batch_df.columns:
                mask = mask | (batch_df["email_key"] == email_key)
            if student_key and "student_key" in batch_df.columns:
                mask = mask | (batch_df["student_key"] == student_key)
            part = batch_df[mask]
            if not part.empty:
                matched_batch_row = part.iloc[0]
                matched_ctx = data["activity_ctx"].get(sheet, {})
                break

        if matched_batch_row is None or matched_ctx is None:
            continue

        event_info = matched_ctx.get("event_info", pd.DataFrame())
        if event_info is None or event_info.empty:
            continue

        student_id = student_key or email_key or clean_text(tx_row.get("student_name", ""))
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
            attendance_hits[ev_type] = attendance_hits.get(ev_type, 0) + 1
            unique_student_hits.setdefault(ev_type, set()).add(student_id)

    rows = []
    event_types = sorted(set(list(occurrence_counts.keys()) + list(attendance_hits.keys())))
    for ev_type in event_types:
        occ = int(occurrence_counts.get(ev_type, 0))
        hits = int(attendance_hits.get(ev_type, 0))
        students_attended = len([s for s in unique_student_hits.get(ev_type, set()) if clean_text(s)])
        denom = total_students * occ if total_students and occ else 0
        pct = round((hits / denom * 100), 2) if denom else 0.0
        rows.append({
            "event_type": ev_type,
            "Students Attended": students_attended,
            "Attended %": pct,
            "Event Occurrences": occ,
            "Attendance Hits": hits,
        })

    return pd.DataFrame(rows).sort_values(["Attended %", "Students Attended", "event_type"], ascending=[False, False, True]) if rows else pd.DataFrame(columns=columns)


def render_live_ist_clock(connected_ok: bool, connection_note: str):
    status_text = f"LIVE · {connection_note or 'Google Sheets'}" if connected_ok else f"OFFLINE · {connection_note or 'Google Sheets'}"
    status_bg = '#e8f6ed' if connected_ok else '#fdeceb'
    status_fg = GREEN if connected_ok else '#7a1f1b'
    status_border = '#cfe8d9' if connected_ok else '#f3cdca'
    status_dot = '#1bb55c' if connected_ok else RED
    html = f"""
    <div style="display:flex; justify-content:flex-end; margin-bottom:10px; font-family: Arial, sans-serif;">
      <div style="display:flex; flex-direction:column; align-items:flex-end; gap:8px;">
        <div id="ist-live-clock" style="padding:10px 14px; border-radius:999px; border:1px solid #dbeee0; background:#ffffff; color:#0b3d2e; font-weight:700; min-width:300px; text-align:center; box-shadow:0 2px 10px rgba(11, 61, 46, 0.05);">IST · --</div>
        <div style="display:inline-flex; align-items:center; gap:10px; padding:10px 14px; border-radius:999px; font-weight:800; border:1px solid {status_border}; color:{status_fg}; background:{status_bg}; white-space:nowrap;">
          <span style="width:12px; height:12px; border-radius:50%; background:{status_dot}; display:inline-block;"></span>
          {status_text}
        </div>
      </div>
    </div>
    <script>
    (function() {{
      function updateISTClock() {{
        var el = document.getElementById('ist-live-clock');
        if (!el) return;
        var parts = new Intl.DateTimeFormat('en-IN', {{
          timeZone: 'Asia/Kolkata',
          weekday: 'short',
          day: '2-digit',
          month: 'short',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
          hour12: true
        }}).format(new Date());
        el.textContent = 'IST · ' + parts;
      }}
      updateISTClock();
      setInterval(updateISTClock, 1000);
    }})();
    </script>
    """
    components.html(html, height=110)


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
            st.caption("Based on paid/admitted Tetr-X students only, using their attended batch-sheet events before payment date.")
            if not type_counts.empty:
                fig = px.bar(type_counts, x="event_type", y="Attended %", text="Students Attended", title="Pre-Payment Batch Attendance by Event Type", hover_data=["Students Attended", "Event Occurrences", "Attendance Hits"], color="event_type")
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



def collect_student_profile_events(data, email_key: str, student_key: str, student_name: str, pay_dt=pd.NaT, offered_dt=pd.NaT, deadline_dt=pd.NaT):
    rows = []
    for sheet, ctx in data.get("activity_ctx", {}).items():
        sdf = data.get("activities", {}).get(sheet, pd.DataFrame())
        if sdf.empty or ctx.get("event_info", pd.DataFrame()).empty:
            continue
        mask = pd.Series(False, index=sdf.index)
        if email_key and "email_key" in sdf.columns:
            mask = mask | sdf["email_key"].astype(str).eq(email_key)
        if student_key and "student_key" in sdf.columns:
            mask = mask | sdf["student_key"].astype(str).eq(student_key)
        part = sdf.loc[mask].copy()
        if part.empty:
            continue
        event_info = ctx.get("event_info", pd.DataFrame())
        for _, prow in part.iterrows():
            for _, ev in event_info.iterrows():
                col = ev.get("column_name")
                if not col or col not in prow.index:
                    continue
                attended = pd.to_numeric(pd.Series([prow.get(col, 0)]), errors="coerce").fillna(0).iloc[0]
                if attended <= 0:
                    continue
                ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
                ev_type = clean_text(ev.get("event_type", "Other")) or "Other"
                ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
                ev_date_norm = ev_date.normalize() if pd.notna(ev_date) else pd.NaT
                date_key = ev_date_norm.strftime("%Y-%m-%d") if pd.notna(ev_date_norm) else "undated"
                dedupe_key = "|".join([
                    student_key or email_key or normalize_name(student_name),
                    normalize_name(ev_name),
                    normalize_name(ev_type),
                    date_key,
                ])
                rows.append({
                    "sheet": sheet,
                    "source_group": "tetrx" if sheet in TX_SHEETS else "batch",
                    "event_name": ev_name,
                    "event_type": ev_type,
                    "event_date": ev_date_norm,
                    "count": int(attended),
                    "dedupe_key": dedupe_key,
                })
    ev_df = pd.DataFrame(rows)
    if ev_df.empty:
        return ev_df
    ev_df = (
        ev_df.sort_values(["event_date", "event_name", "sheet"], na_position="last")
        .groupby("dedupe_key", as_index=False)
        .agg({
            "event_name": "first",
            "event_type": "first",
            "event_date": "first",
            "count": "max",
            "sheet": lambda s: ", ".join(sorted(dict.fromkeys([clean_text(x) for x in s if clean_text(x)]))),
            "source_group": lambda s: ", ".join(sorted(dict.fromkeys([clean_text(x) for x in s if clean_text(x)]))),
        })
        .rename(columns={"sheet": "source_sheets"})
    )
    pay_dt = pd.to_datetime(pay_dt, errors="coerce")
    offered_dt = pd.to_datetime(offered_dt, errors="coerce")
    deadline_dt = pd.to_datetime(deadline_dt, errors="coerce")
    ev_df["in_first30"] = False
    ev_df["after_paid"] = False
    ev_df["in_t7"] = False
    ev_df["in_tplus7"] = False
    if pd.notna(offered_dt) and pd.notna(deadline_dt):
        ev_df["in_first30"] = ev_df["event_date"].between(offered_dt.normalize(), deadline_dt.normalize(), inclusive="both")
    if pd.notna(pay_dt):
        norm_pay = pay_dt.normalize()
        ev_df["after_paid"] = ev_df["event_date"].ge(norm_pay)
        delta = (ev_df["event_date"] - norm_pay).dt.days
        ev_df["in_t7"] = delta.between(-7, -1, inclusive="both")
        ev_df["in_tplus7"] = delta.between(1, 7, inclusive="both")
    return ev_df


def create_student_profiles_pdf(profile_payloads):
    if not profile_payloads:
        return None
    bio = BytesIO()
    with PdfPages(bio) as pdf:
        for payload in profile_payloads:
            fig = plt.figure(figsize=(8.27, 11.69))
            gs = fig.add_gridspec(5, 2, height_ratios=[0.9, 1.1, 1.6, 1.4, 1.3], hspace=0.7, wspace=0.45)
            ax_title = fig.add_subplot(gs[0, :]); ax_title.axis("off")
            ax_title.text(0, 0.92, payload["title"], fontsize=18, fontweight="bold", color="#0b3d2e", va="top")
            ax_title.text(0, 0.65, payload.get("subtitle", ""), fontsize=10, color="#12372a", va="top")
            metrics = payload.get("metrics", {})
            metric_lines = [f"{k}: {v}" for k, v in metrics.items()]
            ax_title.text(0, 0.18, "   |   ".join(metric_lines), fontsize=9, color="#1f7a56", va="top")

            ax_info = fig.add_subplot(gs[1, 0]); ax_info.axis("off")
            info_lines = [f"{k}: {v}" for k, v in payload.get("info", {}).items()]
            ax_info.text(0, 1, "\n".join(info_lines), fontsize=8.5, va="top")

            ax_t7 = fig.add_subplot(gs[1, 1]); ax_t7.axis("off")
            t7_lines = [f"{k}: {v}" for k, v in payload.get("t7_info", {}).items()]
            ax_t7.text(0, 1, "\n".join(t7_lines), fontsize=8.5, va="top")

            type_df = payload.get("type_df", pd.DataFrame())
            ax_type = fig.add_subplot(gs[2, 0])
            if not type_df.empty:
                d = type_df.head(8)
                ax_type.barh(d["event_type"], d["count"])
                ax_type.set_title("Event Type Participation", fontsize=10)
                ax_type.invert_yaxis()
            else:
                ax_type.text(0.5, 0.5, "No event type data", ha="center", va="center")
                ax_type.set_axis_off()

            timeline_df = payload.get("timeline_df", pd.DataFrame())
            ax_time = fig.add_subplot(gs[2, 1])
            if not timeline_df.empty:
                ax_time.plot(timeline_df["event_date"], timeline_df["count"], marker="o")
                if pd.notna(payload.get("payment_date", pd.NaT)):
                    ax_time.axvline(pd.to_datetime(payload["payment_date"]), linestyle="--", linewidth=1.5)
                ax_time.set_title("Engagement Timeline", fontsize=10)
                ax_time.tick_params(axis="x", labelrotation=30, labelsize=8)
            else:
                ax_time.text(0.5, 0.5, "No dated events", ha="center", va="center")
                ax_time.set_axis_off()

            ax_events = fig.add_subplot(gs[3:, :]); ax_events.axis("off")
            events_df = payload.get("events_df", pd.DataFrame())
            if not events_df.empty:
                show = events_df.copy()
                show["event_date"] = show["event_date"].apply(lambda x: format_date_display(x) if pd.notna(pd.to_datetime(x, errors="coerce")) else "")
                show = show[["event_date", "event_type", "event_name", "source_sheets"]].head(18)
                tbl = ax_events.table(cellText=show.values, colLabels=show.columns, loc="upper left", cellLoc="left")
                tbl.auto_set_font_size(False)
                tbl.set_fontsize(7.5)
                tbl.scale(1, 1.2)
                ax_events.set_title("Event Details", fontsize=10, loc="left")
            else:
                ax_events.text(0.5, 0.5, "No event details available", ha="center", va="center")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    bio.seek(0)
    return bio.getvalue()



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

    profile_window_df = build_t7_event_window_data(data)
    pdf_payloads = []

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


        winner_df = data.get("winner_df", pd.DataFrame())
        student_wins = pd.DataFrame()
        if winner_df is not None and not winner_df.empty:
            wmask = pd.Series(False, index=winner_df.index)
            if email_key and "email_key" in winner_df.columns:
                wmask = wmask | winner_df["email_key"].astype(str).eq(email_key)
            if name_key and "student_key" in winner_df.columns:
                wmask = wmask | winner_df["student_key"].astype(str).eq(name_key)
            if master.get("Batch", "") and "batch_key" in winner_df.columns:
                batch_key = normalize_batch_token(master.get("Batch", ""))
                narrowed = winner_df.loc[wmask & winner_df["batch_key"].astype(str).eq(batch_key)].copy()
                if not narrowed.empty:
                    student_wins = narrowed
                else:
                    student_wins = winner_df.loc[wmask].copy()
            else:
                student_wins = winner_df.loc[wmask].copy()

        batch_only_df = related_df[~related_df["source_sheet"].isin(TX_SHEETS)].copy() if not related_df.empty and "source_sheet" in related_df.columns else pd.DataFrame()
        batch_comm = ""
        if not batch_only_df.empty and "community_status_value" in batch_only_df.columns:
            comm_series = batch_only_df["community_status_value"].replace("", np.nan).dropna()
            if not comm_series.empty:
                batch_comm = comm_series.mode().iat[0] if not comm_series.mode().empty else comm_series.iloc[0]

        pay_dt = pd.to_datetime(master.get("resolved_payment_date", pd.NaT), errors="coerce")
        if pd.isna(pay_dt) and not related_df.empty and "payment_date_parsed" in related_df.columns:
            rel_pay = pd.to_datetime(related_df["payment_date_parsed"], errors="coerce").dropna()
            if not rel_pay.empty:
                pay_dt = rel_pay.min()

        dates_row = find_student_dates_row(
            data.get("dates_df", pd.DataFrame()),
            master.get("student_name", ""),
            email_key,
            name_key,
            master.get("Program", ""),
            master.get("Batch", ""),
        )
        offered_dt = pd.to_datetime(dates_row.get("offered_date_parsed", pd.NaT), errors="coerce") if dates_row is not None else pd.NaT
        deadline_dt = pd.to_datetime(dates_row.get("deadline_parsed", pd.NaT), errors="coerce") if dates_row is not None else pd.NaT

        profile_event_df = collect_student_profile_events(data, email_key, name_key, master.get("student_name", ""), pay_dt=pay_dt, offered_dt=offered_dt, deadline_dt=deadline_dt)

        wmask = pd.Series(False, index=profile_window_df.index)
        if not profile_window_df.empty:
            if email_key and "email_key" in profile_window_df.columns:
                wmask = wmask | profile_window_df["email_key"].astype(str).eq(email_key)
            if name_key and "student_key" in profile_window_df.columns:
                wmask = wmask | profile_window_df["student_key"].astype(str).eq(name_key)
            if not wmask.any():
                wmask = profile_window_df["student_name"].astype(str).str.lower().eq(clean_text(master.get("student_name", "")).lower())
        stu_window = profile_window_df.loc[wmask].copy() if not profile_window_df.empty else pd.DataFrame()

        st.markdown("---")
        st.markdown(f"### {master['student_name']}")

        p1, p2, p3, p4, p5, p6, p7 = st.columns(7)
        p1.metric("Program", clean_text(master.get("Program", "")))
        p2.metric("Batch", clean_text(master.get("Batch", "")))
        p3.metric("Paid Status", clean_text(master.get("resolved_status", "Not Paid")))
        p4.metric("Payment Date", format_date_display(pay_dt))
        p5.metric("Offered Date", format_date_display(offered_dt))
        p6.metric("Deadline", format_date_display(deadline_dt))
        p7.metric("Community Status", batch_comm if batch_comm else "—")

        first30_count = int(profile_event_df.loc[profile_event_df["in_first30"], "dedupe_key"].nunique()) if not profile_event_df.empty else 0
        after_paid_count = int(profile_event_df.loc[profile_event_df["after_paid"], "dedupe_key"].nunique()) if (not profile_event_df.empty and pd.notna(pay_dt)) else 0
        t7_count = int(stu_window.loc[stu_window["window"] == "T-7 to T", "dedupe_key"].nunique()) if not stu_window.empty else 0
        tp7_count = int(stu_window.loc[stu_window["window"] == "T+1 to T+7", "dedupe_key"].nunique()) if not stu_window.empty else 0

        total_events = int(profile_event_df["dedupe_key"].nunique()) if not profile_event_df.empty else 0

        winner_rows = pd.DataFrame()
        spotlight_rows = pd.DataFrame()
        winner_count = 0
        total_money_won = 0.0
        winner_challenges = ""
        spotlight_count = 0
        spotlight_challenges = ""
        if student_wins is not None and not student_wins.empty:
            winner_rows = student_wins[student_wins["is_winner"]].copy() if "is_winner" in student_wins.columns else pd.DataFrame()
            spotlight_rows = student_wins[student_wins["is_spotlight"]].copy() if "is_spotlight" in student_wins.columns else pd.DataFrame()
            winner_count = int(len(winner_rows))
            total_money_won = float(winner_rows.get("amount_usd", pd.Series(dtype=float)).fillna(0).sum()) if not winner_rows.empty else 0.0
            winner_challenges = ", ".join(sorted(dict.fromkeys([clean_text(x) for x in winner_rows.get("challenge_name", pd.Series(dtype=str)).tolist() if clean_text(x)])))
            spotlight_count = int(len(spotlight_rows))
            spotlight_challenges = ", ".join(sorted(dict.fromkeys([clean_text(x) for x in spotlight_rows.get("challenge_name", pd.Series(dtype=str)).tolist() if clean_text(x)])))

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
                "Payment Date": format_date_display(pay_dt),
                "Offered Date": format_date_display(offered_dt),
                "Deadline": format_date_display(deadline_dt),
                "Community Status (Batch)": batch_comm,
            }
            st.dataframe(pd.DataFrame(master_display.items(), columns=["Field", "Value"]), use_container_width=True, hide_index=True, key=f"profile_info_{i}")
        admitted_flag = clean_text(master.get("resolved_status", "")).lower() == "admitted"
        with c2:
            stat_row_1 = st.columns(3)
            stat_row_1[0].metric("Total Participation", total_events)
            stat_row_1[1].metric("First 30 Days", first30_count)
            stat_row_1[2].metric("After Payment", after_paid_count if admitted_flag else 0)

            stat_row_2 = st.columns(2)
            stat_row_2[0].metric("T-7 Count", t7_count if admitted_flag else 0)
            stat_row_2[1].metric("T+7 Count", tp7_count if admitted_flag else 0)

            stat_row_3 = st.columns(3)
            stat_row_3[0].metric("Winner", winner_count)
            stat_row_3[1].metric("Shoutout", spotlight_count)
            stat_row_3[2].metric("Money Won (USD)", f"{total_money_won:,.0f}" if abs(total_money_won - round(total_money_won)) < 1e-9 else f"{total_money_won:,.2f}")

            win_info = []
            if winner_challenges:
                win_info.append({"Field": "Winner Challenges", "Value": winner_challenges})
            if spotlight_challenges:
                win_info.append({"Field": "Spotlight Challenges", "Value": spotlight_challenges})
            if win_info:
                st.dataframe(pd.DataFrame(win_info), use_container_width=True, hide_index=True, key=f"profile_winner_info_{i}")

        if not related_df.empty:
            if not profile_event_df.empty:
                type_df = profile_event_df.groupby("event_type")["dedupe_key"].nunique().reset_index(name="count").sort_values("count", ascending=False)
                total = type_df["count"].sum()
                type_df["percentage"] = np.where(total > 0, type_df["count"] / total * 100, 0)

                plot_type_df = type_df.copy()
                plot_type_df["plot_event_type"] = plot_type_df["event_type"].apply(map_profile_plot_event_type)
                plot_type_df = plot_type_df.groupby("plot_event_type", as_index=False)["count"].sum().sort_values("count", ascending=False)

                x1, x2 = st.columns(2)
                with x1:
                    fig = px.bar(plot_type_df, x="plot_event_type", y="count", title="Event Type Participation")
                    fig.update_traces(marker_color=GREEN_2)
                    st.plotly_chart(nice_layout(fig, height=340, x_tickangle=-25), use_container_width=True, key=f"profile_type_bar_{i}")
                with x2:
                    fig = px.pie(plot_type_df, names="plot_event_type", values="count", hole=0.58, title="Event Type % Share")
                    st.plotly_chart(nice_layout(fig, height=340), use_container_width=True, key=f"profile_type_pie_{i}")

                timeline = profile_event_df.dropna(subset=["event_date"]).sort_values("event_date")
                if not timeline.empty:
                    timeline = timeline.groupby(["event_date", "event_name"], as_index=False)["dedupe_key"].nunique().rename(columns={"dedupe_key": "count"})
                    fig = px.line(timeline, x="event_date", y="count", markers=True, title="Engagement Timeline")
                    fig.update_traces(line_color=GREEN, marker_color=GREEN)
                    if pd.notna(pay_dt):
                        x = pd.Timestamp(pay_dt)
                        fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, xref="x", yref="paper", line=dict(color=RED, width=2, dash="dash"))
                        fig.add_annotation(x=x, y=1, yref="paper", text="Payment Date", showarrow=False, font=dict(color=RED), bgcolor="white")
                    st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"profile_timeline_{i}")

                if clean_text(master.get("resolved_status", "")).lower() == "admitted" and not stu_window.empty:
                    st.markdown("#### T-7 & T+7 Attendance")
                    detail_rows = []
                    for label in ["T-7 to T", "T+1 to T+7"]:
                        partw = stu_window[stu_window["window"] == label].copy()
                        detail_rows.append({
                            "Window": label,
                            "Activities": int(partw["dedupe_key"].nunique()) if not partw.empty else 0,
                            "Event Types": ", ".join(sorted(dict.fromkeys([clean_text(x) for x in partw["event_type"].tolist() if clean_text(x)]))),
                            "Events": ", ".join(sorted(dict.fromkeys([clean_text(x) for x in partw["event_name"].tolist() if clean_text(x)]))),
                        })
                    st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True, key=f"profile_t7_table_{i}")

                st.markdown("#### Event Details")
                show = profile_event_df.sort_values(["event_date", "event_type", "event_name"], ascending=[True, True, True]).copy()
                st.dataframe(show, use_container_width=True, height=320, key=f"profile_events_{i}")
            else:
                type_df = pd.DataFrame()
                timeline = pd.DataFrame()
                st.info("Matched records found, but no attended events were recorded for this student.")

            st.markdown("#### Matched Batch / Tetr-X Records")
            record_cols = [c for c in ["source_sheet", "Batch", "engagement_pct", "engagement_score", "sheet_status_raw", "payment_date_parsed"] if c in related_df.columns]
            st.dataframe(related_df[record_cols].sort_values(["source_sheet", "Batch"]), use_container_width=True, height=250, key=f"profile_records_{i}")
        else:
            type_df = pd.DataFrame()
            timeline = pd.DataFrame()

        pdf_payloads.append({
            "title": clean_text(master.get("student_name", "")),
            "subtitle": f"{clean_text(master.get('Program', ''))} | {clean_text(master.get('Batch', ''))} | Status: {clean_text(master.get('resolved_status', ''))}",
            "metrics": {
                "Total So Far": total_events,
                "First 30 Days": first30_count,
                "After Paid": after_paid_count if clean_text(master.get("resolved_status", "")).lower() == "admitted" else 0,
                "T-7": t7_count if clean_text(master.get("resolved_status", "")).lower() == "admitted" else 0,
                "T+7": tp7_count if clean_text(master.get("resolved_status", "")).lower() == "admitted" else 0,
            },
            "info": master_display,
            "t7_info": {
                "Payment Date": format_date_display(pay_dt),
                "Offered Date": format_date_display(offered_dt),
                "Deadline": format_date_display(deadline_dt),
                "Community Status": batch_comm if batch_comm else "—",
            },
            "type_df": type_df if 'type_df' in locals() else pd.DataFrame(),
            "timeline_df": timeline if 'timeline' in locals() else pd.DataFrame(),
            "events_df": profile_event_df if not profile_event_df.empty else pd.DataFrame(),
            "payment_date": pay_dt,
        })

    if pdf_payloads:
        pdf_bytes = create_student_profiles_pdf(pdf_payloads)
        if pdf_bytes:
            label = "Download Student Profile PDF" if len(pdf_payloads) == 1 else "Download Selected Student Profiles PDF"
            st.download_button(label, data=pdf_bytes, file_name="student_profiles.pdf", mime="application/pdf", key="student_profile_pdf_download")





def build_combined_activity_context(sheets, data):
    available = [s for s in sheets if s in data.get("activities", {}) and not data["activities"][s].empty]
    if not available:
        return pd.DataFrame(), {"event_info": pd.DataFrame(columns=["column_name", "event_name", "event_type", "event_date", "sheet"]), "country_col": None}

    frames = [data["activities"][s] for s in available]
    combined_df = pd.concat(frames, ignore_index=True, sort=False)

    event_infos = []
    country_col = None
    for s in available:
        ctx = data.get("activity_ctx", {}).get(s, {})
        ei = ctx.get("event_info", pd.DataFrame())
        if ei is not None and not ei.empty:
            event_infos.append(ei.copy())
        if country_col is None and ctx.get("country_col"):
            country_col = ctx.get("country_col")

    if event_infos:
        combined_event_info = pd.concat(event_infos, ignore_index=True, sort=False)
        if "column_name" in combined_event_info.columns:
            combined_event_info = combined_event_info.drop_duplicates(subset=["column_name"]).reset_index(drop=True)
    else:
        combined_event_info = pd.DataFrame(columns=["column_name", "event_name", "event_type", "event_date", "sheet"])

    combined_ctx = {
        "event_info": combined_event_info,
        "country_col": country_col,
    }
    return combined_df, combined_ctx


def render_combined_program_section(title, sheets, data, prefix):
    combined_df, combined_ctx = build_combined_activity_context(sheets, data)
    if combined_df.empty:
        st.warning(f"No data available for {title}.")
        return
    render_sheet_detail(title, combined_df, combined_ctx, prefix, data=data)
def render_program_page(title, sheets, data, page_prefix):
    st.subheader(title)
    available = [s for s in sheets if s in data["activities"]]
    if not available:
        st.warning("No sheets available for this section.")
        return
    combined_label = f"All {title}"
    tab_labels = [combined_label] + available
    tabs = st.tabs(tab_labels)
    with tabs[0]:
        render_combined_program_section(combined_label, sheets, data, f"{page_prefix}_combined")
    for tab, sheet in zip(tabs[1:], available):
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




def build_t7_event_window_data(data):
    columns = ["student_name", "email_key", "student_key", "program", "payment_date", "window", "event_name", "event_type", "event_date", "source_sheet", "dedupe_key"]
    overview_df = data.get("overview_df", pd.DataFrame())
    if overview_df.empty:
        return pd.DataFrame(columns=columns)

    admitted = overview_df[overview_df["is_paid"]].copy() if "is_paid" in overview_df.columns else pd.DataFrame()
    if admitted.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    activity_ctx = data.get("activity_ctx", {})
    activities = data.get("activities", {})
    for _, stu in admitted.iterrows():
        email_key = clean_text(stu.get("email_key", ""))
        student_key = clean_text(stu.get("student_key", ""))
        student_name = clean_text(stu.get("student_name", ""))
        program = clean_text(stu.get("Program", ""))
        pay_dt = pd.to_datetime(stu.get("resolved_payment_date", stu.get("master_payment_date_parsed", pd.NaT)), errors="coerce")
        if pd.isna(pay_dt):
            pay_dt = pd.to_datetime(stu.get("master_payment_date_parsed", pd.NaT), errors="coerce")

        candidate_sheets = list(UG_BATCH_SHEETS if program == "UG" else PG_BATCH_SHEETS)
        tx_sheet = "Tetr-X-UG" if program == "UG" else "Tetr-X-PG"
        if tx_sheet in activities:
            candidate_sheets.append(tx_sheet)

        if pd.isna(pay_dt):
            fallback_pays = []
            for sheet in candidate_sheets:
                if sheet not in activities:
                    continue
                sdf = activities[sheet]
                if sdf.empty or "payment_date_parsed" not in sdf.columns:
                    continue
                mask = pd.Series(False, index=sdf.index)
                if email_key and "email_key" in sdf.columns:
                    mask = mask | sdf["email_key"].astype(str).eq(email_key)
                if student_key and "student_key" in sdf.columns:
                    mask = mask | sdf["student_key"].astype(str).eq(student_key)
                part_pay = pd.to_datetime(sdf.loc[mask, "payment_date_parsed"], errors="coerce").dropna()
                if not part_pay.empty:
                    fallback_pays.extend(part_pay.tolist())
            if fallback_pays:
                pay_dt = min(fallback_pays)
        if pd.isna(pay_dt):
            continue
        pay_dt = pd.to_datetime(pay_dt, errors="coerce").normalize()

        for sheet in candidate_sheets:
            if sheet not in activities or sheet not in activity_ctx:
                continue
            sdf = activities[sheet]
            if sdf.empty:
                continue
            mask = pd.Series(False, index=sdf.index)
            if email_key and "email_key" in sdf.columns:
                mask = mask | sdf["email_key"].astype(str).eq(email_key)
            if student_key and "student_key" in sdf.columns:
                mask = mask | sdf["student_key"].astype(str).eq(student_key)
            part = sdf[mask].copy()
            if part.empty:
                continue
            event_info = activity_ctx[sheet].get("event_info", pd.DataFrame())
            if event_info is None or event_info.empty:
                continue
            for _, prow in part.iterrows():
                for _, ev in event_info.iterrows():
                    col = ev.get("column_name")
                    if not col or col not in prow.index:
                        continue
                    attended = pd.to_numeric(pd.Series([prow.get(col, 0)]), errors="coerce").fillna(0).iloc[0]
                    if attended <= 0:
                        continue
                    ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
                    if pd.isna(ev_date):
                        continue
                    ev_date = ev_date.normalize()
                    delta = (ev_date - pay_dt).days
                    if -7 <= delta <= 0:
                        window = "T-7 to T"
                    elif 1 <= delta <= 7:
                        window = "T+1 to T+7"
                    else:
                        continue
                    ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
                    ev_type = clean_text(ev.get("event_type", "Other")) or "Other"
                    dedupe_key = "|".join([
                        student_key or email_key or normalize_name(student_name),
                        normalize_name(ev_name),
                        normalize_name(ev_type),
                        ev_date.strftime('%Y-%m-%d')
                    ])
                    rows.append({
                        "student_name": student_name,
                        "email_key": email_key,
                        "student_key": student_key,
                        "program": program,
                        "payment_date": pay_dt,
                        "window": window,
                        "event_name": ev_name,
                        "event_type": ev_type,
                        "event_date": ev_date,
                        "source_sheet": sheet,
                        "dedupe_key": dedupe_key,
                    })
    if not rows:
        return pd.DataFrame(columns=columns)
    out = pd.DataFrame(rows)
    out = out.sort_values(["student_name", "event_date", "event_name", "source_sheet"]).drop_duplicates(subset=["dedupe_key"]).reset_index(drop=True)
    return out



def build_t7_student_summary_table(data):
    overview_df = data.get("overview_df", pd.DataFrame())
    if overview_df.empty or "is_paid" not in overview_df.columns:
        return pd.DataFrame()

    admitted = overview_df[overview_df["is_paid"]].copy()
    if admitted.empty:
        return pd.DataFrame()

    activities = data.get("activities", {})
    activity_ctx = data.get("activity_ctx", {})
    rows = []

    for _, stu in admitted.iterrows():
        student_name = clean_text(stu.get("student_name", ""))
        program = clean_text(stu.get("Program", ""))
        batch = clean_text(stu.get("Batch", ""))
        email_key = clean_text(stu.get("email_key", ""))
        student_key = clean_text(stu.get("student_key", ""))
        pay_dt = pd.to_datetime(stu.get("resolved_payment_date", stu.get("master_payment_date_parsed", pd.NaT)), errors="coerce")
        if pd.isna(pay_dt):
            pay_dt = pd.to_datetime(stu.get("master_payment_date_parsed", pd.NaT), errors="coerce")

        batch_sheets = list(UG_BATCH_SHEETS if program == "UG" else PG_BATCH_SHEETS)
        tx_sheet = "Tetr-X-UG" if program == "UG" else "Tetr-X-PG"
        candidate_sheets = batch_sheets + ([tx_sheet] if tx_sheet in activities else [])
        dates_row = find_student_dates_row(data.get("dates_df", pd.DataFrame()), student_name, email_key, student_key, program, batch)
        offered_dt = pd.to_datetime(dates_row.get("offered_date_parsed", pd.NaT), errors="coerce") if dates_row is not None else pd.NaT
        deadline_dt = pd.to_datetime(dates_row.get("deadline_parsed", pd.NaT), errors="coerce") if dates_row is not None else pd.NaT

        if pd.isna(pay_dt):
            fallback_pays = []
            for sheet in candidate_sheets:
                if sheet not in activities:
                    continue
                sdf = activities[sheet]
                if sdf.empty or "payment_date_parsed" not in sdf.columns:
                    continue
                mask = pd.Series(False, index=sdf.index)
                if email_key and "email_key" in sdf.columns:
                    mask = mask | sdf["email_key"].astype(str).eq(email_key)
                if student_key and "student_key" in sdf.columns:
                    mask = mask | sdf["student_key"].astype(str).eq(student_key)
                vals = pd.to_datetime(sdf.loc[mask, "payment_date_parsed"], errors="coerce").dropna()
                if not vals.empty:
                    fallback_pays.extend(vals.tolist())
            if fallback_pays:
                pay_dt = min(fallback_pays)
        if pd.isna(pay_dt):
            continue
        pay_dt = pd.to_datetime(pay_dt, errors="coerce").normalize()

        # community status from batch sheets only
        comm_vals = []
        for sheet in batch_sheets:
            if sheet not in activities:
                continue
            sdf = activities[sheet]
            if sdf.empty:
                continue
            mask = pd.Series(False, index=sdf.index)
            if email_key and "email_key" in sdf.columns:
                mask = mask | sdf["email_key"].astype(str).eq(email_key)
            if student_key and "student_key" in sdf.columns:
                mask = mask | sdf["student_key"].astype(str).eq(student_key)
            part = sdf.loc[mask]
            if not part.empty and "community_status_value" in part.columns:
                comm_vals.extend([clean_text(x) for x in part["community_status_value"].tolist() if clean_text(x)])
        community_yes_no = "Yes" if any(v in {"Tetr X", "In"} for v in comm_vals) else "No"

        event_rows = []
        for sheet in candidate_sheets:
            if sheet not in activities or sheet not in activity_ctx:
                continue
            sdf = activities[sheet]
            if sdf.empty:
                continue
            mask = pd.Series(False, index=sdf.index)
            if email_key and "email_key" in sdf.columns:
                mask = mask | sdf["email_key"].astype(str).eq(email_key)
            if student_key and "student_key" in sdf.columns:
                mask = mask | sdf["student_key"].astype(str).eq(student_key)
            part = sdf.loc[mask].copy()
            if part.empty:
                continue
            event_info = activity_ctx[sheet].get("event_info", pd.DataFrame())
            if event_info is None or event_info.empty:
                continue
            for _, prow in part.iterrows():
                for _, ev in event_info.iterrows():
                    col = ev.get("column_name")
                    if not col or col not in prow.index:
                        continue
                    attended = pd.to_numeric(pd.Series([prow.get(col, 0)]), errors="coerce").fillna(0).iloc[0]
                    if attended <= 0:
                        continue
                    ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
                    if pd.isna(ev_date):
                        continue
                    ev_date = ev_date.normalize()
                    delta = (ev_date - pay_dt).days
                    source_group = "tetrx" if sheet == tx_sheet else "batch"
                    # offered->deadline logic: before payment from batch only, on/after payment from Tetr-X only
                    in_total30 = False
                    if pd.notna(offered_dt) and pd.notna(deadline_dt) and offered_dt.normalize() <= ev_date <= deadline_dt.normalize():
                        in_total30 = ((source_group == "batch" and ev_date < pay_dt) or
                                      (source_group == "tetrx" and ev_date >= pay_dt))
                    # T+7 logic: both batch and tetrx after payment, merged later
                    in_tplus7 = 0 <= delta <= 7
                    in_tminus7 = source_group == "batch" and -7 <= delta <= 0
                    if not (in_total30 or in_tplus7 or in_tminus7):
                        continue
                    ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
                    ev_type = clean_text(ev.get("event_type", "Other")) or "Other"
                    dedupe_key = "|".join([
                        student_key or email_key or normalize_name(student_name),
                        normalize_name(ev_name),
                        normalize_name(ev_type),
                        ev_date.strftime('%Y-%m-%d')
                    ])
                    event_rows.append({
                        "student_name": student_name,
                        "program": program,
                        "batch": batch,
                        "payment_date": pay_dt,
                        "event_name": ev_name,
                        "event_type": ev_type,
                        "event_date": ev_date,
                        "source_sheet": sheet,
                        "source_group": source_group,
                        "delta": delta,
                        "dedupe_key": dedupe_key,
                        "in_total30": in_total30,
                        "in_tminus7": in_tminus7,
                        "in_tplus7": in_tplus7,
                    })

        ev_df = pd.DataFrame(event_rows)
        if not ev_df.empty:
            ev_df = ev_df.sort_values(["event_date", "event_name", "source_sheet"]).drop_duplicates(subset=["dedupe_key"]).reset_index(drop=True)

        # Helper to get per-type counts
        def type_count_map(frame):
            if frame.empty:
                return {}
            return frame.groupby("event_type")["dedupe_key"].nunique().to_dict()

        total30_df = ev_df[ev_df["in_total30"]].copy() if not ev_df.empty else pd.DataFrame()
        tminus7_df = ev_df[ev_df["in_tminus7"]].copy() if not ev_df.empty else pd.DataFrame()
        tplus7_df = ev_df[ev_df["in_tplus7"]].copy() if not ev_df.empty else pd.DataFrame()

        row = {
            "Student Name": student_name,
            "UG PG": program,
            "Batch": batch,
            "Date of payment": pay_dt,
            "Community status (yes/no)": community_yes_no,
            "Total activities (30D)": int(total30_df["dedupe_key"].nunique()) if not total30_df.empty else 0,
            "Number of activities T-7": int(tminus7_df["dedupe_key"].nunique()) if not tminus7_df.empty else 0,
            "Number of activities T+7": int(tplus7_df["dedupe_key"].nunique()) if not tplus7_df.empty else 0,
        }

        for event_type, count in type_count_map(total30_df).items():
            row[f"30D | {event_type}"] = int(count)
        for event_type, count in type_count_map(tminus7_df).items():
            row[f"T-7 | {event_type}"] = int(count)
        for event_type, count in type_count_map(tplus7_df).items():
            row[f"T+7 | {event_type}"] = int(count)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    static_cols = [
        "Student Name", "UG PG", "Batch", "Date of payment", "Total activities (30D)",
        "Community status (yes/no)", "Number of activities T-7", "Number of activities T+7"
    ]
    thirty_cols = sorted([c for c in out.columns if c.startswith("30D | ")])
    minus_cols = sorted([c for c in out.columns if c.startswith("T-7 | ")])
    plus_cols = sorted([c for c in out.columns if c.startswith("T+7 | ")])
    ordered = static_cols[:5] + thirty_cols + [static_cols[5], static_cols[6]] + minus_cols + [static_cols[7]] + plus_cols
    for c in ordered:
        if c not in out.columns:
            out[c] = 0 if "|" in c or "activities" in c else ""
    out["Date of payment"] = pd.to_datetime(out["Date of payment"], errors="coerce")
    return out[ordered].sort_values(["UG PG", "Batch", "Student Name"]).reset_index(drop=True)

def render_t7_analysis_page(data):
    st.subheader("T-7 & T+7 Analysis")
    window_df = build_t7_event_window_data(data)
    overview_df = data.get("overview_df", pd.DataFrame())
    admitted_df = overview_df[overview_df["is_paid"]].copy() if (not overview_df.empty and "is_paid" in overview_df.columns) else pd.DataFrame()
    admitted_total = int(len(admitted_df))

    k1, k2, k3 = st.columns(3)
    k1.metric("Admitted Students", f"{admitted_total:,}")
    k2.metric("Students with T-7 Activity", f"{int(window_df.loc[window_df['window'] == 'T-7 to T', 'student_name'].nunique()) if not window_df.empty else 0:,}")
    k3.metric("Students with T+7 Activity", f"{int(window_df.loc[window_df['window'] == 'T+1 to T+7', 'student_name'].nunique()) if not window_df.empty else 0:,}")

    if window_df.empty:
        st.info("No admitted students with dated attended events inside the 7-day payment windows were found.")
        return

    type_summary = (
        window_df.groupby(["window", "event_type"], as_index=False)
        .agg(
            student_count=("student_name", "nunique"),
            event_count=("dedupe_key", "nunique"),
            event_names=("event_name", lambda s: ", ".join(sorted(dict.fromkeys([clean_text(x) for x in s if clean_text(x)]))))
        )
        .sort_values(["window", "student_count", "event_count", "event_type"], ascending=[True, False, False, True])
    )
    type_summary["student_pct"] = np.where(admitted_total > 0, type_summary["student_count"] / admitted_total * 100, 0.0)

    st.markdown("#### Event Type Participation Summary")
    st.caption("Shows how many admitted students participated in each event type within 7 days before or after payment.")
    st.dataframe(
        type_summary.rename(columns={"event_type": "Event Type", "student_count": "Students", "event_count": "Attended Events", "student_pct": "% of Admitted", "event_names": "Events"}),
        use_container_width=True,
        height=min(420, 120 + 36 * len(type_summary)),
        key="t7_type_summary_table"
    )

    online_events = window_df[window_df["event_type"].astype(str).str.strip().str.lower().eq("online event")].copy()
    st.markdown("#### Online Event Breakdown")
    if online_events.empty:
        st.info("No Online Event activity found inside the T-7 / T+7 windows.")
    else:
        online_summary = (
            online_events.groupby(["window", "event_name"], as_index=False)
            .agg(student_count=("student_name", "nunique"), event_date=("event_date", "first"))
            .sort_values(["window", "student_count", "event_date", "event_name"], ascending=[True, False, True, True])
        )
        online_summary["student_pct"] = np.where(admitted_total > 0, online_summary["student_count"] / admitted_total * 100, 0.0)
        st.dataframe(
            online_summary.rename(columns={"event_name": "Online Event", "student_count": "Students", "student_pct": "% of Admitted", "event_date": "Date"}),
            use_container_width=True,
            height=min(320, 120 + 34 * len(online_summary)),
            key="t7_online_summary_table"
        )

    st.markdown("#### Student-wise T-7 and T+7 Attendance")
    student_window = (
        window_df.groupby(["student_name", "program", "payment_date", "window"], as_index=False)
        .agg(
            event_count=("dedupe_key", "nunique"),
            event_names=("event_name", lambda s: ", ".join(sorted(dict.fromkeys([clean_text(x) for x in s if clean_text(x)]))))
        )
    )
    before = student_window[student_window["window"] == "T-7 to T"][["student_name", "program", "payment_date", "event_count", "event_names"]].rename(columns={"event_count": "T-7 Count", "event_names": "T-7 Events"})
    after = student_window[student_window["window"] == "T+1 to T+7"][["student_name", "program", "payment_date", "event_count", "event_names"]].rename(columns={"event_count": "T+7 Count", "event_names": "T+7 Events"})
    student_view = admitted_df[["student_name", "Program", "resolved_payment_date"]].drop_duplicates().rename(columns={"Program": "program", "resolved_payment_date": "payment_date"})
    student_view = student_view.merge(before, on=["student_name", "program", "payment_date"], how="left")
    student_view = student_view.merge(after, on=["student_name", "program", "payment_date"], how="left")
    for col in ["T-7 Count", "T+7 Count"]:
        student_view[col] = pd.to_numeric(student_view[col], errors="coerce").fillna(0).astype(int)
    for col in ["T-7 Events", "T+7 Events"]:
        student_view[col] = student_view[col].fillna("")
    student_view["payment_date"] = pd.to_datetime(student_view["payment_date"], errors="coerce")
    student_view = student_view.sort_values(["program", "student_name"]).reset_index(drop=True)

    compact = student_view.copy()
    compact["T-7"] = compact.apply(lambda r: f"{r['T-7 Count']} · {r['T-7 Events']}" if r['T-7 Events'] else str(r['T-7 Count']), axis=1)
    compact["T+7"] = compact.apply(lambda r: f"{r['T+7 Count']} · {r['T+7 Events']}" if r['T+7 Events'] else str(r['T+7 Count']), axis=1)
    st.dataframe(compact[["student_name", "program", "payment_date", "T-7", "T+7"]].rename(columns={"student_name": "Student", "program": "Program", "payment_date": "Payment Date"}),
                 use_container_width=True, height=380, key="t7_student_table")

    chart_df = student_view.melt(id_vars=["student_name", "program"], value_vars=["T-7 Count", "T+7 Count"], var_name="Window", value_name="Events")
    chart_df["Window"] = chart_df["Window"].replace({"T-7 Count": "T-7", "T+7 Count": "T+7"})
    fig = px.bar(chart_df, x="student_name", y="Events", color="Window", barmode="group", title="Each Student's Attendance Around Payment Date",
                 hover_data={"program": True}, color_discrete_map={"T-7": GREEN, "T+7": GREEN_3})
    st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-45), use_container_width=True, key="t7_student_bar")

    st.markdown("#### Paid / Tetr X Student Activity Table")
    student_summary = build_t7_student_summary_table(data)
    if student_summary.empty:
        st.info("No student-level T-7 / T+7 summary rows could be built.")
    else:
        st.dataframe(student_summary, use_container_width=True, height=420, key="t7_student_summary_table")

    st.markdown("#### Detailed Window Event Log")
    show = window_df.sort_values(["student_name", "window", "event_date", "event_type", "event_name"]).copy()
    st.dataframe(show[["student_name", "program", "payment_date", "window", "event_date", "event_type", "event_name", "source_sheet"]],
                 use_container_width=True, height=320, key="t7_detail_table")



def build_retention_data(data):
    activities = data.get("activities", {})
    activity_ctx = data.get("activity_ctx", {})
    if not activities:
        return pd.DataFrame(), pd.DataFrame()

    tx_frames = []
    for tx_sheet in TX_SHEETS:
        tx_df = activities.get(tx_sheet, pd.DataFrame())
        if tx_df is None or tx_df.empty:
            continue
        frame = tx_df.copy()
        if "sheet_is_paid" in frame.columns:
            frame = frame[frame["sheet_is_paid"]].copy()
        else:
            status_series = frame.get("status_value", pd.Series("", index=frame.index)).astype(str).str.strip().str.lower()
            frame = frame[status_series.eq("admitted")].copy()
        if frame.empty:
            continue
        frame["tx_sheet"] = tx_sheet
        frame["program"] = "UG" if tx_sheet.endswith("UG") else "PG"
        frame["payment_date_tx"] = pd.to_datetime(frame.get("payment_date_parsed", pd.NaT), errors="coerce")
        frame = frame[frame["payment_date_tx"].notna()].copy()
        if frame.empty:
            continue
        tx_frames.append(frame)

    if not tx_frames:
        return pd.DataFrame(), pd.DataFrame()

    tx_students = pd.concat(tx_frames, ignore_index=True)
    dedupe_ids = tx_students.apply(lambda r: clean_text(r.get("email_key", "")) or clean_text(r.get("student_key", "")) or normalize_name(r.get("student_name", "")), axis=1)
    tx_students["_ret_student_id"] = dedupe_ids
    tx_students = tx_students.sort_values(["payment_date_tx", "student_name"]).drop_duplicates(subset=["_ret_student_id"], keep="first").reset_index(drop=True)

    summary_rows = []
    event_rows = []
    dates_df = data.get("dates_df", pd.DataFrame())

    for _, stu in tx_students.iterrows():
        student_name = clean_text(stu.get("student_name", ""))
        program = clean_text(stu.get("program", "")) or clean_text(stu.get("Program", ""))
        batch = clean_text(stu.get("Batch", ""))
        email_key = clean_text(stu.get("email_key", ""))
        student_key = clean_text(stu.get("student_key", ""))
        pay_dt = pd.to_datetime(stu.get("payment_date_tx", pd.NaT), errors="coerce")
        if pd.isna(pay_dt):
            continue
        pay_dt = pay_dt.normalize()

        dates_row = find_student_dates_row(dates_df, student_name, email_key, student_key, program, batch)
        offered_dt = pd.to_datetime(dates_row.get("offered_date_parsed", pd.NaT), errors="coerce") if dates_row is not None else pd.NaT
        deadline_dt = pd.to_datetime(dates_row.get("deadline_parsed", pd.NaT), errors="coerce") if dates_row is not None else pd.NaT

        batch_sheets = list(UG_BATCH_SHEETS if program == "UG" else PG_BATCH_SHEETS)
        tx_sheet = "Tetr-X-UG" if program == "UG" else "Tetr-X-PG"
        candidate_sheets = batch_sheets + ([tx_sheet] if tx_sheet in activities else [])

        event_records = []
        for sheet in candidate_sheets:
            if sheet not in activities or sheet not in activity_ctx:
                continue
            sdf = activities[sheet]
            if sdf.empty:
                continue
            mask = pd.Series(False, index=sdf.index)
            if email_key and "email_key" in sdf.columns:
                mask = mask | sdf["email_key"].astype(str).eq(email_key)
            if student_key and "student_key" in sdf.columns:
                mask = mask | sdf["student_key"].astype(str).eq(student_key)
            part = sdf.loc[mask].copy()
            if part.empty:
                continue
            event_info = activity_ctx[sheet].get("event_info", pd.DataFrame())
            if event_info is None or event_info.empty:
                continue
            for _, prow in part.iterrows():
                for _, ev in event_info.iterrows():
                    col = ev.get("column_name")
                    if not col or col not in prow.index:
                        continue
                    attended = pd.to_numeric(pd.Series([prow.get(col, 0)]), errors="coerce").fillna(0).iloc[0]
                    if attended <= 0:
                        continue
                    ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
                    if pd.isna(ev_date):
                        continue
                    ev_date = ev_date.normalize()
                    delta = (ev_date - pay_dt).days
                    source_group = "tetrx" if sheet == tx_sheet else "batch"
                    in_first30 = False
                    if pd.notna(offered_dt) and pd.notna(deadline_dt) and offered_dt.normalize() <= ev_date <= deadline_dt.normalize():
                        in_first30 = ((source_group == "batch" and ev_date < pay_dt) or (source_group == "tetrx" and ev_date >= pay_dt))
                    in_tminus7 = source_group == "batch" and -7 <= delta <= 0
                    in_tplus7 = 0 <= delta <= 7
                    post_payment = ev_date >= pay_dt
                    if not (in_first30 or in_tminus7 or in_tplus7 or post_payment):
                        continue
                    ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
                    ev_type = clean_text(ev.get("event_type", "Other")) or "Other"
                    dedupe_key = "|".join([
                        student_key or email_key or normalize_name(student_name),
                        normalize_name(ev_name),
                        normalize_name(ev_type),
                        ev_date.strftime('%Y-%m-%d'),
                    ])
                    occurrence_key = "|".join([
                        program,
                        normalize_name(ev_name),
                        normalize_name(ev_type),
                        ev_date.strftime('%Y-%m-%d'),
                    ])
                    event_records.append({
                        "student_name": student_name,
                        "program": program,
                        "batch": batch,
                        "payment_date": pay_dt,
                        "event_name": ev_name,
                        "event_type": ev_type,
                        "event_date": ev_date,
                        "source_sheet": sheet,
                        "source_group": source_group,
                        "delta": delta,
                        "dedupe_key": dedupe_key,
                        "occurrence_key": occurrence_key,
                        "in_first30": in_first30,
                        "in_tminus7": in_tminus7,
                        "in_tplus7": in_tplus7,
                        "post_payment": post_payment,
                    })

        ev_df = pd.DataFrame(event_records)
        if not ev_df.empty:
            ev_df = ev_df.sort_values(["event_date", "event_name", "source_sheet"]).drop_duplicates(subset=["dedupe_key"]).reset_index(drop=True)
            event_rows.append(ev_df)

        summary_rows.append({
            "student_name": student_name,
            "program": program,
            "batch": batch,
            "payment_date": pay_dt,
            "first30_count": int(ev_df.loc[ev_df["in_first30"], "dedupe_key"].nunique()) if not ev_df.empty else 0,
            "tminus7_count": int(ev_df.loc[ev_df["in_tminus7"], "dedupe_key"].nunique()) if not ev_df.empty else 0,
            "tplus7_count": int(ev_df.loc[ev_df["in_tplus7"], "dedupe_key"].nunique()) if not ev_df.empty else 0,
            "post_payment_count": int(ev_df.loc[ev_df["post_payment"], "dedupe_key"].nunique()) if not ev_df.empty else 0,
        })

    summary_df = pd.DataFrame(summary_rows)
    events_df = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame()
    return summary_df, events_df



def build_count_distribution(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").fillna(0).astype(int)
    if s.empty:
        return pd.DataFrame(columns=["Activities", "Students"]), 0
    max_val = int(s.max())
    rows = []
    zero_streak = 0
    for n in range(0, max_val + 2):
        cnt = int((s == n).sum())
        if n <= max_val:
            rows.append({"Activities": n, "Students": cnt})
        else:
            if cnt == 0:
                zero_streak += 1
        if n > max_val and zero_streak >= 1:
            break
    dist_df = pd.DataFrame(rows)
    at_least_one = int((s >= 1).sum())
    return dist_df, at_least_one


def summarize_event_types(events_df: pd.DataFrame, flag_col: str, eligible_students: int | None = None):
    cols = [
        "Event Type", "Student Count", "Attendance Hits", "Event Occurrences",
        "Avg Attendance per Occurrence", "Unique Student Reach %", "Repeat Pull", "Performance Score"
    ]
    if events_df.empty or flag_col not in events_df.columns:
        return pd.DataFrame(columns=cols)
    frame = events_df[events_df[flag_col]].copy()
    if frame.empty:
        return pd.DataFrame(columns=cols)
    eligible = int(eligible_students if eligible_students is not None else frame["student_name"].nunique())
    if eligible <= 0:
        eligible = int(frame["student_name"].nunique())
    out = frame.groupby("event_type", as_index=False).agg(
        **{
            "Student Count": ("student_name", "nunique"),
            "Attendance Hits": ("dedupe_key", "nunique"),
            "Event Occurrences": ("occurrence_key", "nunique"),
        }
    ).rename(columns={"event_type": "Event Type"})
    out["Avg Attendance per Occurrence"] = np.where(
        out["Event Occurrences"] > 0,
        out["Attendance Hits"] / out["Event Occurrences"],
        0.0,
    )
    out["Unique Student Reach %"] = np.where(
        eligible > 0,
        out["Student Count"] / eligible * 100,
        0.0,
    )
    out["Repeat Pull"] = np.where(
        out["Student Count"] > 0,
        out["Attendance Hits"] / out["Student Count"],
        0.0,
    )
    score_components = [
        ("Avg Attendance per Occurrence", 0.5),
        ("Unique Student Reach %", 0.3),
        ("Repeat Pull", 0.2),
    ]
    score = 0
    for col, weight in score_components:
        max_val = float(out[col].max()) if not out.empty else 0.0
        norm = (out[col] / max_val) if max_val > 0 else 0.0
        score = score + weight * norm
    out["Performance Score"] = score * 100
    return out.sort_values(
        ["Performance Score", "Avg Attendance per Occurrence", "Unique Student Reach %", "Repeat Pull", "Attendance Hits"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def render_distribution_block(title: str, series: pd.Series, key_prefix: str):
    dist_df, at_least_one = build_count_distribution(series)
    st.markdown(f"#### {title}")
    if dist_df.empty:
        st.info(f"No data available for {title}.")
        return
    c1, c2 = st.columns([1.35, 1])
    with c1:
        plot_df = dist_df.copy()
        plot_df["Activities Label"] = plot_df["Activities"].astype(str)
        fig = px.bar(plot_df, x="Activities Label", y="Students", title=f"Students by Activity Count · {title}")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(nice_layout(fig, height=320), use_container_width=True, key=f"{key_prefix}_distplot")
    with c2:
        c21, c22 = st.columns(2)
        c21.metric("0 activities", f"{int((series.fillna(0).astype(int) == 0).sum()):,}")
        c22.metric("At least 1 activity", f"{at_least_one:,}")
        st.dataframe(dist_df, use_container_width=True, height=260, key=f"{key_prefix}_disttable")


def render_event_type_block(title: str, events_df: pd.DataFrame, flag_col: str, key_prefix: str, eligible_students: int | None = None):
    summary = summarize_event_types(events_df, flag_col, eligible_students=eligible_students)
    st.markdown(f"#### {title}")
    if summary.empty:
        st.info(f"No event-type attendance found for {title}.")
        return
    top_row = summary.iloc[0]
    low_row = summary.iloc[-1]
    c1, c2 = st.columns(2)
    c1.metric(
        "Best Performing Event Type",
        top_row["Event Type"],
        delta=f"Score {float(top_row['Performance Score']):.1f} | Avg/Occ {float(top_row['Avg Attendance per Occurrence']):.2f}"
    )
    c2.metric(
        "Lowest Performing Event Type",
        low_row["Event Type"],
        delta=f"Score {float(low_row['Performance Score']):.1f} | Avg/Occ {float(low_row['Avg Attendance per Occurrence']):.2f}"
    )
    v1, v2 = st.columns([1.35, 1])
    with v1:
        plot_df = summary.head(12).copy()
        fig = px.bar(
            plot_df,
            x="Event Type",
            y="Performance Score",
            hover_data=["Student Count", "Attendance Hits", "Event Occurrences", "Avg Attendance per Occurrence", "Unique Student Reach %", "Repeat Pull"],
            title=f"Event Type Performance Score · {title}"
        )
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=340, x_tickangle=-25), use_container_width=True, key=f"{key_prefix}_etypeplot")
    with v2:
        st.dataframe(summary, use_container_width=True, height=320, key=f"{key_prefix}_etypetable")


def render_retention_page(data):
    st.subheader("Retention")
    summary_df, events_df = build_retention_data(data)
    if summary_df.empty:
        st.warning("No admitted/paid students with usable payment dates were found.")
        return

    total = int(len(summary_df))
    ug_total = int((summary_df["program"] == "UG").sum())
    pg_total = int((summary_df["program"] == "PG").sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Admitted / Paid / Tetr X Students", f"{total:,}")
    k2.metric("UG", f"{ug_total:,}")
    k3.metric("PG", f"{pg_total:,}")

    render_distribution_block("First 30 Days", summary_df["first30_count"], "ret_first30")
    render_event_type_block("First 30 Days Event Type Performance", events_df, "in_first30", "ret_first30", eligible_students=total)

    render_distribution_block("T-7", summary_df["tminus7_count"], "ret_tminus7")
    render_event_type_block("T-7 Event Type Performance", events_df, "in_tminus7", "ret_tminus7", eligible_students=total)

    render_distribution_block("T+7", summary_df["tplus7_count"], "ret_tplus7")
    render_event_type_block("T+7 Event Type Performance", events_df, "in_tplus7", "ret_tplus7", eligible_students=total)

    render_distribution_block("Post Payment Journey", summary_df["post_payment_count"], "ret_post")
    c1, c2 = st.columns(2)
    c1.metric("Students Active Post Payment", f"{int((summary_df['post_payment_count'] >= 1).sum()):,}")
    c2.metric("Students with 0 Post Payment Activities", f"{int((summary_df['post_payment_count'] == 0).sum()):,}")
    render_event_type_block("Post Payment Journey Event Type Performance", events_df, "post_payment", "ret_post", eligible_students=total)
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
        default_pages = ["Overview", "Student Profile", "UG", "PG", "UG vs PG", "Tetr-X", "T-7 & T+7 Analysis", "Retention"]
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
    elif page == "T-7 & T+7 Analysis":
        render_t7_analysis_page(data)
    elif page == "Retention":
        render_retention_page(data)


if __name__ == "__main__":
    main()
