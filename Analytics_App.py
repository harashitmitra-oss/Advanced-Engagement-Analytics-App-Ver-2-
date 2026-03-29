import json
import re
from io import BytesIO

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

st.set_page_config(page_title="Tetr Analytics", layout="wide")

MASTER_SHEETS = ["Master UG", "Master PG"]
DETAIL_SHEETS = ["UG B9", "UG B8", "UG B7", "UG B6", "PG B5"]
ALL_REQUIRED = MASTER_SHEETS + DETAIL_SHEETS

GREEN = "#0A4D38"
GREEN_DARK = "#063b2b"
GREEN_MID = "#2A855D"
GREEN_SOFT = "#DDF2E5"
GREEN_LINE = "#D6E8DD"
TEXT = "#0A3F2E"
MUTED = "#7E8F89"
BG = "#F7FAF8"
OFF = "#D9534F"

GSHEETS_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

PAID_HINTS = {"paid", "admitted"}

HARDCODED_SHEET_ID = "1By2Zb8vKQnTIQn72JRgyEuuRgO6ZZARCZ1JNklmf25U"
HARDCODED_CREDS = r'''{
  "type": "service_account",
  "project_id": "strong-summer-488709-b9",
  "private_key_id": "f25a1820b4e6ae7caa4767d00d1215f16c00e8fd",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBgkqhkiG9w0BAQEFAASCBKgwggSkAgEAAoIBAQCdFgRUX31oMQSX\nDaXNVA/PAB0FtwXtZFbEfySpbjXbZnQlmAF0StGcwctjCkw3oEMzdNwYhEYxrMjt\nYrXJToMBtrd+tQVNoK+gSnotsGSVw9TBKSWwRC6FUKxEG7ZjZyoDoMDDAM1bDBEI\n+N8rDKWGURQREWg3HrWtpJOAPAWJAJL+TbrjdbQVqGYlkR4O+1QXve+6O+juSZvG\nw9oSoMHIOJFtix8NsbmTX30cCWbeBlbEvNmygfF1HhTYZYOyrDhjZY0qB1QhjkVp\ns2QjWWZJNwJJ9eqQeIoOWHd/8Aa1cAcdA0LOTcZghXNtaTbTXmMCnfjOkgjcGFvo\nJftbimtrAgMBAAECggEAE/TmG+n9vqzrjl03gTx6vaugBEfaQuyKcXBNshDEWtlL\nTDNhi+qtcqLUOgLm5/I+V10zURIM8OaoqC/wNGD7F86kxT4uLEyLw2ty1jcgsD4u\n21Nk7F3dIya6m/SBWWOT3N4hXyTM8hI4X9FuWMPEi6nlSL3TZZ1LK4JLEvGNKnhr\nN4r4RIemWtqAPiNzQ7KGJuoHTNPHng1YxMuzNAIvcLDXvo+pfKOUL/CBzFj7VWtN\nu5DqbQZ76aWepqbsFEsKyFHzLumT+IEAH3uIJMc7kz75d9f1n9jVUA+fugQOffvo\nFjjxaTjUWRXNZelkHkcpMGH5QE0uXLT0dnQPFIhT4QKBgQDP9zvz7CnoAiUgfU8r\np4V85peDfDrO+G4IBCCZ0NEfK6O9bsNj61Rl482sh4u4iSbMQLo9E8JYmKsGlWG6\n5/dBwnOLi/bQc4rx9kEkpz+QgHsBvJHPA133vYOA1I78PFJqd4qLlpCKNlEs0VrC\n5KjuY9TIhzCuP6IWrivbLSMd6QKBgQDBXlLnwfu0YrDd9WFzQrF+mKOeRahpyO/3\nzo2a6IpZKSXw2rOPN9rPnHwtSXGzDeOv8FfICWmpNuvxqJ2+PSyoy/f8jGRtO1us\nesZSalkpxTZafjg3mTQ9jKMJb6zJOgKs4HlnRTLyrpiuTCa9+0aZGa1IEY9FB7qN\n7kMCtRgGMwKBgElXH0V+W6j+WKmEh48VnPXXPEeaYALtiaA6FGUqRxV/Blef4Dg0\nnabxF29ovdVuSMhvaz5u4XLtJCNGOxj3BTOjp6vmyDvrA20hMwgCE2CabsbGAYXH\n4jOkGeQtEd/SRh5V2f4wMvkK/sWXbzKcARdRDZFKW9iXiEoHUmARIvlBAoGBAK4B\n0nspzfaqpNxn9zTAfHcepoZDyuS+5GrMHhObVPwdEj5moBSuP6J6ACjEoaNuSUlG\n78db6RBUEwiZIrJR3IFdYyCJucmuE7XnmdYKS4hSJrJSSQaHQEJu7zwLmaJPKJ8n\nNTigRdOrGEwozOhDWWlmeM+Utad55//Wu8iQ7DiPAoGBAJFPPj8XXml0py8NIYUe\nGloR8bIeSMXKZPz2y9crIOF1eQzyREcIKOw6iWlXrRrg9EFxhJrZFbkZsixtPnQQ\nEmqWuvBkCVufoI5cfZdAwY6NyM6OU4R3zMeEszgHkWEPeDCCfhHbBigb8b9Qw0BW\nwmKSE2/iR2ueAYZxZMaN/zO+\n-----END PRIVATE KEY-----\n",
  "client_email": "tetr-101@strong-summer-488709-b9.iam.gserviceaccount.com",
  "client_id": "110965885023187393080",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/tetr-101%40strong-summer-488709-b9.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
'''


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background: {BG}; }}
        section[data-testid="stSidebar"] {{
            background: #EEF3F0;
            border-right: 1px solid #D9E4DD;
            min-width: 320px !important;
            max-width: 320px !important;
        }}
        .side-logo {{
            display:flex; align-items:center; gap:12px; margin-bottom: 26px;
        }}
        .side-icon {{
            width:48px; height:48px; border-radius:14px; background:{GREEN_DARK};
            color:white; display:flex; align-items:center; justify-content:center;
            font-size:26px; font-weight:800;
        }}
        .side-title {{ font-size:24px; font-weight:900; color:{GREEN_DARK}; line-height:1; }}
        .side-label {{ color:#2C6150; font-size:13px; font-weight:800; letter-spacing:1px; margin:16px 0 8px; }}
        .hero-row {{ display:flex; justify-content:space-between; align-items:flex-start; gap:18px; margin-bottom:18px; }}
        .hero-title {{ font-size:54px; font-weight:900; color:{GREEN_DARK}; line-height:1.05; }}
        .hero-sub {{ font-size:18px; color:#2E6F5B; font-weight:500; margin-top:8px; }}
        .live-pill {{
            display:inline-flex; align-items:center; gap:10px; white-space:nowrap;
            border:1px solid #BFE3CB; background:#DFF4E6; color:{GREEN};
            border-radius:999px; padding:10px 18px; font-weight:800; font-size:16px;
        }}
        .live-pill.offline {{ background:#FBE9E7; border-color:#F2C4BE; color:#7A241C; }}
        .hb {{ position:relative; width:12px; height:12px; }}
        .hb:before {{ content:''; position:absolute; inset:0; border-radius:50%; background:#22B35E; }}
        .hb:after {{ content:''; position:absolute; inset:0; border-radius:50%; background:rgba(34,179,94,.30); animation: ping 1.5s infinite; }}
        .off-dot {{ width:12px; height:12px; border-radius:50%; background:{OFF}; display:inline-block; }}
        @keyframes ping {{ 0% {{ transform:scale(1); opacity:1; }} 100% {{ transform:scale(2.5); opacity:0; }} }}
        .card {{
            background:white; border:1px solid {GREEN_LINE}; border-radius:22px;
            box-shadow: 0 4px 18px rgba(8,50,35,.05); padding:22px 24px;
        }}
        .metric-card {{ min-height:136px; }}
        .metric-label {{ color:#2D6E59; font-size:18px; font-weight:800; text-transform:uppercase; letter-spacing:.8px; }}
        .metric-value {{ color:{GREEN_DARK}; font-size:40px; font-weight:900; line-height:1.1; margin-top:8px; }}
        .nav-sep {{ margin:14px 0 8px; height:1px; background:#D7E1DB; }}
        .stRadio > div {{ gap: 10px; }}
        div[data-testid="stMetric"] {{
            background:white; border:1px solid {GREEN_LINE}; border-radius:20px; padding:10px 12px;
            box-shadow: 0 2px 10px rgba(8,50,35,.04);
        }}
        div[data-testid="stMetric"] label {{ color:#2D6E59 !important; font-weight:800 !important; }}
        .block-container {{ padding-top: 2rem; padding-bottom: 2rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def clean_text(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").replace("\xa0", " ").strip()


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


def normalize_yes_no(x):
    s = clean_text(x).lower()
    return 1 if s in {"yes", "y", "1", "true", "present", "attended", "done"} else 0


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


def best_matching_col(df: pd.DataFrame, candidates):
    lowered = {c: clean_text(c).lower() for c in df.columns}
    for cand in candidates:
        for col, low in lowered.items():
            if cand in low:
                return col
    return None


def find_header_row(raw: pd.DataFrame, max_scan=30):
    for i in range(min(max_scan, len(raw))):
        row = " | ".join(clean_text(v).lower() for v in raw.iloc[i].tolist())
        if "student name" in row or ("email" in row and "payment" in row):
            return i
    return None


def infer_program_from_sheet(sheet_name: str) -> str:
    s = clean_text(sheet_name).lower()
    if "ug" in s:
        return "UG"
    if "pg" in s:
        return "PG"
    return ""


def infer_batch_from_sheet_name(sheet_name: str) -> str:
    m = re.search(r"\b(b\d+)\b", clean_text(sheet_name).lower())
    return m.group(1).upper() if m else ""


def is_probably_event_series(series: pd.Series) -> bool:
    s = series.fillna("").astype(str).str.strip().str.lower()
    allowed = {"yes", "no", "y", "n", "true", "false", "1", "0", "attended", "present", "absent", "done", "", "nan"}
    return ((s.isin(allowed)).mean() >= 0.45) if len(s) else False


def build_event_name(category, name, dt):
    n = clean_text(name)
    c = clean_text(category)
    if not n:
        n = c
    if pd.notna(dt):
        return f"{n} -- {dt.strftime('%d %b %Y')}"
    return n


def get_credentials_payload():
    if hasattr(st, "secrets"):
        if "GOOGLE_SERVICE_ACCOUNT" in st.secrets:
            return json.dumps(dict(st.secrets["GOOGLE_SERVICE_ACCOUNT"]))
        if "gcp_service_account" in st.secrets:
            return json.dumps(dict(st.secrets["gcp_service_account"]))
    return HARDCODED_CREDS


def _get_gsheets_client(credentials_payload: str):
    creds_dict = json.loads(credentials_payload)
    creds = Credentials.from_service_account_info(creds_dict, scopes=GSHEETS_SCOPES)
    return gspread.authorize(creds)


@st.cache_data(show_spinner=False, ttl=180)
def gsheets_get_sheet_names(spreadsheet_id: str, credentials_payload: str):
    gc = _get_gsheets_client(credentials_payload)
    sh = gc.open_by_key(spreadsheet_id)
    return [ws.title for ws in sh.worksheets()]


@st.cache_data(show_spinner=False, ttl=180)
def gsheets_read_raw_sheet(spreadsheet_id: str, sheet_name: str, credentials_payload: str):
    gc = _get_gsheets_client(credentials_payload)
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(sheet_name)
    values = ws.get_all_values()
    if not values:
        return pd.DataFrame()
    max_cols = max(len(r) for r in values)
    padded = [r + [""] * (max_cols - len(r)) for r in values]
    df = pd.DataFrame(padded)
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


def load_raw_sheet(source_mode: str, sheet_name: str, spreadsheet_id=None, credentials_payload=None, file_bytes=None):
    if source_mode == "gsheets":
        return gsheets_read_raw_sheet(spreadsheet_id, sheet_name, credentials_payload)
    return excel_read_raw_sheet(file_bytes, sheet_name)


def get_sheet_names(source_mode: str, spreadsheet_id=None, credentials_payload=None, file_bytes=None):
    if source_mode == "gsheets":
        return gsheets_get_sheet_names(spreadsheet_id, credentials_payload)
    return excel_get_sheet_names(file_bytes)


def load_master_sheet(raw: pd.DataFrame, program: str):
    header_row = find_header_row(raw, max_scan=25)
    if header_row is None:
        raise ValueError(f"Could not detect header row in master sheet for {program}.")

    df = raw.iloc[header_row + 1 :].copy().reset_index(drop=True)
    df.columns = make_unique(raw.iloc[header_row].tolist())
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["student name", "name"])
    email_col = best_matching_col(df, ["email"])
    country_col = best_matching_col(df, ["country"])
    batch_col = best_matching_col(df, ["batch"])
    income_col = best_matching_col(df, ["income"])
    engagement_pct_col = best_matching_col(df, ["overall engagement %", "engagement %", "engagement"])
    engagement_score_col = best_matching_col(df, ["overall engagement score", "engagement score"])
    payment_status_col = best_matching_col(df, ["payment status", "status", "payment"])
    payment_date_col = best_matching_col(df, ["payment date"])

    if name_col is None:
        raise ValueError(f"Name column not found in master sheet for {program}.")

    df = df[df[name_col].astype(str).str.strip().ne("")].copy()
    df["Program"] = program
    df["Batch"] = df[batch_col].astype(str).str.strip() if batch_col else ""

    if engagement_pct_col and engagement_pct_col in df.columns:
        df["engagement_pct"] = pd.to_numeric(df[engagement_pct_col], errors="coerce").fillna(0)
        if df["engagement_pct"].max() <= 1.05:
            df["engagement_pct"] = df["engagement_pct"] * 100
    else:
        df["engagement_pct"] = 0

    if engagement_score_col and engagement_score_col in df.columns:
        df["engagement_score"] = pd.to_numeric(df[engagement_score_col], errors="coerce").fillna(0)
    else:
        df["engagement_score"] = 0

    payment_series = df[payment_status_col].astype(str).str.lower().str.strip() if payment_status_col else pd.Series("", index=df.index)
    df["is_paid"] = payment_series.apply(lambda s: any(h in s for h in PAID_HINTS))
    df["paid_label"] = np.where(df["is_paid"], "Paid / Admitted", "Not Paid")
    df["is_active"] = df["engagement_pct"] > 0
    if payment_date_col and payment_date_col in df.columns:
        df[payment_date_col] = df[payment_date_col].apply(parse_date_safe)

    ctx = {
        "name_col": name_col,
        "email_col": email_col,
        "country_col": country_col,
        "batch_col": "Batch",
        "income_col": income_col,
        "engagement_pct_col": engagement_pct_col,
        "engagement_score_col": engagement_score_col,
        "payment_status_col": payment_status_col,
        "payment_date_col": payment_date_col,
    }
    return df, ctx


def load_detail_sheet(raw: pd.DataFrame, sheet_name: str):
    header_row = find_header_row(raw, max_scan=30)
    if header_row is None:
        raise ValueError(f"Could not detect header row in {sheet_name}.")

    event_category_row = header_row - 5 if header_row >= 5 else None
    event_name_row = header_row - 4 if header_row >= 4 else None
    event_date_row = header_row - 3 if header_row >= 3 else None

    header_vals = raw.iloc[header_row].tolist()
    category_vals = raw.iloc[event_category_row].tolist() if event_category_row is not None else [""] * len(header_vals)
    event_name_vals = raw.iloc[event_name_row].tolist() if event_name_row is not None else [""] * len(header_vals)
    event_date_vals = raw.iloc[event_date_row].tolist() if event_date_row is not None else [""] * len(header_vals)

    cols = []
    event_dates = {}
    for idx, hv in enumerate(header_vals):
        h = clean_text(hv)
        ev_name = clean_text(event_name_vals[idx]) if idx < len(event_name_vals) else ""
        ev_cat = clean_text(category_vals[idx]) if idx < len(category_vals) else ""
        ev_dt = parse_event_date(event_date_vals[idx]) if idx < len(event_date_vals) else pd.NaT

        if h:
            cols.append(h)
        elif ev_name or ev_cat:
            label = build_event_name(ev_cat, ev_name, ev_dt)
            cols.append(label or f"Event_{idx}")
            if pd.notna(ev_dt):
                event_dates[label] = ev_dt
        else:
            cols.append(f"Unnamed_{idx}")
    cols = make_unique(cols)

    df = raw.iloc[header_row + 1 :].copy().reset_index(drop=True)
    df.columns = cols
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["student name", "name"])
    email_col = best_matching_col(df, ["email"])
    country_col = best_matching_col(df, ["country"])
    income_col = best_matching_col(df, ["income"])
    payment_status_col = best_matching_col(df, ["payment status", "status", "payment"])
    payment_date_col = best_matching_col(df, ["payment date"])
    engagement_pct_col = best_matching_col(df, ["overall engagement %", "engagement %"])
    engagement_score_col = best_matching_col(df, ["overall engagement score", "engagement score"])

    if name_col is None:
        raise ValueError(f"Name column not found in {sheet_name}.")

    df = df[df[name_col].astype(str).str.strip().ne("")].copy()
    df["Program"] = infer_program_from_sheet(sheet_name)
    df["Batch"] = infer_batch_from_sheet_name(sheet_name)

    metadata_cols = {
        c for c in [name_col, email_col, country_col, income_col, payment_status_col, payment_date_col, engagement_pct_col, engagement_score_col, "Program", "Batch"] if c
    }

    event_cols = []
    for col in df.columns:
        if col in metadata_cols:
            continue
        if col.startswith("Unnamed"):
            continue
        if is_probably_event_series(df[col]):
            event_cols.append(col)
            if col not in event_dates:
                dt = parse_event_date(col)
                if pd.notna(dt):
                    event_dates[col] = dt

    for col in event_cols:
        df[col] = df[col].apply(normalize_yes_no).astype(int)

    df["participation_count"] = df[event_cols].sum(axis=1) if event_cols else 0
    df["engagement_score"] = pd.to_numeric(df[engagement_score_col], errors="coerce").fillna(df["participation_count"]) if engagement_score_col else df["participation_count"]
    if engagement_pct_col:
        df["engagement_pct"] = pd.to_numeric(df[engagement_pct_col], errors="coerce").fillna(0)
        if df["engagement_pct"].max() <= 1.05:
            df["engagement_pct"] = df["engagement_pct"] * 100
    else:
        total_events = max(len(event_cols), 1)
        df["engagement_pct"] = (df["participation_count"] / total_events) * 100

    payment_series = df[payment_status_col].astype(str).str.lower().str.strip() if payment_status_col else pd.Series("", index=df.index)
    df["is_paid"] = payment_series.apply(lambda s: any(h in s for h in PAID_HINTS))
    df["paid_label"] = np.where(df["is_paid"], "Paid / Admitted", "Not Paid")
    df["is_active"] = df["engagement_pct"] > 0
    if payment_date_col:
        df[payment_date_col] = df[payment_date_col].apply(parse_date_safe)

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
    }
    return df, ctx


@st.cache_data(show_spinner=False, ttl=180)
def load_dashboard_data(source_mode: str, spreadsheet_id=None, credentials_payload=None, file_bytes=None):
    sheet_names = get_sheet_names(source_mode, spreadsheet_id, credentials_payload, file_bytes)
    missing = [s for s in ALL_REQUIRED if s not in sheet_names]

    masters, details, master_contexts, detail_contexts = {{}}, {{}}, {{}}, {{}}

    for sheet in MASTER_SHEETS:
        if sheet in sheet_names:
            raw = load_raw_sheet(source_mode, sheet, spreadsheet_id, credentials_payload, file_bytes)
            program = "UG" if sheet.endswith("UG") else "PG"
            masters[sheet], master_contexts[sheet] = load_master_sheet(raw, program)

    for sheet in DETAIL_SHEETS:
        if sheet in sheet_names:
            raw = load_raw_sheet(source_mode, sheet, spreadsheet_id, credentials_payload, file_bytes)
            details[sheet], detail_contexts[sheet] = load_detail_sheet(raw, sheet)

    overview_df = pd.concat(list(masters.values()), ignore_index=True) if masters else pd.DataFrame()
    return {
        "sheet_names": sheet_names,
        "missing": missing,
        "masters": masters,
        "details": details,
        "master_contexts": master_contexts,
        "detail_contexts": detail_contexts,
        "overview_df": overview_df,
    }


def nice_layout(fig, height=360, x_tickangle=None):
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color=TEXT),
        title_font=dict(color=TEXT),
        margin=dict(l=20, r=20, t=60, b=40),
        height=height,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#E8F0EB", tickangle=x_tickangle)
    fig.update_yaxes(showgrid=True, gridcolor="#E8F0EB")
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
            "bar": {"color": GREEN_DARK},
            "bgcolor": "white",
            "steps": [
                {"range": [0, maximum * 0.5], "color": "#EDF8F0"},
                {"range": [maximum * 0.5, maximum * 0.8], "color": "#CFEAD7"},
                {"range": [maximum * 0.8, maximum], "color": "#8FCEA8"},
            ],
        },
    ))
    return nice_layout(fig, height=290)


def donut_chart(labels, values, title):
    colors = [GREEN_DARK, GREEN_MID, "#5FA97F", "#9BCFB1", "#DDF2E5"]
    fig = go.Figure(go.Pie(labels=labels, values=values, hole=0.62, marker=dict(colors=colors[:len(labels)]), textinfo="label+percent"))
    fig.update_layout(title=title)
    return nice_layout(fig, height=330)


def live_status_html(is_connected: bool, mode_label: str):
    if is_connected:
        return f'<div class="live-pill"><span class="hb"></span> Live Data</div>'
    return f'<div class="live-pill offline"><span class="off-dot"></span> Offline</div>'


def plot_chart(fig, key):
    st.plotly_chart(fig, use_container_width=True, key=key)


def render_metric_card(label, value):
    st.markdown(
        f'''<div class="card metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div></div>''',
        unsafe_allow_html=True,
    )


def render_overview(overview_df, ctx):
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
    ug_students = int((overview_df["Program"] == "UG").sum())
    pg_students = int((overview_df["Program"] == "PG").sum())
    ug_paid = int(((overview_df["Program"] == "UG") & (overview_df["is_paid"])).sum())
    pg_paid = int(((overview_df["Program"] == "PG") & (overview_df["is_paid"])).sum())
    conv_rate = round((total_paid / total_students * 100), 1) if total_students else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1: render_metric_card("Total Students", f"{total_students:,}")
    with m2: render_metric_card("Active Students", f"{total_active:,}")
    with m3: render_metric_card("Paid / Admitted", f"{total_paid:,}")
    with m4: render_metric_card("Conversion Rate", f"{conv_rate:.1f}%")

    g1, g2, g3 = st.columns([1.1, 1, 1])
    with g1:
        plot_chart(gauge_chart(total_students, "Total Students", maximum=max(total_students, 1)), "ov_gauge")
    with g2:
        plot_chart(donut_chart(["UG", "PG"], [ug_students, pg_students], "UG / PG Distribution"), "ov_prog")
    with g3:
        plot_chart(donut_chart(["UG Paid", "PG Paid"], [ug_paid, pg_paid], "Paid Distribution"), "ov_paid")

    a1, a2 = st.columns(2)
    with a1:
        if batch_col and batch_col in overview_df.columns:
            batch_plot = overview_df.groupby(batch_col, dropna=False)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False)
            batch_plot[batch_col] = batch_plot[batch_col].replace("", "Unknown")
            fig = px.bar(batch_plot, x=batch_col, y="Students", title="Students by Batch")
            fig.update_traces(marker_color=GREEN_MID)
            plot_chart(nice_layout(fig, height=380, x_tickangle=-25), "ov_batch")
    with a2:
        status_plot = overview_df.groupby(["Program", "paid_label"])[name_col].count().reset_index(name="Students")
        fig = px.bar(status_plot, x="Program", y="Students", color="paid_label", barmode="group", title="Paid vs Not Paid by Program", color_discrete_map={"Paid / Admitted": GREEN_DARK, "Not Paid": "#B7DCC5"})
        plot_chart(nice_layout(fig, height=380), "ov_status")

    b1, b2 = st.columns(2)
    with b1:
        if country_col and country_col in overview_df.columns:
            country_plot = overview_df.groupby(country_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False).head(12)
            if not country_plot.empty:
                fig = px.bar(country_plot, x=country_col, y="Students", title="Top Countries")
                fig.update_traces(marker_color=GREEN_MID)
                plot_chart(nice_layout(fig, height=400, x_tickangle=-30), "ov_country")
    with b2:
        if income_col and income_col in overview_df.columns:
            income_plot = overview_df.groupby(income_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False)
            if not income_plot.empty:
                fig = px.bar(income_plot, x=income_col, y="Students", title="Income Distribution")
                fig.update_traces(marker_color=GREEN_DARK)
                plot_chart(nice_layout(fig, height=400, x_tickangle=-25), "ov_income")

    st.markdown("#### Overview Table")
    preview_cols = [c for c in [name_col, "Program", batch_col, country_col, income_col, "engagement_pct", "engagement_score", "paid_label"] if c and c in overview_df.columns]
    st.dataframe(overview_df[preview_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False), use_container_width=True, height=420, key="ov_table")


def render_detail_tab(sheet_name, df, ctx):
    if df.empty:
        st.warning(f"No data available for {sheet_name}.")
        return

    name_col = ctx["name_col"]
    country_col = ctx["country_col"]
    payment_date_col = ctx["payment_date_col"]
    event_cols = ctx["event_cols"]
    event_dates = ctx["event_dates"]

    total_students = int(df[name_col].count())
    active_students = int(df["is_active"].sum())
    paid_students = int(df["is_paid"].sum())
    avg_engagement = round(float(df["engagement_pct"].mean()), 1) if len(df) else 0

    k1, k2, k3, k4 = st.columns(4)
    with k1: render_metric_card("Students", f"{total_students:,}")
    with k2: render_metric_card("Active", f"{active_students:,}")
    with k3: render_metric_card("Paid / Admitted", f"{paid_students:,}")
    with k4: render_metric_card("Avg Engagement", f"{avg_engagement:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="engagement_pct", nbins=10, title="Engagement Distribution")
        fig.update_traces(marker_color=GREEN_MID)
        plot_chart(nice_layout(fig, height=360), f"{sheet_name}_hist")
    with c2:
        status = df["paid_label"].value_counts().reset_index()
        status.columns = ["Status", "Students"]
        fig = px.pie(status, names="Status", values="Students", hole=0.58, color="Status", color_discrete_map={"Paid / Admitted": GREEN_DARK, "Not Paid": "#B7DCC5"})
        fig.update_layout(title="Paid Status")
        plot_chart(nice_layout(fig, height=360), f"{sheet_name}_pie")

    d1, d2 = st.columns(2)
    with d1:
        if event_cols:
            event_counts = pd.DataFrame({
                "Event": event_cols,
                "Participants": [int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in event_cols],
            }).sort_values("Participants", ascending=False).head(12)
            fig = px.bar(event_counts, x="Participants", y="Event", orientation="h", title="Top Events by Participation")
            fig.update_traces(marker_color=GREEN_DARK)
            plot_chart(nice_layout(fig, height=460), f"{sheet_name}_events")
    with d2:
        if country_col and country_col in df.columns:
            top_country = df.groupby(country_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False).head(10)
            fig = px.bar(top_country, x=country_col, y="Students", title="Country Split")
            fig.update_traces(marker_color=GREEN_MID)
            plot_chart(nice_layout(fig, height=430, x_tickangle=-30), f"{sheet_name}_country")

    t1, t2 = st.columns(2)
    with t1:
        students = df[[name_col, "engagement_pct", "engagement_score", "paid_label"]].sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
        st.markdown("#### Top Students")
        st.dataframe(students, use_container_width=True, height=390, key=f"{sheet_name}_top")
    with t2:
        target = df[(~df["is_paid"]) & (df["engagement_pct"] > 0)][[name_col, "engagement_pct", "engagement_score", "paid_label"]]
        target = target.sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
        st.markdown("#### Best Upgrade Targets")
        st.dataframe(target, use_container_width=True, height=390, key=f"{sheet_name}_upgrade")

    if event_cols:
        timeline = pd.DataFrame({
            "Event": event_cols,
            "Date": [event_dates.get(c, pd.NaT) for c in event_cols],
            "Participants": [int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in event_cols],
        }).dropna(subset=["Date"]).sort_values("Date")
        if not timeline.empty:
            fig = px.line(timeline, x="Date", y="Participants", markers=True, title="Participation Timeline")
            fig.update_traces(line_color=GREEN_DARK, marker_color=GREEN_DARK)
            plot_chart(nice_layout(fig, height=360), f"{sheet_name}_timeline")

    st.markdown("#### Full Student Table")
    display_cols = [c for c in [name_col, country_col, payment_date_col, "engagement_pct", "engagement_score", "paid_label"] if c and c in df.columns]
    display_cols += [c for c in event_cols[:8] if c in df.columns]
    st.dataframe(df[display_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False), use_container_width=True, height=440, key=f"{sheet_name}_table")


def resolve_source():
    spreadsheet_id = HARDCODED_SHEET_ID
    credentials_payload = get_credentials_payload()
    file_bytes = None
    connection_note = ""

    if GSPREAD_AVAILABLE:
        with st.sidebar:
            st.markdown('<div class="side-label">DATA SOURCE</div>', unsafe_allow_html=True)
            source_choice = st.radio("", ["Google Sheets", "Upload Excel"], label_visibility="collapsed")
            if source_choice == "Google Sheets":
                spreadsheet_id = st.text_input("Spreadsheet ID", value=spreadsheet_id, label_visibility="collapsed")
                if st.button("Fetch Data", use_container_width=True):
                    st.cache_data.clear()
            else:
                uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"], label_visibility="collapsed")
                if uploaded_file is not None:
                    file_bytes = uploaded_file.getvalue()
    else:
        source_choice = "Upload Excel"
        with st.sidebar:
            uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"], label_visibility="collapsed")
            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()

    connected_ok = False
    error_message = ""
    source_mode = "gsheets" if source_choice == "Google Sheets" else "excel"

    try:
        if source_mode == "gsheets":
            gsheets_get_sheet_names(spreadsheet_id, credentials_payload)
            connected_ok = True
            connection_note = "Connected to Google Sheets"
        elif file_bytes is not None:
            excel_get_sheet_names(file_bytes)
            connected_ok = True
            connection_note = "Workbook loaded"
    except Exception as e:
        error_message = str(e)

    return {
        "source_mode": source_mode,
        "spreadsheet_id": spreadsheet_id,
        "credentials_payload": credentials_payload,
        "file_bytes": file_bytes,
        "connected_ok": connected_ok,
        "connection_note": connection_note,
        "error_message": error_message,
    }


def render_sidebar_pages(available_detail_sheets):
    with st.sidebar:
        st.markdown(
            '<div class="side-logo"><div class="side-icon">∿</div><div class="side-title">TETR ANALYTICS</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="nav-sep"></div>', unsafe_allow_html=True)
        st.markdown('<div class="side-label">NAVIGATION</div>', unsafe_allow_html=True)
        page = st.radio("", ["Overview"] + available_detail_sheets, label_visibility="collapsed")
    return page


def main():
    inject_css()
    cfg = resolve_source()
    live_mode = cfg["source_mode"] == "gsheets"

    right_col1, right_col2 = st.columns([6, 1.4])
    with right_col1:
        st.markdown(
            f'''<div class="hero-title">Engagement Analytics</div><div class="hero-sub">Overview from Master UG + Master PG • detailed batch analytics from the live sheet</div>''',
            unsafe_allow_html=True,
        )
    with right_col2:
        st.markdown(live_status_html(cfg["connected_ok"], "Google Sheets" if live_mode else "Workbook"), unsafe_allow_html=True)

    if not cfg["connected_ok"]:
        st.warning(cfg["error_message"] or "Connect a data source to begin analysis")
        return

    data = load_dashboard_data(
        cfg["source_mode"],
        spreadsheet_id=cfg["spreadsheet_id"],
        credentials_payload=cfg["credentials_payload"],
        file_bytes=cfg["file_bytes"],
    )

    if data["missing"]:
        st.info("Missing expected sheets: " + ", ".join(data["missing"]))

    available_detail_sheets = [s for s in DETAIL_SHEETS if s in data["details"]]
    page = render_sidebar_pages(available_detail_sheets)

    overview_ctx = next(iter(data["master_contexts"].values())) if data["master_contexts"] else {
        "name_col": "Name", "country_col": None, "batch_col": "Batch", "income_col": None
    }

    if page == "Overview":
        render_overview(data["overview_df"].copy(), overview_ctx)
    else:
        st.markdown(f"### {page}")
        render_detail_tab(page, data["details"][page], data["detail_contexts"][page])


if __name__ == "__main__":
    main()
