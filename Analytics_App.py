import json
import re
from io import BytesIO
from typing import Dict, List, Optional, Tuple

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
DETAIL_SHEETS = ["UG B9", "UG B8", "UG B7", "UG B6", "PG B5"]
NAV_ITEMS = ["Overview", "Student Profile"] + DETAIL_SHEETS
ALL_REQUIRED = MASTER_SHEETS + DETAIL_SHEETS
DEFAULT_SHEET_ID = "1By2Zb8vKQnTIQn72JRgyEuuRgO6ZZARCZ1JNklmf25U"
GSHEETS_SCOPES = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive",
]

GREEN = "#0b3d2e"
GREEN_2 = "#1f7a56"
GREEN_3 = "#56a77b"
GREEN_4 = "#9cd4b5"
GREEN_5 = "#dff3e7"
DARK = "#12372a"
LIGHT_BG = "#f7fbf8"
RED = "#d9534f"
PAID_HINTS = {"paid", "admitted"}
EVENT_RESPONSES = {"yes", "y", "1", "true", "present", "attended", "done", "no", "n", "0", "false", "absent", ""}
MASTER_SKIP_ROWS_AFTER_HEADER = 2


def inject_css() -> None:
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
        @keyframes heartbeatPing {{
            0% {{ transform: scale(0.9); opacity: 0.9; }}
            70% {{ transform: scale(2.2); opacity: 0; }}
            100% {{ transform: scale(2.2); opacity: 0; }}
        }}
        div[data-testid="stMetric"] {{
            background: #ffffff; border: 1px solid #dbeee0; border-radius: 16px;
            padding: 10px 12px; box-shadow: 0 2px 10px rgba(11, 61, 46, 0.05);
        }}
        div[data-testid="stMetric"] label {{ color: {GREEN_2} !important; font-weight: 700 !important; }}
        h1, h2, h3 {{ color: {DARK} !important; }}
        .stRadio > div {{ gap: 0.45rem; }}
        .stRadio label {{ width: 100%; }}
        .stRadio [role="radiogroup"] > label {{
            background: linear-gradient(180deg, #edf8f1 0%, #e3f3ea 100%);
            border: 1px solid #d5eadf;
            padding: 10px 12px;
            border-radius: 12px;
            margin-bottom: 6px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def clean_text(x) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    return str(x).replace("\n", " ").replace("\r", " ").replace("\xa0", " ").strip()


def normalize_name(x: str) -> str:
    return re.sub(r"\s+", " ", clean_text(x).lower()).strip()


def to_numeric_clean(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


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


def make_unique(cols: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out = []
    for c in cols:
        base = clean_text(c) or "Unnamed"
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}_{seen[base]}")
    return out


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


def best_matching_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    lowered = {c: clean_text(c).lower() for c in df.columns}
    for cand in candidates:
        cand = cand.lower()
        for col, low in lowered.items():
            if cand in low:
                return col
    return None


def find_master_header_row(raw: pd.DataFrame) -> Optional[int]:
    for i in range(min(15, len(raw))):
        row = [clean_text(v).lower() for v in raw.iloc[i].tolist()]
        hits = sum(any(token in cell for token in ["name", "email", "batch", "country", "status", "payment"]) for cell in row)
        if "name" in row[0] or hits >= 5:
            return i
    return None


def find_detail_header_row(raw: pd.DataFrame) -> Optional[int]:
    for i in range(min(20, len(raw))):
        row = " | ".join(clean_text(v).lower() for v in raw.iloc[i].tolist())
        if "student name" in row and "payment" in row:
            return i
    return None


def normalize_event_value(x) -> int:
    s = clean_text(x).lower()
    return 1 if s in {"yes", "y", "1", "true", "present", "attended", "done"} else 0


def is_event_response_series(series: pd.Series) -> bool:
    s = series.fillna("").astype(str).str.strip().str.lower()
    if len(s) == 0:
        return False
    return (s.isin(EVENT_RESPONSES).mean() >= 0.55)


def build_event_col_name(category, name, date_val) -> str:
    cat = clean_text(category)
    title = clean_text(name)
    date_txt = ""
    dt = parse_event_date(date_val)
    if pd.notna(dt):
        date_txt = dt.strftime("%Y-%m-%d")
    parts = [p for p in [cat, title, date_txt] if p]
    if not parts:
        return ""
    return " | ".join(parts)


def load_master_sheet(raw: pd.DataFrame, program: str) -> Tuple[pd.DataFrame, Dict]:
    header_row = find_master_header_row(raw)
    if header_row is None:
        raise ValueError(f"Could not detect header row in master sheet for {program}.")

    headers = make_unique(raw.iloc[header_row].tolist())
    start_row = min(header_row + 1 + MASTER_SKIP_ROWS_AFTER_HEADER, len(raw))
    df = raw.iloc[start_row:].copy().reset_index(drop=True)
    df.columns = headers
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["name", "student name"])
    email_col = best_matching_col(df, ["email"])
    country_col = best_matching_col(df, ["country"])
    batch_col = best_matching_col(df, ["batch"])
    income_col = best_matching_col(df, ["income", "household income"])
    payment_status_col = best_matching_col(df, ["payment", "payment status"])
    status_col = best_matching_col(df, ["status"])
    engagement_pct_col = best_matching_col(df, ["engagement %", "overall engagement %"])
    engagement_score_col = best_matching_col(df, ["engagement score", "overall engagement score"])

    if not name_col:
        raise ValueError(f"Name column not found in master sheet for {program}.")

    df[name_col] = df[name_col].map(clean_text)
    df = df[df[name_col].ne("")].copy()
    df["Program"] = program
    df["Batch"] = df[batch_col].map(clean_text) if batch_col else ""
    df["email_clean"] = df[email_col].map(clean_text) if email_col else ""
    df["country_clean"] = df[country_col].map(clean_text) if country_col else ""
    df["income_clean"] = df[income_col].map(clean_text) if income_col else ""

    if engagement_pct_col:
        df["engagement_pct"] = to_numeric_clean(df[engagement_pct_col]).fillna(0)
        if float(df["engagement_pct"].max() or 0) <= 1.05:
            df["engagement_pct"] = df["engagement_pct"] * 100
    else:
        event_candidates = []
        metadata = {name_col, email_col, country_col, batch_col, income_col, payment_status_col, status_col}
        for col in df.columns:
            if col in metadata:
                continue
            s = df[col].fillna("").astype(str).str.strip().str.lower()
            if s.isin(EVENT_RESPONSES).mean() >= 0.5:
                event_candidates.append(col)
        if event_candidates:
            df["engagement_pct"] = (df[event_candidates].applymap(normalize_event_value).sum(axis=1) / max(len(event_candidates), 1)) * 100
        else:
            df["engagement_pct"] = 0

    if engagement_score_col:
        df["engagement_score"] = to_numeric_clean(df[engagement_score_col]).fillna(0)
    else:
        df["engagement_score"] = df["engagement_pct"]

    payment_source_col = payment_status_col or status_col
    payment_series = df[payment_source_col].map(clean_text).str.lower() if payment_source_col else pd.Series("", index=df.index)
    df["payment_status_clean"] = payment_series
    df["is_paid"] = payment_series.apply(lambda s: any(h in s for h in PAID_HINTS))
    df["paid_label"] = np.where(df["is_paid"], "Paid / Admitted", "Not Paid")
    df["is_active"] = df["engagement_pct"] > 0

    ctx = {
        "name_col": name_col,
        "email_col": email_col,
        "country_col": country_col,
        "batch_col": "Batch",
        "income_col": income_col,
        "engagement_pct_col": engagement_pct_col,
        "engagement_score_col": engagement_score_col,
    }
    return df, ctx


def load_detail_sheet(raw: pd.DataFrame, sheet_name: str) -> Tuple[pd.DataFrame, Dict]:
    header_row = find_detail_header_row(raw)
    if header_row is None:
        raise ValueError(f"Could not detect header row in {sheet_name}.")

    headers = list(raw.iloc[header_row].tolist())
    name_row = raw.iloc[header_row - 4].tolist() if header_row >= 4 else [""] * len(headers)
    date_row = raw.iloc[header_row - 3].tolist() if header_row >= 3 else [""] * len(headers)
    enriched_headers = []
    event_dates = {}
    for idx, h in enumerate(headers):
        base = clean_text(h)
        if base:
            enriched_headers.append(base)
        else:
            built = build_event_col_name(raw.iloc[header_row - 5, idx] if header_row >= 5 else "", name_row[idx], date_row[idx])
            enriched_headers.append(built or f"Unnamed_{idx}")
        dt = parse_event_date(date_row[idx])
        if pd.notna(dt):
            event_dates[enriched_headers[-1]] = dt
    enriched_headers = make_unique(enriched_headers)

    df = raw.iloc[header_row + 1:].copy().reset_index(drop=True)
    df.columns = enriched_headers
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["student name", "name"])
    email_col = best_matching_col(df, ["email"])
    country_col = best_matching_col(df, ["country"])
    income_col = best_matching_col(df, ["income"])
    payment_status_col = best_matching_col(df, ["payment status", "status"])
    payment_date_col = best_matching_col(df, ["payment date"])
    engagement_pct_col = best_matching_col(df, ["overall engagement %", "engagement %"])
    engagement_score_col = best_matching_col(df, ["overall engagement score", "engagement score"])

    if not name_col:
        raise ValueError(f"Name column not found in {sheet_name}.")

    df[name_col] = df[name_col].map(clean_text)
    df = df[df[name_col].ne("")].copy()
    df["Program"] = infer_program_from_sheet(sheet_name)
    df["Batch"] = infer_batch_from_sheet_name(sheet_name)
    df["email_clean"] = df[email_col].map(clean_text) if email_col else ""
    df["country_clean"] = df[country_col].map(clean_text) if country_col else ""

    metadata_cols = {c for c in [name_col, email_col, country_col, income_col, payment_status_col, payment_date_col, engagement_pct_col, engagement_score_col] if c}
    event_cols = []
    for col in df.columns:
        if col in metadata_cols or col in {"Program", "Batch", "email_clean", "country_clean"}:
            continue
        if is_event_response_series(df[col]):
            event_cols.append(col)

    for col in event_cols:
        df[col] = df[col].apply(normalize_event_value).astype(int)

    df["participation_count"] = df[event_cols].sum(axis=1) if event_cols else 0
    if engagement_pct_col:
        df["engagement_pct"] = to_numeric_clean(df[engagement_pct_col]).fillna(0)
        if float(df["engagement_pct"].max() or 0) <= 1.05:
            df["engagement_pct"] = df["engagement_pct"] * 100
    else:
        df["engagement_pct"] = (df["participation_count"] / max(len(event_cols), 1)) * 100

    if engagement_score_col:
        df["engagement_score"] = to_numeric_clean(df[engagement_score_col]).fillna(df["participation_count"])
    else:
        df["engagement_score"] = df["participation_count"]

    payment_source = df[payment_status_col].map(clean_text).str.lower() if payment_status_col else pd.Series("", index=df.index)
    df["payment_status_clean"] = payment_source
    df["is_paid"] = payment_source.apply(lambda s: any(h in s for h in PAID_HINTS))
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


def resolve_streamlit_service_account() -> str:
    if not hasattr(st, "secrets"):
        raise ValueError("Streamlit secrets are not available.")
    if "GOOGLE_SERVICE_ACCOUNT" in st.secrets:
        return json.dumps(dict(st.secrets["GOOGLE_SERVICE_ACCOUNT"]))
    if "gcp_service_account" in st.secrets:
        return json.dumps(dict(st.secrets["gcp_service_account"]))
    raise ValueError("Missing GOOGLE_SERVICE_ACCOUNT in Streamlit secrets.")


def resolve_spreadsheet_id_default() -> str:
    if hasattr(st, "secrets") and "GSHEET_SPREADSHEET_ID" in st.secrets:
        return st.secrets["GSHEET_SPREADSHEET_ID"]
    return DEFAULT_SHEET_ID


def _get_gsheets_client(credentials_payload: str):
    info = json.loads(credentials_payload)
    creds = Credentials.from_service_account_info(info, scopes=GSHEETS_SCOPES)
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
    max_len = max(len(r) for r in values)
    padded = [r + [""] * (max_len - len(r)) for r in values]
    df = pd.DataFrame(padded)
    df.replace("", np.nan, inplace=True)
    return df.dropna(how="all").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def excel_get_sheet_names(file_bytes: bytes):
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return xls.sheet_names


@st.cache_data(show_spinner=False)
def excel_read_raw_sheet(file_bytes: bytes, sheet_name: str):
    xls = pd.ExcelFile(BytesIO(file_bytes))
    return pd.read_excel(xls, sheet_name=sheet_name, header=None).dropna(how="all").reset_index(drop=True)


def load_raw_sheet(source_mode: str, sheet_name: str, spreadsheet_id=None, credentials_payload=None, file_bytes=None):
    if source_mode == "gsheets":
        return gsheets_read_raw_sheet(spreadsheet_id, sheet_name, credentials_payload)
    return excel_read_raw_sheet(file_bytes, sheet_name)


def get_sheet_names(source_mode: str, spreadsheet_id=None, credentials_payload=None, file_bytes=None):
    if source_mode == "gsheets":
        return gsheets_get_sheet_names(spreadsheet_id, credentials_payload)
    return excel_get_sheet_names(file_bytes)


@st.cache_data(show_spinner=False, ttl=180)
def load_dashboard_data(source_mode: str, spreadsheet_id=None, credentials_payload=None, file_bytes=None):
    sheet_names = get_sheet_names(source_mode, spreadsheet_id, credentials_payload, file_bytes)
    missing = [s for s in ALL_REQUIRED if s not in sheet_names]

    masters, details, master_contexts, detail_contexts = { }, { }, { }, { }
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
        <div class=\"live-pill\">
            <span class=\"heartbeat-wrap\"><span class=\"heartbeat-ping\"></span><span class=\"heartbeat-dot\"></span></span>
            LIVE · {mode_label}
        </div>
        """
    return f"""
    <div class=\"live-pill offline\">
        <span class=\"offline-dot\"></span>
        OFFLINE · {mode_label}
    </div>
    """


def render_overview(overview_df, ctx, key_prefix="overview"):
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
    ug_students = int((overview_df["Program"] == "UG").sum())
    pg_students = int((overview_df["Program"] == "PG").sum())
    ug_paid = int(((overview_df["Program"] == "UG") & (overview_df["is_paid"])).sum())
    pg_paid = int(((overview_df["Program"] == "PG") & (overview_df["is_paid"])).sum())
    active_rate = round((total_active / total_students * 100), 1) if total_students else 0
    paid_rate = round((total_paid / total_students * 100), 1) if total_students else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Students", f"{total_students:,}")
    m2.metric("Active Students", f"{total_active:,}", delta=f"{active_rate}% active")
    m3.metric("Paid / Admitted", f"{total_paid:,}", delta=f"{paid_rate}% paid")
    m4.metric("UG vs PG", f"{ug_students:,} / {pg_students:,}")

    g1, g2, g3 = st.columns([1.2, 1, 1])
    with g1:
        st.plotly_chart(gauge_chart(total_students, "Total Students", maximum=max(total_students, 1)), use_container_width=True, key=f"{key_prefix}_gauge")
    with g2:
        st.plotly_chart(donut_chart(["UG", "PG"], [ug_students, pg_students], "UG / PG Distribution"), use_container_width=True, key=f"{key_prefix}_program")
    with g3:
        st.plotly_chart(donut_chart(["UG Paid", "PG Paid"], [ug_paid, pg_paid], "Paid Distribution"), use_container_width=True, key=f"{key_prefix}_paid")

    a1, a2 = st.columns(2)
    with a1:
        if batch_col in overview_df.columns:
            batch_plot = overview_df.groupby(batch_col, dropna=False)[name_col].count().reset_index(name="Students")
            batch_plot[batch_col] = batch_plot[batch_col].replace("", "Unknown")
            fig = px.bar(batch_plot.sort_values("Students", ascending=False), x=batch_col, y="Students", title="Students by Batch")
            fig.update_traces(marker_color=GREEN_2)
            st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-25), use_container_width=True, key=f"{key_prefix}_batch")
    with a2:
        status_plot = overview_df.groupby(["Program", "paid_label"])[name_col].count().reset_index(name="Students")
        fig = px.bar(status_plot, x="Program", y="Students", color="paid_label", barmode="group", title="Paid vs Not Paid by Program", color_discrete_map={"Paid / Admitted": GREEN, "Not Paid": GREEN_4})
        st.plotly_chart(nice_layout(fig, height=380), use_container_width=True, key=f"{key_prefix}_status")

    b1, b2 = st.columns(2)
    with b1:
        if country_col and country_col in overview_df.columns:
            country_plot = overview_df.groupby(country_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False).head(12)
            fig = px.bar(country_plot, x=country_col, y="Students", title="Top Countries")
            fig.update_traces(marker_color=GREEN_3)
            st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-30), use_container_width=True, key=f"{key_prefix}_country")
    with b2:
        if income_col and income_col in overview_df.columns:
            income_plot = overview_df.groupby(income_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False)
            fig = px.bar(income_plot, x=income_col, y="Students", title="Income Distribution")
            fig.update_traces(marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-25), use_container_width=True, key=f"{key_prefix}_income")

    preview_cols = [c for c in [name_col, "Program", batch_col, country_col, income_col, "engagement_pct", "engagement_score", "payment_status_clean", "paid_label"] if c and c in overview_df.columns]
    st.markdown("#### Overview Table")
    st.dataframe(overview_df[preview_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False), use_container_width=True, height=420, key=f"{key_prefix}_table")


def render_detail_tab(sheet_name, df, ctx, key_prefix="detail"):
    st.subheader(sheet_name)
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
    k1.metric("Students", f"{total_students:,}")
    k2.metric("Active", f"{active_students:,}", delta=f"{(active_students / total_students * 100 if total_students else 0):.1f}%")
    k3.metric("Paid / Admitted", f"{paid_students:,}", delta=f"{(paid_students / total_students * 100 if total_students else 0):.1f}%")
    k4.metric("Avg Engagement", f"{avg_engagement:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="engagement_pct", nbins=10, title="Engagement Distribution")
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{key_prefix}_hist")
    with c2:
        status = df["paid_label"].value_counts().reset_index()
        status.columns = ["Status", "Students"]
        fig = px.pie(status, names="Status", values="Students", hole=0.58, color="Status", color_discrete_map={"Paid / Admitted": GREEN, "Not Paid": GREEN_4})
        fig.update_layout(title="Paid Status")
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{key_prefix}_pie")

    d1, d2 = st.columns(2)
    with d1:
        if event_cols:
            event_counts = pd.DataFrame({
                "Event": event_cols,
                "Participants": [int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in event_cols],
            }).sort_values("Participants", ascending=False).head(12)
            fig = px.bar(event_counts, x="Participants", y="Event", orientation="h", title="Top Events by Participation")
            fig.update_traces(marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=460), use_container_width=True, key=f"{key_prefix}_events")
    with d2:
        if country_col and country_col in df.columns:
            top_country = df.groupby(country_col)[name_col].count().reset_index(name="Students").sort_values("Students", ascending=False).head(10)
            fig = px.bar(top_country, x=country_col, y="Students", title="Country Split")
            fig.update_traces(marker_color=GREEN_3)
            st.plotly_chart(nice_layout(fig, height=430, x_tickangle=-30), use_container_width=True, key=f"{key_prefix}_country")

    t1, t2 = st.columns(2)
    with t1:
        students = df[[name_col, "engagement_pct", "engagement_score", "paid_label"]].sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
        st.markdown("#### Top Students")
        st.dataframe(students, use_container_width=True, height=390, key=f"{key_prefix}_top")
    with t2:
        target = df[(~df["is_paid"]) & (df["engagement_pct"] > 0)][[name_col, "engagement_pct", "engagement_score", "paid_label"]]
        target = target.sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
        st.markdown("#### Best Upgrade Targets")
        st.dataframe(target, use_container_width=True, height=390, key=f"{key_prefix}_upgrade")

    if event_cols:
        timeline = pd.DataFrame({
            "Event": event_cols,
            "Date": [event_dates.get(c, pd.NaT) for c in event_cols],
            "Participants": [int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in event_cols],
        }).dropna(subset=["Date"]).sort_values("Date")
        if not timeline.empty:
            fig = px.line(timeline, x="Date", y="Participants", markers=True, title="Participation Timeline")
            fig.update_traces(line_color=GREEN, marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{key_prefix}_timeline")

    st.markdown("#### Full Student Table")
    display_cols = [c for c in [name_col, country_col, payment_date_col, "engagement_pct", "engagement_score", "payment_status_clean", "paid_label"] if c and c in df.columns]
    display_cols += [c for c in event_cols[:8] if c in df.columns]
    st.dataframe(df[display_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False), use_container_width=True, height=440, key=f"{key_prefix}_full")


def build_profile_index(data):
    profiles = {}

    def get_key(row, name_col):
        email = clean_text(row.get("email_clean", "")).lower()
        if email:
            return f"email:{email}"
        return f"name:{normalize_name(row.get(name_col, ''))}"

    for sheet, df in data["masters"].items():
        ctx = data["master_contexts"][sheet]
        name_col = ctx["name_col"]
        for _, row in df.iterrows():
            key = get_key(row, name_col)
            profiles.setdefault(key, {"master": None, "details": [], "name": clean_text(row[name_col]), "email": clean_text(row.get("email_clean", ""))})
            profiles[key]["master"] = row.to_dict()
            profiles[key]["name"] = clean_text(row[name_col])
            profiles[key]["email"] = clean_text(row.get("email_clean", ""))

    for sheet, df in data["details"].items():
        ctx = data["detail_contexts"][sheet]
        name_col = ctx["name_col"]
        for _, row in df.iterrows():
            key = get_key(row, name_col)
            profiles.setdefault(key, {"master": None, "details": [], "name": clean_text(row[name_col]), "email": clean_text(row.get("email_clean", ""))})
            profiles[key]["details"].append({"sheet": sheet, "row": row.to_dict(), "ctx": ctx})
            if not profiles[key]["name"]:
                profiles[key]["name"] = clean_text(row[name_col])
            if not profiles[key]["email"]:
                profiles[key]["email"] = clean_text(row.get("email_clean", ""))
    return profiles


def render_student_profile(data):
    st.subheader("Student Profile")
    profiles = build_profile_index(data)
    if not profiles:
        st.warning("No student data available.")
        return

    names = sorted({p["name"] for p in profiles.values() if p["name"]})
    query = st.text_input("Search student name", placeholder="Type a student name", key="profile_search")
    pasted = st.text_area("Search multiple names", placeholder="Paste one name per line", key="profile_multiline")
    pasted_names = [line.strip() for line in pasted.splitlines() if line.strip()]
    filtered = names
    if query:
        filtered = [n for n in filtered if query.lower() in n.lower()]
    if pasted_names:
        norm_targets = {normalize_name(n) for n in pasted_names}
        filtered = [n for n in names if normalize_name(n) in norm_targets]

    selected = st.multiselect("Select one or more students", options=filtered if filtered else names, default=(filtered[:1] if filtered else []), key="profile_select")
    targets = selected or filtered[:10]
    if not targets:
        st.info("No matching students found.")
        return

    name_to_profiles = {}
    for profile in profiles.values():
        name_to_profiles.setdefault(profile["name"], []).append(profile)

    for idx, title in enumerate(targets, start=1):
        matches = name_to_profiles.get(title, [])
        for match_idx, profile in enumerate(matches, start=1):
            st.markdown(f"### {title}")
            if profile.get("email"):
                st.caption(profile["email"])
            master = profile.get("master") or {}
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Program", clean_text(master.get("Program", "-")) or "-")
            c2.metric("Batch", clean_text(master.get("Batch", "-")) or "-")
            c3.metric("Paid", "Yes" if master.get("is_paid") else "No")
            c4.metric("Active", "Yes" if master.get("is_active") else "No")

            summary_rows = []
            if master:
                summary_rows.append({
                    "Source": "Master",
                    "Program": master.get("Program", ""),
                    "Batch": master.get("Batch", ""),
                    "Country": master.get("country_clean", ""),
                    "Payment Status": master.get("payment_status_clean", ""),
                    "Engagement %": round(float(master.get("engagement_pct", 0) or 0), 2),
                    "Engagement Score": round(float(master.get("engagement_score", 0) or 0), 2),
                })
            for entry in profile.get("details", []):
                row = entry["row"]
                summary_rows.append({
                    "Source": entry["sheet"],
                    "Program": row.get("Program", ""),
                    "Batch": row.get("Batch", ""),
                    "Country": row.get("country_clean", ""),
                    "Payment Status": row.get("payment_status_clean", ""),
                    "Engagement %": round(float(row.get("engagement_pct", 0) or 0), 2),
                    "Engagement Score": round(float(row.get("engagement_score", 0) or 0), 2),
                })
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, key=f"profile_summary_{normalize_name(title)}_{idx}_{match_idx}")

            attendance_rows = []
            for entry in profile.get("details", []):
                row = entry["row"]
                ctx = entry["ctx"]
                attended = [ev for ev in ctx["event_cols"] if int(row.get(ev, 0) or 0) == 1]
                if attended:
                    detail_rows = [{
                        "Sheet": entry["sheet"],
                        "Event": ev,
                        "Date": ctx["event_dates"].get(ev, pd.NaT),
                    } for ev in attended]
                    attendance_rows.extend(detail_rows)
            if attendance_rows:
                att_df = pd.DataFrame(attendance_rows).sort_values(["Date", "Event"], na_position="last")
                st.markdown("#### Attendance History")
                st.dataframe(att_df, use_container_width=True, key=f"profile_att_{normalize_name(title)}_{idx}_{match_idx}")


def resolve_credentials_and_source():
    spreadsheet_id = ""
    credentials_payload = None
    file_bytes = None
    connected_ok = False
    connection_note = ""
    debug_note = ""

    if GSPREAD_AVAILABLE:
        with st.sidebar:
            st.markdown("## 📡 Data Source")
            source_choice = st.radio("Source", ["Google Sheets (auto)", "Upload Excel (manual)"], index=0, key="source_choice")
    else:
        source_choice = "Upload Excel (manual)"
        with st.sidebar:
            st.warning("Install `gspread` and `google-auth` to enable Google Sheets auto-fetch.")

    if source_choice == "Google Sheets (auto)":
        spreadsheet_id = st.sidebar.text_input("Spreadsheet ID", value=resolve_spreadsheet_id_default(), help="Google Sheets ID from the URL", key="spreadsheet_id")
        try:
            credentials_payload = resolve_streamlit_service_account()
            sheet_names = gsheets_get_sheet_names(spreadsheet_id, credentials_payload)
            connected_ok = True
            connection_note = f"Connected to Google Sheets · {len(sheet_names)} tabs found"
        except Exception as e:
            connected_ok = False
            connection_note = f"Connection failed: {type(e).__name__}"
            debug_note = str(e)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Master Engagement Tracker Excel File", type=["xlsx"], key="upload_excel")
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            connected_ok = True
            connection_note = "Using uploaded workbook"

    return {
        "source_mode": "gsheets" if source_choice == "Google Sheets (auto)" else "excel",
        "spreadsheet_id": spreadsheet_id,
        "credentials_payload": credentials_payload,
        "file_bytes": file_bytes,
        "connected_ok": connected_ok,
        "connection_note": connection_note,
        "debug_note": debug_note,
    }


def main():
    inject_css()
    cfg = resolve_credentials_and_source()
    live_mode = cfg["source_mode"] == "gsheets"

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Navigation")
        page = st.radio("Navigation", NAV_ITEMS, index=0, label_visibility="collapsed", key="page_nav")

    st.markdown(
        f"""
        <div class="hero-card">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:34px;font-weight:900;color:{DARK};">📊 Tetr Business School Analytics Dashboard</div>
                    <div style="margin-top:6px;color:#2a6a52;font-size:16px;font-weight:600;">
                        Overview from <b>Master UG</b> + <b>Master PG</b> • Detailed analytics for <b>UG B9, UG B8, UG B7, UG B6, PG B5</b>
                    </div>
                </div>
                {live_status_html(cfg['connected_ok'], 'Google Sheets' if live_mode else 'Workbook')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(cfg["connection_note"])

    if not cfg["connected_ok"]:
        st.warning("Connect the Google Sheet or upload the workbook to load the dashboard.")
        if cfg["debug_note"]:
            st.code(cfg["debug_note"])
        st.stop()

    data = load_dashboard_data(
        cfg["source_mode"],
        spreadsheet_id=cfg["spreadsheet_id"],
        credentials_payload=cfg["credentials_payload"],
        file_bytes=cfg["file_bytes"],
    )

    if data["missing"]:
        st.warning("Missing expected sheets: " + ", ".join(data["missing"]))

    overview_ctx = next(iter(data["master_contexts"].values())) if data["master_contexts"] else {
        "name_col": "Name", "country_col": None, "batch_col": "Batch", "income_col": None
    }

    if page == "Overview":
        render_overview(data["overview_df"].copy(), overview_ctx, key_prefix="overview_page")
    elif page == "Student Profile":
        render_student_profile(data)
    else:
        if page in data["details"]:
            render_detail_tab(page, data["details"][page], data["detail_contexts"][page], key_prefix=f"detail_{page.lower().replace(' ', '_')}")
        else:
            st.warning(f"{page} is not available in the connected source.")


if __name__ == "__main__":
    main()
