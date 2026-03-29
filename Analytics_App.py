import json
import re
from io import BytesIO
from pathlib import Path

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
ALL_REQUIRED = MASTER_SHEETS + DETAIL_SHEETS

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

PAID_HINTS = {"paid", "admitted"}


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
        h1, h2, h3 {{ color: {DARK} !important; }}
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


def normalize_yes_no(x):
    s = clean_text(x).lower()
    return 1 if s in {"yes", "y", "1", "true", "present", "attended", "done"} else 0


def parse_date_safe(x):
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT


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


def best_matching_col(df: pd.DataFrame, candidates):
    lowered = {c: clean_text(c).lower() for c in df.columns}
    for cand in candidates:
        for col, low in lowered.items():
            if cand in low:
                return col
    return None


def find_header_row(raw: pd.DataFrame, max_scan=25):
    for i in range(min(max_scan, len(raw))):
        row = " | ".join(clean_text(v).lower() for v in raw.iloc[i].tolist())
        if "student name" in row or ("email" in row and "country" in row and "payment" in row):
            return i
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


def is_probably_event_series(series: pd.Series) -> bool:
    s = series.fillna("").astype(str).str.strip().str.lower()
    allowed = {"yes", "no", "y", "n", "true", "false", "1", "0", "attended", "present", "absent", "done", "", "nan"}
    return ((s.isin(allowed)).mean() >= 0.55) if len(s) else False


def load_master_sheet(raw: pd.DataFrame, program: str):
    header_row = find_header_row(raw, max_scan=20)
    if header_row is None:
        raise ValueError(f"Could not detect header row in master sheet for {program}.")

    df = raw.iloc[header_row + 1 :].copy().reset_index(drop=True)
    df.columns = make_unique(raw.iloc[header_row].tolist())
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["student name", "name"])
    email_col = best_matching_col(df, ["email"])
    country_col = best_matching_col(df, ["country"])
    batch_col = best_matching_col(df, ["batch"])
    income_col = best_matching_col(df, ["income", "household income"])
    engagement_pct_col = best_matching_col(df, ["overall engagement %", "engagement %"])
    engagement_score_col = best_matching_col(df, ["overall engagement score", "engagement score"])
    payment_status_col = best_matching_col(df, ["payment status", "status"])
    payment_date_col = best_matching_col(df, ["payment date"])

    if name_col is None:
        raise ValueError(f"Name column not found in master sheet for {program}.")

    df = df[df[name_col].astype(str).str.strip().ne("")].copy()
    df["Program"] = program
    df["Batch"] = df[batch_col].astype(str).str.strip() if batch_col else ""

    if engagement_pct_col and engagement_pct_col in df.columns:
        df["engagement_pct"] = pd.to_numeric(df[engagement_pct_col], errors="coerce").fillna(0)
        # convert fractions into percentages when needed
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
    header_row = find_header_row(raw, max_scan=20)
    if header_row is None:
        raise ValueError(f"Could not detect header row in {sheet_name}.")

    event_category_row = header_row - 5 if header_row >= 5 else None
    event_name_row = header_row - 4 if header_row >= 4 else None
    event_date_row = header_row - 3 if header_row >= 3 else None

    df = raw.iloc[header_row + 1 :].copy().reset_index(drop=True)
    headers = make_unique(raw.iloc[header_row].tolist())
    df.columns = headers
    df = df.dropna(how="all")

    name_col = best_matching_col(df, ["student name", "name"])
    email_col = best_matching_col(df, ["email"])
    country_col = best_matching_col(df, ["country"])
    income_col = best_matching_col(df, ["income"])
    payment_status_col = best_matching_col(df, ["payment status", "status"])
    payment_date_col = best_matching_col(df, ["payment date"])
    engagement_pct_col = best_matching_col(df, ["overall engagement %", "engagement %"])
    engagement_score_col = best_matching_col(df, ["overall engagement score", "engagement score"])

    if name_col is None:
        raise ValueError(f"Name column not found in {sheet_name}.")

    event_cols = []
    event_dates = {}
    for idx, col in enumerate(headers):
        if idx < 19:
            continue
        if col.startswith("Unnamed"):
            continue
        ser = df[col] if col in df.columns else pd.Series(dtype=object)
        if is_probably_event_series(ser):
            event_cols.append(col)
            if event_date_row is not None and idx < raw.shape[1]:
                dt = parse_event_date(raw.iloc[event_date_row, idx])
                if pd.notna(dt):
                    event_dates[col] = dt

    df = df[df[name_col].astype(str).str.strip().ne("")].copy()
    df["Program"] = infer_program_from_sheet(sheet_name)
    df["Batch"] = infer_batch_from_sheet_name(sheet_name)

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


def _get_gsheets_client(credentials_payload: str):
    key_dict = json.loads(credentials_payload)
    creds = Credentials.from_service_account_info(key_dict, scopes=GSHEETS_SCOPES)
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

    masters = {}
    details = {}
    master_contexts = {}
    detail_contexts = {}

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


def render_overview(overview_df, ctx):
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
    m4.metric("UG vs PG", f"{ug_students:,} / {pg_students:,}", delta="UG / PG")

    g1, g2, g3 = st.columns([1.2, 1, 1])
    with g1:
        st.plotly_chart(gauge_chart(total_students, "Total Students", maximum=max(total_students, 1)), use_container_width=True)
    with g2:
        st.plotly_chart(donut_chart(["UG", "PG"], [ug_students, pg_students], "UG / PG Distribution"), use_container_width=True)
    with g3:
        st.plotly_chart(donut_chart(["UG Paid", "PG Paid"], [ug_paid, pg_paid], "Paid Distribution"), use_container_width=True)

    a1, a2 = st.columns(2)
    with a1:
        batch_df = overview_df.copy()
        if batch_col and batch_col in batch_df.columns:
            batch_plot = (
                batch_df.groupby(batch_col, dropna=False)[name_col]
                .count().reset_index(name="Students")
                .sort_values("Students", ascending=False)
            )
            batch_plot[batch_col] = batch_plot[batch_col].replace("", "Unknown")
            fig = px.bar(batch_plot, x=batch_col, y="Students", title="Students by Batch")
            fig.update_traces(marker_color=GREEN_2)
            st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-25), use_container_width=True)

    with a2:
        status_plot = (
            overview_df.groupby(["Program", "paid_label"])[name_col]
            .count().reset_index(name="Students")
        )
        fig = px.bar(status_plot, x="Program", y="Students", color="paid_label", barmode="group", title="Paid vs Not Paid by Program",
                     color_discrete_map={"Paid / Admitted": GREEN, "Not Paid": GREEN_4})
        st.plotly_chart(nice_layout(fig, height=380), use_container_width=True)

    b1, b2 = st.columns(2)
    with b1:
        if country_col and country_col in overview_df.columns:
            country_plot = (
                overview_df.groupby(country_col)[name_col]
                .count().reset_index(name="Students")
                .sort_values("Students", ascending=False)
                .head(12)
            )
            if not country_plot.empty:
                fig = px.bar(country_plot, x=country_col, y="Students", title="Top Countries")
                fig.update_traces(marker_color=GREEN_3)
                st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-30), use_container_width=True)
    with b2:
        if income_col and income_col in overview_df.columns:
            income_plot = (
                overview_df.groupby(income_col)[name_col]
                .count().reset_index(name="Students")
                .sort_values("Students", ascending=False)
            )
            if not income_plot.empty:
                fig = px.bar(income_plot, x=income_col, y="Students", title="Income Distribution")
                fig.update_traces(marker_color=GREEN)
                st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-25), use_container_width=True)

    st.markdown("#### Overview Table")
    preview_cols = [c for c in [name_col, "Program", batch_col, country_col, income_col, "engagement_pct", "engagement_score", "paid_label"] if c and c in overview_df.columns]
    st.dataframe(overview_df[preview_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False), use_container_width=True, height=420)


def render_detail_tab(sheet_name, df, ctx):
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
    k2.metric("Active", f"{active_students:,}", delta=f"{(active_students/total_students*100 if total_students else 0):.1f}%")
    k3.metric("Paid / Admitted", f"{paid_students:,}", delta=f"{(paid_students/total_students*100 if total_students else 0):.1f}%")
    k4.metric("Avg Engagement", f"{avg_engagement:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="engagement_pct", nbins=10, title="Engagement Distribution")
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True)
    with c2:
        status = df["paid_label"].value_counts().reset_index()
        status.columns = ["Status", "Students"]
        fig = px.pie(status, names="Status", values="Students", hole=0.58,
                     color="Status", color_discrete_map={"Paid / Admitted": GREEN, "Not Paid": GREEN_4})
        fig.update_layout(title="Paid Status")
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        if event_cols:
            event_counts = pd.DataFrame({
                "Event": event_cols,
                "Participants": [int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in event_cols],
            }).sort_values("Participants", ascending=False).head(12)
            fig = px.bar(event_counts, x="Participants", y="Event", orientation="h", title="Top Events by Participation")
            fig.update_traces(marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=460), use_container_width=True)
    with d2:
        if country_col and country_col in df.columns:
            top_country = (
                df.groupby(country_col)[name_col]
                .count().reset_index(name="Students")
                .sort_values("Students", ascending=False)
                .head(10)
            )
            fig = px.bar(top_country, x=country_col, y="Students", title="Country Split")
            fig.update_traces(marker_color=GREEN_3)
            st.plotly_chart(nice_layout(fig, height=430, x_tickangle=-30), use_container_width=True)

    t1, t2 = st.columns(2)
    with t1:
        students = df[[name_col, "engagement_pct", "engagement_score", "paid_label"]].sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
        st.markdown("#### Top Students")
        st.dataframe(students, use_container_width=True, height=390)
    with t2:
        target = df[(~df["is_paid"]) & (df["engagement_pct"] > 0)][[name_col, "engagement_pct", "engagement_score", "paid_label"]]
        target = target.sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
        st.markdown("#### Best Upgrade Targets")
        st.dataframe(target, use_container_width=True, height=390)

    if event_cols:
        timeline = pd.DataFrame({
            "Event": event_cols,
            "Date": [event_dates.get(c, pd.NaT) for c in event_cols],
            "Participants": [int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in event_cols],
        }).dropna(subset=["Date"]).sort_values("Date")
        if not timeline.empty:
            fig = px.line(timeline, x="Date", y="Participants", markers=True, title="Participation Timeline")
            fig.update_traces(line_color=GREEN, marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=360), use_container_width=True)

    st.markdown("#### Full Student Table")
    display_cols = [c for c in [name_col, country_col, payment_date_col, "engagement_pct", "engagement_score", "paid_label"] if c and c in df.columns]
    event_preview = event_cols[:8]
    display_cols += [c for c in event_preview if c in df.columns]
    st.dataframe(df[display_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False), use_container_width=True, height=440)


def resolve_credentials_and_source():
    default_sheet_id = st.secrets.get("GSHEET_SPREADSHEET_ID", "") if hasattr(st, "secrets") else ""
    secrets_credentials = None
    if hasattr(st, "secrets"):
        if "GOOGLE_SERVICE_ACCOUNT" in st.secrets:
            secrets_credentials = json.dumps(dict(st.secrets["GOOGLE_SERVICE_ACCOUNT"]))
        elif "gcp_service_account" in st.secrets:
            secrets_credentials = json.dumps(dict(st.secrets["gcp_service_account"]))

    with st.sidebar:
        st.markdown("## Data source")
        options = ["Google Sheets (live)", "Excel upload (fallback)"] if GSPREAD_AVAILABLE else ["Excel upload (fallback)"]
        source_choice = st.radio("Source", options, index=0)
        spreadsheet_id = ""
        credentials_payload = None
        file_bytes = None
        connected_ok = False
        connection_note = ""

        if source_choice == "Google Sheets (live)":
            spreadsheet_id = st.text_input("Spreadsheet ID", value=default_sheet_id)
            credentials_mode = st.radio("Credentials", ["Use Streamlit secrets", "Paste JSON key"], index=0 if secrets_credentials else 1)
            if credentials_mode == "Use Streamlit secrets":
                credentials_payload = secrets_credentials
                if not credentials_payload:
                    st.info("Add GOOGLE_SERVICE_ACCOUNT or gcp_service_account to Streamlit secrets.")
            else:
                pasted = st.text_area("Service account JSON", height=180, help="Paste the full service account JSON.")
                credentials_payload = pasted.strip() or None

            if spreadsheet_id and credentials_payload:
                try:
                    gsheets_get_sheet_names(spreadsheet_id, credentials_payload)
                    connected_ok = True
                    connection_note = "Connected to Google Sheets"
                except Exception as e:
                    connected_ok = False
                    connection_note = f"Connection failed: {e}"
                    st.error(connection_note)
        else:
            uploaded_file = st.file_uploader("Excel workbook", type=["xlsx"])
            if uploaded_file is not None:
                file_bytes = uploaded_file.getvalue()
                connected_ok = True
                connection_note = "Using uploaded workbook"
            else:
                local_path = Path("New Master Engagement  (13).xlsx")
                if local_path.exists():
                    file_bytes = local_path.read_bytes()
                    connected_ok = True
                    connection_note = "Using local workbook"

        st.markdown("## Global filters")
        only_active = st.checkbox("Show only active students", value=False)
        only_paid = st.checkbox("Show only paid/admitted students", value=False)
        selected_programs = st.multiselect("Program filter", ["UG", "PG"], default=["UG", "PG"])

    source_mode = "gsheets" if source_choice == "Google Sheets (live)" else "excel"
    return {
        "source_mode": source_mode,
        "spreadsheet_id": spreadsheet_id,
        "credentials_payload": credentials_payload,
        "file_bytes": file_bytes,
        "connected_ok": connected_ok,
        "connection_note": connection_note,
        "only_active": only_active,
        "only_paid": only_paid,
        "selected_programs": selected_programs,
    }


def main():
    cfg = resolve_credentials_and_source()
    live_mode = cfg["source_mode"] == "gsheets"

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

    if not cfg["connected_ok"]:
        st.warning("Connect the Google Sheet or upload the workbook to load the dashboard.")
        st.stop()

    data = load_dashboard_data(
        cfg["source_mode"],
        spreadsheet_id=cfg["spreadsheet_id"],
        credentials_payload=cfg["credentials_payload"],
        file_bytes=cfg["file_bytes"],
    )

    if data["missing"]:
        st.warning("Missing expected sheets: " + ", ".join(data["missing"]))

    overview_df = data["overview_df"].copy()
    if not overview_df.empty:
        if cfg["selected_programs"]:
            overview_df = overview_df[overview_df["Program"].isin(cfg["selected_programs"])]
        if cfg["only_active"]:
            overview_df = overview_df[overview_df["is_active"]]
        if cfg["only_paid"]:
            overview_df = overview_df[overview_df["is_paid"]]

    for key in list(data["details"].keys()):
        ddf = data["details"][key]
        if cfg["selected_programs"]:
            ddf = ddf[ddf["Program"].isin(cfg["selected_programs"])]
        if cfg["only_active"]:
            ddf = ddf[ddf["is_active"]]
        if cfg["only_paid"]:
            ddf = ddf[ddf["is_paid"]]
        data["details"][key] = ddf

    overview_ctx = next(iter(data["master_contexts"].values())) if data["master_contexts"] else {
        "name_col": "Name", "country_col": None, "batch_col": "Batch", "income_col": None
    }

    tabs = st.tabs(["Overview"] + [s for s in DETAIL_SHEETS if s in data["details"]])

    with tabs[0]:
        render_overview(overview_df, overview_ctx)

    for i, sheet_name in enumerate([s for s in DETAIL_SHEETS if s in data["details"]], start=1):
        with tabs[i]:
            render_detail_tab(sheet_name, data["details"][sheet_name], data["detail_contexts"][sheet_name])


if __name__ == "__main__":
    main()
