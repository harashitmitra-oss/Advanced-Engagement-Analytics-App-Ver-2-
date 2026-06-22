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
UG_BATCH_SHEETS = ["UG - B1 to B4", "UG B5", "UG B6", "UG B7", "UG B8", "UG B9", "UG B10", "UG B11", "UG B12", "UG B13", "UG B14", "UG B15", "UG B16"]
PG_BATCH_SHEETS = ["PG - B1 & B2", "PG - B3 & B4", "PG B5", "PG B6", "PG B7", "PG B8"]
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

def normalize_phone(x):
    s = re.sub(r"\D+", "", clean_text(x))
    # keep last 10-12 digits to tolerate country codes while retaining uniqueness
    return s[-12:] if len(s) > 12 else s


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



def normalize_event_type_for_profile_graph(event_type: str) -> str:
    """Compatibility alias used by Courses event table.
    Keeps the same event-type grouping as Student Profile graphs.
    """
    return map_profile_plot_event_type(event_type)

def normalize_community_status(x):
    s = clean_text(x).strip().lower()
    if s in {"tetr x", "tetrx", "added to term 0"}:
        return "Tetr X"
    if s == "in":
        return "In"
    return "Out"


def is_community_in_value(x) -> bool:
    """Business rule for Joined WA/In Community counts.
    Treat In, TetrX, Tetr X, and Added to Term 0 as in-community.
    Works with both raw sheet values and normalized values.
    """
    s = clean_text(x).strip().lower().replace("-", " ")
    return s in {"in", "tetrx", "tetr x", "added to term 0"}


def is_community_in_series(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    return series.apply(is_community_in_value).astype(bool)


def parse_date_safe(x):
    try:
        return pd.to_datetime(x, errors="coerce", dayfirst=True)
    except Exception:
        return pd.NaT


def get_today_ist():
    return pd.Timestamp.now(tz="Asia/Kolkata").normalize().tz_localize(None)

def build_recent_activity_data(data, start_date=None, end_date=None):
    today = get_today_ist()
    if start_date is None or pd.isna(start_date):
        start_date = today - pd.Timedelta(days=6)
    else:
        start_date = pd.to_datetime(start_date, errors="coerce").normalize()
        if pd.isna(start_date):
            start_date = today - pd.Timedelta(days=6)
    if end_date is None or pd.isna(end_date):
        end_date = today
    else:
        end_date = pd.to_datetime(end_date, errors="coerce").normalize()
        if pd.isna(end_date):
            end_date = today
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    activities = data.get("activities", {})
    activity_ctx = data.get("activity_ctx", {})

    recent_event_rows = []
    batch_student_recent = {}

    def _empty_recent_df():
        return pd.DataFrame(columns=[
            "student_name", "email_key", "Batch", "Program", "status",
            "recent_attendance_count", "recent_events", "recent_active_dates", "reactivation_date"
        ])

    batch_sheet_list = [s for s in (UG_BATCH_SHEETS + PG_BATCH_SHEETS) if s in activities and s in activity_ctx]
    for sheet in batch_sheet_list:
        df = activities.get(sheet, pd.DataFrame())
        ctx = activity_ctx.get(sheet, {})
        event_info = ctx.get("event_info", pd.DataFrame())
        if df is None or df.empty or event_info is None or event_info.empty:
            batch_student_recent[sheet] = _empty_recent_df()
            continue
        recent_events = event_info[event_info["event_date"].notna()].copy()
        if recent_events.empty:
            batch_student_recent[sheet] = _empty_recent_df()
            continue
        recent_events["event_date"] = pd.to_datetime(recent_events["event_date"], errors="coerce").dt.normalize()
        recent_events = recent_events[recent_events["event_date"].between(start_date, end_date, inclusive="both")].copy()
        if recent_events.empty:
            batch_student_recent[sheet] = _empty_recent_df()
            continue

        active_rows = []
        for _, row in df.iterrows():
            attended_names = []
            attended_dates = []
            seen_recent_attendance = set()
            for _, ev in recent_events.iterrows():
                col = ev.get("column_name")
                if not col or col not in row.index:
                    continue
                attended = pd.to_numeric(pd.Series([row.get(col, 0)]), errors="coerce").fillna(0).iloc[0]
                if attended > 0:
                    ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
                    ev_type = clean_text(ev.get("event_type", "Other")) or "Other"
                    ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
                    dedupe_key = f"{normalize_name(ev_name)}|{clean_text(ev_type).lower()}|{ev_date.strftime('%Y-%m-%d') if pd.notna(ev_date) else ''}"
                    if dedupe_key in seen_recent_attendance:
                        continue
                    seen_recent_attendance.add(dedupe_key)
                    if pd.notna(ev_date):
                        ev_norm = ev_date.normalize()
                        attended_dates.append(ev_norm)
                        attended_names.append(f"{ev_name} ({ev_type}, {ev_norm.strftime('%d-%b-%Y')})")
                    else:
                        attended_names.append(f"{ev_name} ({ev_type})")
            unique_active_dates = sorted({d for d in attended_dates if pd.notna(d)})
            active_rows.append({
                "student_name": row.get("student_name", ""),
                "email_key": row.get("email_key", ""),
                "Batch": row.get("Batch", infer_batch_group_from_sheet_name(sheet)),
                "Program": row.get("Program", infer_program_from_sheet(sheet)),
                "status": clean_text(row.get("sheet_status_raw", row.get("status_value", ""))),
                "recent_attendance_count": len(attended_names),
                "recent_events": "; ".join(attended_names),
                "recent_active_dates": "; ".join([d.strftime('%d-%b-%Y') for d in unique_active_dates]),
                "reactivation_date": unique_active_dates[0].strftime('%d-%b-%Y') if unique_active_dates else "",
            })
        batch_student_recent[sheet] = pd.DataFrame(active_rows) if active_rows else _empty_recent_df()

        for _, ev in recent_events.iterrows():
            col = ev.get("column_name")
            if not col or col not in df.columns:
                continue
            attendees = int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum())
            recent_event_rows.append({
                "sheet": sheet,
                "program": infer_program_from_sheet(sheet),
                "batch": infer_batch_group_from_sheet_name(sheet),
                "event_name": clean_text(ev.get("event_name", "")) or clean_text(col),
                "event_type": clean_text(ev.get("event_type", "Other")) or "Other",
                "event_date": pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce"),
                "attendance": attendees,
            })

    recent_events_df = pd.DataFrame(recent_event_rows)

    recent_payments = []
    dates_df = data.get("dates_df", pd.DataFrame())
    tx_frames = []
    for tx_sheet in TX_SHEETS:
        tx_df = activities.get(tx_sheet, pd.DataFrame())
        if tx_df is None or tx_df.empty:
            continue
        frame = tx_df.copy()
        if "sheet_is_paid" in frame.columns:
            frame = frame[frame["sheet_is_paid"]].copy()
        else:
            frame = frame[frame.get("status_value", pd.Series("", index=frame.index)).astype(str).str.strip().str.lower().eq("admitted")].copy()
        if frame.empty:
            continue
        frame["payment_date_recent"] = pd.to_datetime(frame.get("payment_date_parsed", pd.NaT), errors="coerce").dt.normalize()
        frame = frame[frame["payment_date_recent"].between(start_date, end_date, inclusive="both")].copy()
        if frame.empty:
            continue
        frame["tx_sheet"] = tx_sheet
        tx_frames.append(frame)

    if tx_frames:
        tx_all = pd.concat(tx_frames, ignore_index=True)
        tx_all["recent_student_id"] = tx_all.apply(lambda r: clean_text(r.get("email_key", "")) or clean_text(r.get("student_key", "")) or normalize_name(r.get("student_name", "")), axis=1)
        tx_all = tx_all.sort_values(["payment_date_recent", "student_name"]).drop_duplicates(subset=["recent_student_id"], keep="first")
        for _, stu in tx_all.iterrows():
            student_name = clean_text(stu.get("student_name", ""))
            email_key = clean_text(stu.get("email_key", ""))
            student_key = clean_text(stu.get("student_key", ""))
            program = clean_text(stu.get("Program", "")) or infer_program_from_sheet(clean_text(stu.get("tx_sheet", "")))
            batch = clean_text(stu.get("Batch", ""))
            pay_dt = pd.to_datetime(stu.get("payment_date_recent", pd.NaT), errors="coerce")
            dates_row = find_student_dates_row(dates_df, student_name, email_key, student_key, program, batch)
            offered_dt = pd.to_datetime(dates_row.get("offered_date_parsed", pd.NaT), errors="coerce") if dates_row is not None else pd.NaT
            deadline_dt = pd.to_datetime(dates_row.get("deadline_parsed", pd.NaT), errors="coerce") if dates_row is not None else pd.NaT
            ev_df = collect_student_profile_events(data, email_key, student_key, student_name, pay_dt=pay_dt, offered_dt=offered_dt, deadline_dt=deadline_dt)
            if not ev_df.empty:
                ev_df = ev_df.sort_values(["event_date", "event_name"], na_position="last")
                ev_disp = ev_df[["event_date", "event_name", "event_type", "source_sheets"]].copy()
                ev_disp["event_date"] = ev_disp["event_date"].dt.strftime("%d-%b-%Y")
            else:
                ev_disp = pd.DataFrame(columns=["event_date", "event_name", "event_type", "source_sheets"])
            recent_payments.append({
                "student_name": student_name,
                "program": program,
                "batch": batch,
                "email_key": email_key,
                "payment_date": pay_dt,
                "events_df": ev_disp,
                "total_participation": int(ev_df["dedupe_key"].nunique()) if not ev_df.empty else 0,
            })

    return {
        "today": today,
        "start_date": start_date,
        "end_date": end_date,
        "batch_student_recent": batch_student_recent,
        "recent_events_df": recent_events_df,
        "recent_payments": recent_payments,
    }


def map_ugpg_unique_plot_event_type(event_type: str) -> str:
    s = clean_text(event_type).strip().lower()
    if s in {"hackathon", "competition", "competitions"}:
        return "Competitions"
    if s in {"fun", "fun task", "general", "poll"}:
        return "General/Fun"
    return clean_text(event_type) or "Other"




def map_retention_bucket_event_type(event_type: str) -> str:
    s = clean_text(event_type).strip().lower()
    if s in {"online event", "online ama", "online amas", "ama", "masterclass", "skill bootcamp", "bootcamp"}:
        return "Online Events & Masterclasses"
    if s in {"competition", "competitions", "hackathon", "hackerthon"}:
        return "Competitions & Hackathons"
    if s in {"general", "poll", "fun", "fun task"}:
        return "General/Fun"
    return clean_text(event_type) or "Other"


def student_unique_id_from_row(row) -> str:
    return clean_text(row.get("email_key", "")) or clean_text(row.get("student_key", "")) or normalize_name(row.get("student_name", ""))


def build_timeline_from_event_info(df: pd.DataFrame, event_info: pd.DataFrame) -> pd.DataFrame:
    """Build a date-wise timeline that counts unique attendees per date, with event names in hover."""
    if df is None or df.empty or event_info is None or event_info.empty:
        return pd.DataFrame(columns=["event_date", "Participants", "Event Names", "Event Types"])
    rows = []
    for _, ev in event_info.iterrows():
        col = ev.get("column_name")
        ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
        if not col or col not in df.columns or pd.isna(ev_date):
            continue
        attended_mask = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
        if not attended_mask.any():
            continue
        for _, stu in df.loc[attended_mask].iterrows():
            sid = student_unique_id_from_row(stu)
            if not sid:
                continue
            rows.append({
                "event_date": ev_date.normalize(),
                "student_id": sid,
                "event_name": clean_text(ev.get("event_name", "")) or clean_text(col),
                "event_type": clean_text(ev.get("event_type", "")) or "Other",
            })
    if not rows:
        return pd.DataFrame(columns=["event_date", "Participants", "Event Names", "Event Types"])
    long = pd.DataFrame(rows).drop_duplicates(subset=["event_date", "student_id", "event_name", "event_type"])
    out = long.groupby("event_date", as_index=False).agg(
        Participants=("student_id", "nunique"),
        **{
            "Event Names": ("event_name", lambda s: ", ".join(sorted(dict.fromkeys([clean_text(x) for x in s if clean_text(x)])))[:900]),
            "Event Types": ("event_type", lambda s: ", ".join(sorted(dict.fromkeys([clean_text(x) for x in s if clean_text(x)]))))
        }
    ).sort_values("event_date")
    return out


def build_event_attendance_table(df: pd.DataFrame, event_info: pd.DataFrame) -> pd.DataFrame:
    """Return an event-level attendance table for batch/Tetr-X detail pages."""
    cols = ["Event / Activity Name", "Event / Activity Date", "Event / Activity Type", "Attendance"]
    if df is None or df.empty or event_info is None or event_info.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for _, ev in event_info.iterrows():
        col = ev.get("column_name")
        if not col or col not in df.columns:
            continue
        event_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
        attendance = int(pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0).sum())
        rows.append({
            "Event / Activity Name": clean_text(ev.get("event_name", "")) or clean_text(col),
            "Event / Activity Date": event_date.normalize() if pd.notna(event_date) else pd.NaT,
            "Event / Activity Type": clean_text(ev.get("event_type", "")) or "Other",
            "Attendance": attendance,
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame(rows)
    out = out.sort_values("Event / Activity Date", ascending=False, na_position="last").reset_index(drop=True)
    return out

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


def exact_matching_col(df: pd.DataFrame, candidates):
    """Return the first column whose cleaned lowercase header exactly matches a candidate."""
    normalized = {clean_text(c).lower(): c for c in df.columns}
    for cand in candidates:
        key = clean_text(cand).lower()
        if key in normalized:
            return normalized[key]
    return None


def select_payment_date_col(df: pd.DataFrame, sheet_name: str):
    """Payment-date priority rules.

    For Tetr-X sheets, the first source of truth is fixed by sheet:
    - Tetr-X-UG: `Payment date (c3)`
    - Tetr-X-PG: `Payment date`

    Only if the exact source-of-truth column is absent do we fall back to the
    older generic payment/community-join detection so existing non-Tetr-X sheets
    keep working unchanged.
    """
    sheet = clean_text(sheet_name).lower()
    if sheet == "tetr-x-ug":
        col = exact_matching_col(df, ["Payment date (c3)"])
        if col:
            return col
        # last-resort fallback for minor header spacing/case variations
        for c in df.columns:
            low = clean_text(c).lower().replace(" ", "")
            if low == "paymentdate(c3)":
                return c
    elif sheet == "tetr-x-pg":
        col = exact_matching_col(df, ["Payment date"])
        if col:
            return col

    return best_matching_col(df, ["payment date", "date of payment", "paid date", "community join date"])


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


@st.cache_data(show_spinner=False, ttl=600)
def gsheets_get_sheet_names(spreadsheet_id: str):
    gc = _get_gsheets_client()
    sh = gc.open_by_key(spreadsheet_id)
    return [ws.title for ws in sh.worksheets()]


@st.cache_data(show_spinner=False, ttl=600)
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


def _values_to_raw_df(values):
    if not values:
        return pd.DataFrame()
    df = pd.DataFrame(values)
    df.replace("", np.nan, inplace=True)
    return df.dropna(how="all")


def _quote_sheet_range(sheet_name: str) -> str:
    # Use an A1 range that pulls the whole tab. Sheet names with spaces need quotes.
    safe = str(sheet_name).replace("'", "''")
    return f"'{safe}'"


@st.cache_data(show_spinner=False, ttl=600)
def gsheets_read_raw_sheets_batch(spreadsheet_id: str, sheet_names_tuple: tuple):
    """Read all needed Google Sheet tabs in one Sheets API batch request.

    This avoids one API read per worksheet and prevents Google Sheets 429
    read-quota errors while keeping the dashboard live via a short cache TTL.
    """
    sheet_names = list(sheet_names_tuple or [])
    if not sheet_names:
        return {}

    gc = _get_gsheets_client()
    sh = gc.open_by_key(spreadsheet_id)
    ranges = [_quote_sheet_range(s) for s in sheet_names]

    response = sh.values_batch_get(ranges=ranges)
    value_ranges = response.get("valueRanges", []) if isinstance(response, dict) else []

    out = {}
    for sheet_name, vr in zip(sheet_names, value_ranges):
        out[sheet_name] = _values_to_raw_df(vr.get("values", []))

    # If Google returns fewer ranges for any reason, keep the keys predictable.
    for sheet_name in sheet_names:
        out.setdefault(sheet_name, pd.DataFrame())
    return out


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
        return pd.DataFrame(columns=["student_name", "student_key", "email_key", "UG PG", "Batch", "Course", "Offered date", "Deadline", "offered_date_parsed", "deadline_parsed"])

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
    course_col = next((c for c in df.columns if clean_text(c).lower() == "course" or "course" in clean_text(c).lower()), None)

    if name_col is None and email_col is None:
        return pd.DataFrame(columns=["student_name", "student_key", "email_key", "UG PG", "Batch", "Course", "Offered date", "Deadline", "offered_date_parsed", "deadline_parsed"])

    df["student_name"] = df[name_col].map(clean_text) if name_col else ""
    df["student_key"] = df["student_name"].map(normalize_name)
    df["email_key"] = df[email_col].map(normalize_email) if email_col else ""
    df["UG PG"] = df[program_col].map(clean_text) if program_col else ""
    df["Batch"] = df[batch_col].map(normalize_batch_token) if batch_col else ""
    df["Course"] = df[course_col].map(clean_text) if course_col else ""
    df["Offered date"] = df[offered_col].map(clean_text) if offered_col else ""
    df["Deadline"] = df[deadline_col].map(clean_text) if deadline_col else ""
    df["offered_date_parsed"] = df["Offered date"].apply(parse_date_safe)
    df["deadline_parsed"] = df["Deadline"].apply(parse_date_safe)

    keep = ["student_name", "student_key", "email_key", "UG PG", "Batch", "Course", "Offered date", "Deadline", "offered_date_parsed", "deadline_parsed"]
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
            "amount_usd", "entry_type", "is_winner", "is_spotlight", "announcement_date"
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
    announcement_date_col = next((
        c for c in df.columns
        if ("date" in clean_text(c).lower() and ("winner" in clean_text(c).lower() or "announcement" in clean_text(c).lower()))
    ), None)

    if winner_name_col is None and email_col is None:
        return pd.DataFrame(columns=[
            "challenge_name", "winner_name", "student_key", "email_key", "batch_key",
            "amount_usd", "entry_type", "is_winner", "is_spotlight", "announcement_date"
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
    out["announcement_date"] = df[announcement_date_col].apply(parse_date_safe) if announcement_date_col else pd.NaT

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
    mobile_col = best_matching_col(df, ["mobile", "phone", "contact"])
    batch_col = best_matching_col(df, ["batch"])
    country_col = best_matching_col(df, ["country"])
    income_col = best_matching_col(df, ["income"])
    status_col = best_matching_col(df, ["status"])
    payment_col = best_matching_col(df, ["payment"])
    payment_date_col = best_matching_col(df, ["payment date", "date of payment", "paid date"])
    # Overview active-student source of truth: newly added master-sheet column.
    # It is intentionally detected before generic event-column parsing so it is not treated as an event.
    batch_engagement_pct_col = next((c for c in df.columns if clean_text(c).lower() == "engagement % (batch data)"), None)
    if batch_engagement_pct_col is None:
        batch_engagement_pct_col = best_matching_col(df, ["engagement % (batch data)", "batch data engagement %"])
    community_status_col = best_matching_col(df, ["tetr x/term 0 status", "tetr x term 0 status", "term 0 status", "community status", "admitted group"])
    # Overview Joined WA source of truth for Master UG / Master PG.
    # Keep the raw value because values like "Left" must count as Joined WA in Overview,
    # while the normalized community status intentionally maps them to Out elsewhere.
    admitted_group_batch_col = exact_matching_col(df, ["Admitted Group (Batch onwards)"]) or best_matching_col(df, ["admitted group (batch onwards)", "admitted group"])
    term_zero_col = best_matching_col(df, ["tetr x/term 0 status", "tetr x term 0 status", "term 0 status", "term zero group", "added to term 0"])

    if not name_col:
        return _empty_activity_sheet(sheet_name, f"Name column not found in {sheet_name}")

    df = df[df[name_col].apply(is_valid_student_name)].copy()
    df["Program"] = program
    df["Batch"] = df[batch_col].astype(str).str.strip() if batch_col else ""
    df["source_sheet"] = sheet_name
    df["student_name"] = df[name_col].map(clean_text)
    df["student_key"] = df["student_name"].map(normalize_name)
    df["email_key"] = df[email_col].map(normalize_email) if email_col else ""
    df["mobile_key"] = df[mobile_col].map(normalize_phone) if mobile_col else ""
    community_base = df[community_status_col].map(clean_text) if community_status_col else pd.Series("", index=df.index)
    df["admitted_group_batch_onwards_raw"] = df[admitted_group_batch_col].map(clean_text) if admitted_group_batch_col else community_base
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
    protected = {name_col, email_col, batch_col, country_col, income_col, status_col, payment_col, payment_date_col,
                 batch_engagement_pct_col, community_status_col,
                 "Program", "Batch", "source_sheet", "student_name", "student_key", "email_key", "mobile_key", "community_status_value",
                 "master_is_paid", "master_is_refunded", "master_status_value", "master_payment_value", "master_payment_date_parsed"}
    for col in df.columns:
        if col in protected:
            continue
        s = df[col].fillna("").astype(str).str.strip().str.lower()
        if len(s) and (s.isin({"yes", "no", "", "nan"}).mean() > 0.6):
            event_cols.append(col)
            df[col] = s.map(normalize_yes_no)

    df["participation_count_master"] = df[event_cols].sum(axis=1) if event_cols else 0
    # Active students in Overview now come from Engagement % (Batch Data) > 0 when that column exists.
    # Fallback keeps the older behavior for older workbooks.
    if batch_engagement_pct_col and batch_engagement_pct_col in df.columns:
        batch_pct = df[batch_engagement_pct_col].apply(parse_numeric_percent)
        if not batch_pct.dropna().empty and batch_pct.max(skipna=True) <= 1.05:
            batch_pct = batch_pct * 100
        df["engagement_batch_data_pct"] = batch_pct.fillna(0)
        df["active_master"] = df["engagement_batch_data_pct"].gt(0)
    else:
        df["engagement_batch_data_pct"] = np.nan
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
        "mobile_col": mobile_col,
        "batch_col": batch_col,
        "country_col": country_col,
        "income_col": income_col,
        "status_col": status_col,
        "payment_col": payment_col,
        "payment_date_col": payment_date_col,
        "batch_engagement_pct_col": batch_engagement_pct_col,
        "community_status_col": community_status_col,
        "admitted_group_batch_col": admitted_group_batch_col,
        "event_cols": event_cols,
    }
    return df, ctx



def _detect_activity_header_row(raw: pd.DataFrame, default_row: int = 5, max_scan: int = 25):
    """Detect the student header row in batch/Tetr-X sheets.
    Most activity sheets use row 6 as header (0-index 5), but newly added sheets
    can be blank, shifted, or partially formatted. This keeps the dashboard from
    failing while still using the standard row when valid.
    """
    if raw is None or raw.empty:
        return None

    def row_score(i):
        vals = [clean_text(v).lower() for v in raw.iloc[i].tolist()]
        joined = " | ".join(vals)
        score = 0
        # Strong identifiers for a real student-data header.
        if any(v in {"student name", "student names", "name", "full name"} for v in vals):
            score += 8
        if "student name" in joined or "student names" in joined:
            score += 10
        if any("email" in v for v in vals):
            score += 5
        if any("status" in v for v in vals):
            score += 3
        if any("country" in v for v in vals):
            score += 2
        if any("payment" in v for v in vals):
            score += 2
        if any("mobile" in v or "phone" in v or "contact" in v for v in vals):
            score += 2
        if sum(1 for v in vals if v) >= 4:
            score += 1
        return score

    if raw.shape[0] > default_row and row_score(default_row) >= 8:
        return default_row

    best_i, best_score = None, -1
    for i in range(min(max_scan, raw.shape[0])):
        sc = row_score(i)
        if sc > best_score:
            best_i, best_score = i, sc
    return best_i if best_score >= 8 else None


def _empty_activity_sheet(sheet_name: str, reason: str = ""):
    """Return an empty activity sheet with the standard columns/ctx so optional new
    sheets (for example newly-created blank PG B8) do not break the whole app.
    """
    base_cols = [
        "Program", "source_sheet", "student_name", "student_key", "email_key",
        "mobile_key", "Batch", "country", "income", "counsellor_name",
        "community_status_value", "participation_count", "engagement_score",
        "engagement_pct", "payment_date_parsed", "sheet_status_raw",
        "sheet_is_refunded", "sheet_is_paid", "is_active"
    ]
    df = pd.DataFrame(columns=base_cols)
    event_info = pd.DataFrame(columns=["column_name", "event_name", "event_type", "event_date", "sheet"])
    ctx = {
        "name_col": None,
        "email_col": None,
        "mobile_col": None,
        "batch_col": None,
        "country_col": None,
        "income_col": None,
        "counsellor_col": None,
        "counsellor1_col": None,
        "counsellor2_col": None,
        "community_status_col": None,
        "payment_status_col": None,
        "payment_date_col": None,
        "engagement_score_col": None,
        "engagement_pct_col": None,
        "event_info": event_info,
        "event_cols": [],
        "parse_warning": reason,
    }
    return df, ctx


def parse_activity_sheet(raw: pd.DataFrame, sheet_name: str):
    header_row = _detect_activity_header_row(raw, default_row=5, max_scan=30)
    if header_row is None:
        return _empty_activity_sheet(sheet_name, f"Could not detect student header row in {sheet_name}")
    data_start = header_row + 1
    if raw.shape[0] <= header_row:
        return _empty_activity_sheet(sheet_name, f"Sheet too short: {sheet_name}")

    # Event metadata is normally 5/4/3 rows above the student header.
    # This preserves the standard structure and supports shifted/new sheets.
    type_idx = max(0, header_row - 5)
    event_idx = max(0, header_row - 4)
    date_idx = max(0, header_row - 3)
    type_row = raw.iloc[type_idx].tolist() if raw.shape[0] > type_idx else []
    event_row = raw.iloc[event_idx].tolist() if raw.shape[0] > event_idx else []
    date_row = raw.iloc[date_idx].tolist() if raw.shape[0] > date_idx else []
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
    mobile_col = best_matching_col(df, ["mobile", "phone", "contact"])
    batch_col = best_matching_col(df, ["batch"])
    country_col = best_matching_col(df, ["country"])
    income_col = best_matching_col(df, ["income"])
    # Counsellor rules: UG usually uses Counsellor; PG uses Counsellor 1 unless it is Not required, then Counsellor 2.
    counsellor_col = best_matching_col(df, ["counsellor"])
    counsellor1_col = best_matching_col(df, ["counsellor 1", "counsellor1"])
    counsellor2_col = best_matching_col(df, ["counsellor 2", "counsellor2"])
    mobile_col = best_matching_col(df, ["mobile"])
    community_status_col = best_matching_col(df, ["tetr x/term 0 status", "tetr x term 0 status", "term 0 status", "community status", "admitted group"])
    term_zero_col = best_matching_col(df, ["tetr x/term 0 status", "tetr x term 0 status", "term 0 status", "term zero group", "added to term 0"])
    payment_status_col = best_matching_col(df, ["payment status", "status"])
    payment_date_col = select_payment_date_col(df, sheet_name)
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
    df["mobile_key"] = df[mobile_col].map(normalize_phone) if mobile_col else ""
    df["Batch"] = df[batch_col].map(clean_text) if batch_col else infer_batch_group_from_sheet_name(sheet_name)
    if counsellor1_col and counsellor1_col in df.columns:
        c1 = df[counsellor1_col].map(clean_text)
        c2 = df[counsellor2_col].map(clean_text) if counsellor2_col and counsellor2_col in df.columns else pd.Series("", index=df.index)
        df["counsellor_name"] = c1.where(~c1.str.lower().eq("not required"), c2)
    elif counsellor_col and counsellor_col in df.columns:
        df["counsellor_name"] = df[counsellor_col].map(clean_text)
    else:
        df["counsellor_name"] = ""
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
        "mobile_col": mobile_col,
        "batch_col": batch_col,
        "country_col": country_col,
        "income_col": income_col,
        "counsellor_col": counsellor_col,
        "counsellor1_col": counsellor1_col,
        "counsellor2_col": counsellor2_col,
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


@st.cache_data(show_spinner=False, ttl=600)
def load_dashboard_data(source_mode: str, spreadsheet_id=None, file_bytes=None):
    sheet_names = get_sheet_names(source_mode, spreadsheet_id, file_bytes)
    missing = [s for s in ALL_REQUIRED if s not in sheet_names]

    masters, master_ctx = {}, {}
    activities, activity_ctx = {}, {}
    dates_df = pd.DataFrame()
    winner_df = pd.DataFrame()

    # Google Sheets quota fix: read all required tabs in a single values.batchGet
    # call and reuse these raw frames for the whole dashboard build.
    sheets_to_read = [
        s for s in (MASTER_SHEETS + UG_BATCH_SHEETS + PG_BATCH_SHEETS + TX_SHEETS + [DATES_SHEET, WINNER_SHEET])
        if s in sheet_names
    ]
    if source_mode == "gsheets":
        raw_cache = gsheets_read_raw_sheets_batch(spreadsheet_id, tuple(sheets_to_read))
    else:
        raw_cache = {s: load_raw_sheet(source_mode, s, spreadsheet_id, file_bytes) for s in sheets_to_read}

    for sheet in MASTER_SHEETS:
        if sheet in raw_cache:
            raw = raw_cache.get(sheet, pd.DataFrame())
            if not raw.empty:
                masters[sheet], master_ctx[sheet] = parse_master_sheet(raw, "UG" if sheet.endswith("UG") else "PG", sheet)

    for sheet in UG_BATCH_SHEETS + PG_BATCH_SHEETS + TX_SHEETS:
        if sheet in raw_cache:
            raw = raw_cache.get(sheet, pd.DataFrame())
            if not raw.empty:
                try:
                    activities[sheet], activity_ctx[sheet] = parse_activity_sheet(raw, sheet)
                except Exception as e:
                    # Do not fail the full dashboard because one newly-added/blank batch sheet
                    # is not formatted yet. The sheet will simply render empty until its header
                    # follows the expected student-data layout.
                    activities[sheet], activity_ctx[sheet] = _empty_activity_sheet(sheet, str(e))

    if DATES_SHEET in raw_cache and not raw_cache[DATES_SHEET].empty:
        dates_df = parse_dates_sheet(raw_cache[DATES_SHEET])

    if WINNER_SHEET in raw_cache and not raw_cache[WINNER_SHEET].empty:
        winner_df = parse_winner_sheet(raw_cache[WINNER_SHEET])

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
          {('<span style="position:relative; width:12px; height:12px; display:inline-block;"><span style="position:absolute; inset:0; border-radius:50%; background:rgba(27,181,92,0.30); animation:heartbeatPing 1.5s ease-out infinite;"></span><span style="position:absolute; inset:0; border-radius:50%; background:#1bb55c;"></span></span>' if connected_ok else '<span style="width:12px; height:12px; border-radius:50%; background:' + status_dot + '; display:inline-block;"></span>')}
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

    # Overview-only paid rule, based on the source master sheet:
    # - Master UG: Admitted OR any status containing Deferral
    # - Master PG: Admitted OR status containing Admitted:Deferral / Admitted: Deferral
    # Refund rows are excluded from Paid / Admitted.
    overview_df = overview_df.copy()
    _status_for_overview = overview_df.get("resolved_status", overview_df.get("master_status_value", pd.Series("", index=overview_df.index))).astype(str).str.lower()
    _program_for_overview = overview_df.get("Program", pd.Series("", index=overview_df.index)).astype(str).str.upper().str.strip()
    _refund_for_overview = overview_df.get("is_refunded", pd.Series(False, index=overview_df.index)).fillna(False).astype(bool) | _status_for_overview.str.contains("refund", na=False)
    _is_admitted_exact = _status_for_overview.str.strip().eq("admitted")
    _is_ug_deferral = _program_for_overview.eq("UG") & _status_for_overview.str.contains("deferral", na=False)
    _is_pg_admitted_deferral = _program_for_overview.eq("PG") & _status_for_overview.str.contains(r"admitted\s*:\s*deferral", na=False, regex=True)
    overview_df["is_refunded"] = _refund_for_overview
    overview_df["is_paid"] = (_is_admitted_exact | _is_ug_deferral | _is_pg_admitted_deferral) & (~_refund_for_overview)
    overview_df["status_bucket"] = np.select(
        [overview_df["is_refunded"], overview_df["is_paid"]],
        ["Refunded", "Paid / Admitted"],
        default="Not Paid",
    )

    total_students, total_active, total_paid, total_refunded, ug_students, pg_students, ug_paid, pg_paid, ug_refunded, pg_refunded = overview_metrics(overview_df)
    # Overview-only Joined WA rule: use Master UG / Master PG `Admitted Group (Batch onwards)`.
    # Count In, TetrX, Tetr X, Added to Term 0, and Left irrespective of uppercase/lowercase.
    _overview_comm_series = overview_df.get(
        "admitted_group_batch_onwards_raw",
        overview_df.get("community_status_value", pd.Series("", index=overview_df.index))
    ).astype(str)
    def _overview_is_joined_wa_value(x):
        s = clean_text(x).strip().lower().replace("-", " ")
        s = " ".join(s.split())
        return s in {"in", "tetrx", "tetr x", "added to term 0", "left"}
    _overview_joined_mask = _overview_comm_series.map(_overview_is_joined_wa_value).fillna(False).astype(bool)
    joined_wa = int(_overview_joined_mask.sum()) if not overview_df.empty else 0
    active_joined = int((overview_df.get("is_active", pd.Series(False, index=overview_df.index)).fillna(False).astype(bool) & _overview_joined_mask).sum()) if not overview_df.empty else 0
    joined_rate = round((joined_wa / total_students * 100), 1) if total_students else 0
    active_rate = round((total_active / total_students * 100), 1) if total_students else 0
    active_joined_rate = round((active_joined / joined_wa * 100), 1) if joined_wa else 0
    paid_rate = round((total_paid / total_students * 100), 1) if total_students else 0
    refunded_rate = round((total_refunded / total_students * 100), 1) if total_students else 0

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total Offered Students", f"{total_students:,}")
    m2.metric("Joined WA", f"{joined_wa:,}", delta=f"{joined_rate}% of Total Offered")
    m3.metric("Active Students", f"{total_active:,}", delta=f"{active_rate}% of Total | {active_joined_rate}% of Joined WA")
    m4.metric("Paid / Admitted", f"{total_paid:,}", delta=f"{paid_rate}% paid")
    m5.metric("Refunded", f"{total_refunded:,}", delta=f"{refunded_rate}% refunded")
    m6.metric("UG vs PG", f"{ug_students:,} / {pg_students:,}")

    g1, g2, g3 = st.columns([1.2, 1, 1])
    with g1:
        st.plotly_chart(gauge_chart(total_students, "Total Offered Students", maximum=max(total_students, 1)), use_container_width=True, key="overview_gauge")
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





def build_tetrx_payment_lookup_for_paid_details(data):
    """Tetr-X source-of-truth payment-date lookup for paid student detail tables."""
    rows = []
    activities = data.get("activities", {}) if isinstance(data, dict) else {}
    for sheet in TX_SHEETS:
        tx = activities.get(sheet, pd.DataFrame())
        if tx is None or tx.empty:
            continue
        program = "UG" if sheet == "Tetr-X-UG" else "PG" if sheet == "Tetr-X-PG" else ""
        status = tx.get("sheet_status_raw", pd.Series("", index=tx.index)).astype(str).map(clean_text).str.lower()
        pay = pd.to_datetime(tx.get("payment_date_parsed", pd.NaT), errors="coerce")
        frame = pd.DataFrame({
            "program": program,
            "email_key": tx.get("email_key", ""),
            "student_key": tx.get("student_key", ""),
            "student_name": tx.get("student_name", ""),
            "payment_date": pay,
            "is_admitted": status.eq("admitted"),
        })
        frame["email_key"] = frame["email_key"].astype(str).map(clean_text)
        frame["student_key"] = frame["student_key"].astype(str).map(clean_text)
        frame["student_name_key"] = frame["student_name"].astype(str).map(normalize_name)
        frame = frame[frame["is_admitted"] & frame["payment_date"].notna()].copy()
        if not frame.empty:
            rows.append(frame)

    if not rows:
        return {"by_email": {}, "by_name": {}, "by_email_any": {}, "by_name_any": {}}

    lookup_df = pd.concat(rows, ignore_index=True).sort_values("payment_date")
    by_email, by_name, by_email_any, by_name_any = {}, {}, {}, {}
    for _, r in lookup_df.iterrows():
        prog = clean_text(r.get("program", ""))
        email = clean_text(r.get("email_key", ""))
        name = clean_text(r.get("student_key", "")) or clean_text(r.get("student_name_key", ""))
        dt = pd.to_datetime(r.get("payment_date", pd.NaT), errors="coerce")
        if pd.isna(dt):
            continue
        if email:
            by_email.setdefault((prog, email), dt)
            by_email_any.setdefault(email, dt)
        if name:
            by_name.setdefault((prog, name), dt)
            by_name_any.setdefault(name, dt)
    return {"by_email": by_email, "by_name": by_name, "by_email_any": by_email_any, "by_name_any": by_name_any}


def resolve_tetrx_payment_date_for_row(row, lookup):
    program = clean_text(row.get("Program", ""))
    email = clean_text(row.get("email_key", ""))
    name = clean_text(row.get("student_key", "")) or normalize_name(row.get("student_name", ""))
    candidates = [
        ((program, email), lookup.get("by_email", {})),
        ((program, name), lookup.get("by_name", {})),
        (email, lookup.get("by_email_any", {})),
        (name, lookup.get("by_name_any", {})),
    ]
    for key, dictionary in candidates:
        if key and key in dictionary:
            return dictionary[key]
    return pd.NaT


def render_paid_students_section(df, ctx, prefix, data=None):
    """Show admitted/paid students for UG/PG batch sections without affecting other pages."""
    if df is None or df.empty or "sheet_is_paid" not in df.columns:
        return

    paid_df = df[df["sheet_is_paid"].fillna(False).astype(bool)].copy()
    if paid_df.empty:
        st.markdown("#### Paid / Admitted Students")
        st.info("No paid/admitted students found in this sheet.")
        return

    st.markdown("#### Paid / Admitted Students")

    # Prefer payment dates from the Tetr-X source-of-truth sheets for this paid-student table.
    # UG uses Tetr-X-UG -> Payment date (c3); PG uses Tetr-X-PG -> Payment date.
    if data is not None:
        tx_lookup = build_tetrx_payment_lookup_for_paid_details(data)
        paid_df["tx_payment_date_for_paid_details"] = paid_df.apply(lambda r: resolve_tetrx_payment_date_for_row(r, tx_lookup), axis=1)

    # Compact batch summary first, useful for All UG / All PG combined tabs.
    if "Batch" in paid_df.columns:
        batch_summary = (
            paid_df.groupby("Batch", dropna=False)["student_name"]
            .nunique()
            .reset_index(name="Paid Students")
            .sort_values(["Batch", "Paid Students"], ascending=[True, False])
        )
        batch_summary["Batch"] = batch_summary["Batch"].replace("", "Unknown")
        st.dataframe(batch_summary, use_container_width=True, height=min(260, 70 + len(batch_summary) * 35), key=f"{prefix}_paid_batch_summary")

    # Build a clean detail table from available normalized fields.
    email_col = ctx.get("email_col") if isinstance(ctx, dict) else None
    country_col = ctx.get("country_col") if isinstance(ctx, dict) else None
    income_col = ctx.get("income_col") if isinstance(ctx, dict) else None
    payment_date_col = ctx.get("payment_date_col") if isinstance(ctx, dict) else None

    # Combined sections may not carry ctx columns, so infer safe fallbacks.
    if not email_col or email_col not in paid_df.columns:
        email_col = best_matching_col(paid_df, ["email"])
    if not country_col or country_col not in paid_df.columns:
        country_col = best_matching_col(paid_df, ["country"])
    if not income_col or income_col not in paid_df.columns:
        income_col = best_matching_col(paid_df, ["income", "household income"])
    if "tx_payment_date_for_paid_details" in paid_df.columns and pd.to_datetime(paid_df["tx_payment_date_for_paid_details"], errors="coerce").notna().any():
        payment_date_col = "tx_payment_date_for_paid_details"
    elif not payment_date_col or payment_date_col not in paid_df.columns:
        payment_date_col = "payment_date_parsed" if "payment_date_parsed" in paid_df.columns else None

    detail_cols = []
    rename_map = {}
    for col, label in [
        ("student_name", "Student Name"),
        (email_col, "Email"),
        ("Program", "UG/PG"),
        ("Batch", "Batch"),
        (country_col, "Country"),
        (income_col, "Income"),
        ("sheet_status_raw", "Status"),
        (payment_date_col, "Payment Date"),
        ("community_status_value", "Community Status"),
        ("participation_count", "Total Participations"),
        ("engagement_pct", "Engagement %"),
    ]:
        if col and col in paid_df.columns and col not in detail_cols:
            detail_cols.append(col)
            rename_map[col] = label

    details = paid_df[detail_cols].copy() if detail_cols else paid_df.copy()
    for dt_col in ["payment_date_parsed", "tx_payment_date_for_paid_details"]:
        if dt_col in details.columns:
            details[dt_col] = pd.to_datetime(details[dt_col], errors="coerce").dt.date.astype(str).replace("NaT", "")
    if "engagement_pct" in details.columns:
        details["engagement_pct"] = pd.to_numeric(details["engagement_pct"], errors="coerce").round(1)
    details = details.rename(columns=rename_map)
    if "Payment Date" in details.columns:
        details["Payment Date"] = details["Payment Date"].replace({"NaT": "", "nan": "", "None": ""})

    sort_cols = [c for c in ["Batch", "Student Name"] if c in details.columns]
    if sort_cols:
        details = details.sort_values(sort_cols)
    st.dataframe(details, use_container_width=True, height=420, key=f"{prefix}_paid_students_detail")

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
            fig = px.bar(event_counts, x="Participants", y="event_name", orientation="h", color="event_type", title="Top Events by Participation", hover_name="event_name", hover_data={"event_name": False, "event_type": True, "event_date": True, "Participants": True})
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
        top_cols = ["student_name"]
        if prefix.startswith("course_") and "Batch" in df.columns:
            top_cols.append("Batch")
        top_cols += ["engagement_pct", "engagement_score", "community_status_value"]
        top_cols = [c for c in top_cols if c in df.columns]
        students = df[top_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
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
            target_cols = ["student_name"]
            if prefix.startswith("course_") and "Batch" in df.columns:
                target_cols.append("Batch")
            target_cols += ["engagement_pct", "engagement_score", "community_status_value"]
            target_cols = [c for c in target_cols if c in df.columns]
            target = df[(~df["sheet_is_paid"]) & (~df["sheet_is_refunded"]) & (df["is_active"])][target_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
            st.markdown("#### Best Upgrade Targets")
            st.dataframe(target, use_container_width=True, height=390, key=f"{prefix}_upgrade_df")

    # UG/PG batch pages: show all paid/admitted students and their details below Top Students & Best Upgrade Targets.
    if not prefix.startswith("tx_"):
        render_paid_students_section(df, ctx, prefix, data=data)

    if not event_info.empty and event_info["event_date"].notna().any():
        timeline = build_timeline_from_event_info(df, event_info)
        if not timeline.empty:
            fig = px.line(
                timeline,
                x="event_date",
                y="Participants",
                markers=True,
                title="Participation Timeline",
                hover_name="Event Names",
                hover_data={"Event Names": False, "Event Types": True, "event_date": True, "Participants": True},
            )
            fig.update_traces(line_color=GREEN, marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{prefix}_timeline")
            event_table = build_event_attendance_table(df, event_info)
            if not event_table.empty:
                event_table_display = event_table.copy()
                event_table_display["Event / Activity Date"] = pd.to_datetime(event_table_display["Event / Activity Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("")
                st.markdown("##### Event / Activity Attendance Table")
                st.dataframe(event_table_display, use_container_width=True, hide_index=True, height=320, key=f"{prefix}_event_attendance_table")



def resolve_student_profile_payment_date(data, email_key: str, student_key: str, student_name: str, mobile_key: str = ""):
    """Resolve payment date for Student Profile directly from Tetr-X sheets.

    Student Profile metrics must work for every Tetr-X paid-status row that can
    carry a payment date: Admitted, Deferral / Admitted:Deferral, and Refunded.
    This helper is intentionally used only by Student Profile so no other
    dashboard sections are affected.
    """
    target_name = student_key or normalize_name(student_name)
    target_mobile = normalize_phone(mobile_key)
    found_dates = []
    paid_status_keywords = ("admitted", "deferral", "refund")
    for sheet in TX_SHEETS:
        tx = data.get("activities", {}).get(sheet, pd.DataFrame())
        if tx is None or tx.empty:
            continue
        mask = pd.Series(False, index=tx.index)
        if email_key and "email_key" in tx.columns:
            mask = mask | tx["email_key"].astype(str).eq(email_key)
        if target_name and "student_key" in tx.columns:
            mask = mask | tx["student_key"].astype(str).eq(target_name)
        if target_mobile and "mobile_key" in tx.columns:
            mask = mask | tx["mobile_key"].astype(str).eq(target_mobile)
        part = tx.loc[mask].copy()
        if part.empty:
            continue

        # Keep all rows that represent a paid/payment-status record. This includes
        # Admitted, Deferral / Admitted:Deferral and Refunded rows. If the status
        # column is missing, keep the matched row and rely on the payment-date value.
        if "sheet_status_raw" in part.columns:
            status_low = part["sheet_status_raw"].astype(str).str.lower()
            status_mask = status_low.apply(lambda v: any(k in v for k in paid_status_keywords))
            if status_mask.any():
                part = part.loc[status_mask].copy()

        for c in ["payment_date_parsed", "tx_payment_date", "resolved_payment_date", "Payment date (c3)", "Payment date", "Payment Date"]:
            if c in part.columns:
                vals = pd.to_datetime(part[c], errors="coerce").dropna()
                if not vals.empty:
                    found_dates.extend(vals.tolist())
    if found_dates:
        return min(found_dates)
    return pd.NaT


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
        ev_df["in_t7"] = delta.between(-7, 0, inclusive="both")
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

    base_cols = [c for c in ["student_name", "email_key", "student_key", "mobile_key", "Program", "Batch"] if c in overview_df.columns]
    students = overview_df[base_cols].drop_duplicates().sort_values("student_name")

    search_tab_name, search_tab_contact = st.tabs(["Search by Name", "Search by Email / Phone"])
    selected = []
    pasted_names = []
    contact_matched_names = []

    with search_tab_name:
        search = st.text_input("Search student names", value="", placeholder="Type a name...")
        options = students[students["student_name"].str.contains(search, case=False, na=False)]["student_name"].tolist() if search else students["student_name"].tolist()
        selected = st.multiselect("Select one or more students", options=options, default=[])
        pasted = st.text_area("Or paste multiple student names (one per line)")
        pasted_names = [clean_text(x) for x in pasted.splitlines() if clean_text(x)]

    with search_tab_contact:
        contact_query = st.text_area("Search by email or phone number", placeholder="Paste one or more emails / phone numbers, one per line")
        tokens = [clean_text(x) for x in re.split(r"[\n,;]+", contact_query) if clean_text(x)]
        if tokens:
            c_mask = pd.Series(False, index=students.index)
            for tok in tokens:
                email_tok = normalize_email(tok)
                phone_tok = normalize_phone(tok)
                if email_tok and "email_key" in students.columns:
                    c_mask = c_mask | students["email_key"].astype(str).str.lower().eq(email_tok)
                if phone_tok and "mobile_key" in students.columns:
                    c_mask = c_mask | students["mobile_key"].astype(str).str.endswith(phone_tok[-10:] if len(phone_tok) >= 10 else phone_tok)
            contact_matches = students.loc[c_mask].copy()
            if contact_matches.empty:
                st.info("No student matched the entered email / phone.")
            else:
                st.dataframe(contact_matches[[c for c in ["student_name", "email_key", "mobile_key", "Program", "Batch"] if c in contact_matches.columns]], use_container_width=True, hide_index=True)
                contact_matched_names = contact_matches["student_name"].dropna().astype(str).tolist()

    final_names = list(dict.fromkeys(selected + pasted_names + contact_matched_names))

    if not final_names:
        st.info("Search and select a student to view the profile.")
        return

    # Speed fix: do not build the full T-7/T+7 dataset for every Student Profile view.
    # T-7/T+7 counts for selected students are computed from that student's deduped event list below.
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

        # Speed fix: use the prebuilt profile table instead of scanning every sheet dataframe on every profile render.
        profile_source_df = data.get("profile_df", pd.DataFrame())
        if profile_source_df is not None and not profile_source_df.empty:
            pmask = pd.Series(False, index=profile_source_df.index)
            if email_key and "email_key" in profile_source_df.columns:
                pmask = pmask | profile_source_df["email_key"].astype(str).eq(email_key)
            if name_key and "student_key" in profile_source_df.columns:
                pmask = pmask | profile_source_df["student_key"].astype(str).eq(name_key)
            related_df = profile_source_df.loc[pmask].copy()
            if "profile_source" in related_df.columns:
                related_df = related_df[related_df["profile_source"].astype(str).ne("master")].copy()
        else:
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

        # Student Profile payment date must come from Tetr-X for every paid/payment status
        # that has a payment date: Admitted, Deferral / Admitted:Deferral and Refunded.
        # Prefer the matched Tetr-X row first, then fall back to the master-resolved value.
        tx_profile_pay = resolve_student_profile_payment_date(
            data,
            email_key,
            name_key,
            master.get("student_name", ""),
            master.get("mobile_key", ""),
        )
        if pd.notna(tx_profile_pay):
            pay_dt = pd.to_datetime(tx_profile_pay, errors="coerce")
        else:
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
        course_val = clean_text(dates_row.get("Course", "")) if dates_row is not None else ""

        profile_event_df = collect_student_profile_events(data, email_key, name_key, master.get("student_name", ""), pay_dt=pay_dt, offered_dt=offered_dt, deadline_dt=deadline_dt)

        # Use this student's already-deduped events for T-window metrics instead of rebuilding all admitted students.
        stu_window = pd.DataFrame()
        if not profile_event_df.empty:
            trows = []
            if "in_t7" in profile_event_df.columns:
                part = profile_event_df[profile_event_df["in_t7"]].copy()
                if not part.empty:
                    part["window"] = "T-7 to T"
                    trows.append(part)
            if "in_tplus7" in profile_event_df.columns:
                part = profile_event_df[profile_event_df["in_tplus7"]].copy()
                if not part.empty:
                    part["window"] = "T+1 to T+7"
                    trows.append(part)
            if trows:
                stu_window = pd.concat(trows, ignore_index=True)

        st.markdown("---")
        st.markdown(f"### {master['student_name']}")

        p1, p2, p3, p4, p5, p6, p7, p8 = st.columns([0.75, 0.75, 1.05, 1.05, 1.05, 1.05, 1.05, 1.1])
        p1.metric("Program", clean_text(master.get("Program", "")))
        p2.metric("Batch", clean_text(master.get("Batch", "")))
        p3.metric("Course", course_val if course_val else "—")
        p4.metric("Paid Status", clean_text(master.get("resolved_status", "Not Paid")))
        p5.metric("Payment Date", format_date_display(pay_dt))
        p6.metric("Offered Date", format_date_display(offered_dt))
        p7.metric("Deadline", format_date_display(deadline_dt))
        p8.metric("Community Status", batch_comm if batch_comm else "—")

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
                "Course": course_val if course_val else "",
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
        profile_has_payment_date = pd.notna(pay_dt)
        with c2:
            stat_row_1 = st.columns(3)
            stat_row_1[0].metric("Total Participation", total_events)
            stat_row_1[1].metric("First 30 Days", first30_count)
            stat_row_1[2].metric("After Payment", after_paid_count if profile_has_payment_date else 0)

            stat_row_2 = st.columns(2)
            stat_row_2[0].metric("T-7 Count", t7_count if profile_has_payment_date else 0)
            stat_row_2[1].metric("T+7 Count", tp7_count if profile_has_payment_date else 0)

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

    frames = []
    event_infos = []
    country_col = None

    for s in available:
        src_df = data["activities"][s].copy()
        ctx = data.get("activity_ctx", {}).get(s, {})
        ei = ctx.get("event_info", pd.DataFrame())
        rename_map = {}
        if ei is not None and not ei.empty:
            ei = ei.copy()
            for idx, row in ei.iterrows():
                old_col = row.get("column_name")
                if old_col in src_df.columns:
                    new_col = f"{normalize_name(s)}__{old_col}"
                    rename_map[old_col] = new_col
                    ei.at[idx, "column_name"] = new_col
            event_infos.append(ei)
        if rename_map:
            src_df = src_df.rename(columns=rename_map)
        frames.append(src_df)
        if country_col is None and ctx.get("country_col"):
            country_col = ctx.get("country_col")

    combined_df = pd.concat(frames, ignore_index=True, sort=False)
    if event_infos:
        combined_event_info = pd.concat(event_infos, ignore_index=True, sort=False)
        combined_event_info = combined_event_info.drop_duplicates(subset=["column_name", "event_name", "event_type", "event_date", "sheet"]).reset_index(drop=True)
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



def _course_match_mask(course_series, course_label):
    """Match a course label from Dates -> Course without letting BMT match IBMT."""
    token = clean_text(course_label).upper()
    s = course_series.astype(str).map(clean_text).str.upper()
    # Use token boundaries so BMT does not match IBMT. Keep flexible for values like "BFAI UG".
    if token:
        pattern = rf"(^|[^A-Z0-9]){re.escape(token)}([^A-Z0-9]|$)"
        return s.str.contains(pattern, regex=True, na=False)
    return pd.Series(False, index=course_series.index)


def _course_dates_frame(data, course_label):
    """Return Dates-sheet rows for a UG course. Used only for displaying course student details."""
    dates = data.get("dates_df", pd.DataFrame())
    if dates is None or dates.empty or "Course" not in dates.columns:
        return pd.DataFrame()
    d = dates.copy()
    course_mask = _course_match_mask(d["Course"], course_label)
    if "UG PG" in d.columns:
        ug_mask = d["UG PG"].astype(str).str.upper().str.contains("UG", na=False)
    elif "UG/PG" in d.columns:
        ug_mask = d["UG/PG"].astype(str).str.upper().str.contains("UG", na=False)
    else:
        ug_mask = pd.Series(True, index=d.index)
    d = d.loc[course_mask & ug_mask].copy()
    if d.empty:
        return d
    # Keep one display row per student using email first, then name+batch.
    if "email_key" not in d.columns:
        email_col = next((c for c in d.columns if "email" in clean_text(c).lower()), None)
        d["email_key"] = d[email_col].map(normalize_email) if email_col else ""
    if "student_key" not in d.columns:
        name_col = next((c for c in d.columns if "name" in clean_text(c).lower()), None)
        d["student_key"] = d[name_col].map(normalize_name) if name_col else ""
    if "Batch" not in d.columns:
        d["Batch"] = ""
    d["_course_display_key"] = np.where(
        d["email_key"].astype(str).str.len() > 3,
        "e:" + d["email_key"].astype(str),
        "n:" + d["student_key"].astype(str) + "|" + d["Batch"].astype(str).map(clean_text),
    )
    return d.sort_values(["Batch", "student_name" if "student_name" in d.columns else "_course_display_key"]).drop_duplicates("_course_display_key", keep="first").reset_index(drop=True)


def _course_student_keys(data, course_label):
    """Return normalized email/name keys for UG students in Dates sheet matching the Course value."""
    d = _course_dates_frame(data, course_label)
    if d is None or d.empty:
        return set(), set()
    emails = set(d.get("email_key", pd.Series(dtype=object)).astype(str).map(clean_text).replace("", np.nan).dropna().tolist())
    names = set(d.get("student_key", pd.Series(dtype=object)).astype(str).map(clean_text).replace("", np.nan).dropna().tolist())
    return emails, names


def _render_course_student_details(data, course_label, prefix):
    """Show all Dates-sheet student details for a course above the paid/admitted section."""
    d = _course_dates_frame(data, course_label)
    st.markdown("#### All Student Details of this Course")
    if d is None or d.empty:
        st.info(f"No {course_label} UG students found in the Dates sheet.")
        return
    # Prefer common Dates columns, but keep whatever is available.
    preferred = [
        "Name", "student_name", "Email", "email", "Email id", "email_key",
        "UG/PG", "UG PG", "Batch", "Course", "Offered date", "Deadline",
        "offered_date_parsed", "deadline_parsed",
    ]
    cols = []
    seen = set()
    for c in preferred:
        if c in d.columns and c not in seen:
            cols.append(c); seen.add(c)
    # Add a few useful remaining columns without making the table too wide.
    for c in d.columns:
        lc = clean_text(c).lower()
        if c in seen or c.startswith("_"):
            continue
        if any(k in lc for k in ["name", "email", "batch", "course", "offer", "deadline", "ug", "pg"]):
            cols.append(c); seen.add(c)
    if not cols:
        cols = [c for c in d.columns if not c.startswith("_")]
    display = d[cols].copy()
    for c in display.columns:
        if pd.api.types.is_datetime64_any_dtype(display[c]):
            display[c] = pd.to_datetime(display[c], errors="coerce").dt.strftime("%d-%b-%Y").fillna("")
    st.caption(f"Unique students found in Dates sheet for {course_label} UG: {len(display):,}")
    st.dataframe(display, use_container_width=True, hide_index=True, height=320, key=f"{prefix}_course_all_student_details")


def build_course_activity_context(course_label, data):
    """Build a combined UG-batch activity view filtered to students from a Dates-sheet Course.
    This restores the earlier course-count behavior: count the matched UG batch activity rows,
    while the all-course student details table is shown separately from Dates.
    """
    emails, names = _course_student_keys(data, course_label)
    if not emails and not names:
        return pd.DataFrame(), {"event_info": pd.DataFrame(columns=["column_name", "event_name", "event_type", "event_date", "sheet"]), "country_col": None}

    frames = []
    event_infos = []
    country_col = None
    for s in [x for x in UG_BATCH_SHEETS if x in data.get("activities", {}) and x in data.get("activity_ctx", {})]:
        src_df = data["activities"][s].copy()
        if src_df.empty:
            continue
        mask = pd.Series(False, index=src_df.index)
        if emails and "email_key" in src_df.columns:
            mask = mask | src_df["email_key"].astype(str).isin(emails)
        if names and "student_key" in src_df.columns:
            mask = mask | src_df["student_key"].astype(str).isin(names)
        src_df = src_df.loc[mask].copy()
        if src_df.empty:
            continue

        # Courses section only: ensure a visible Batch column is always present
        # in Top Students / Best Upgrade Targets. Some UG batch sheets either
        # do not carry a Batch column or store it as B7/7/blank, so use the
        # source sheet name as the reliable batch label and keep it in the
        # displayed Batch column (e.g. "UG B7").
        inferred_batch_label = infer_batch_group_from_sheet_name(s)
        if "Batch" not in src_df.columns:
            src_df["Batch"] = inferred_batch_label
        else:
            src_df["Batch"] = src_df["Batch"].map(lambda x: _display_batch_label(x) or inferred_batch_label)
            src_df.loc[src_df["Batch"].astype(str).str.strip().eq(""), "Batch"] = inferred_batch_label
        src_df["Batch"] = src_df["Batch"].map(lambda x: _display_batch_label(x) if clean_text(x) != inferred_batch_label else inferred_batch_label)

        ctx = data.get("activity_ctx", {}).get(s, {})
        ei = ctx.get("event_info", pd.DataFrame())
        rename_map = {}
        if ei is not None and not ei.empty:
            ei = ei.copy()
            for idx, row in ei.iterrows():
                old_col = row.get("column_name")
                if old_col in src_df.columns:
                    new_col = f"{normalize_name(s)}__{old_col}"
                    rename_map[old_col] = new_col
                    ei.at[idx, "column_name"] = new_col
            event_infos.append(ei)
        if rename_map:
            src_df = src_df.rename(columns=rename_map)
        frames.append(src_df)
        if country_col is None and ctx.get("country_col"):
            country_col = ctx.get("country_col")

    if not frames:
        return pd.DataFrame(), {"event_info": pd.DataFrame(columns=["column_name", "event_name", "event_type", "event_date", "sheet"]), "country_col": None}
    combined_df = pd.concat(frames, ignore_index=True, sort=False)
    combined_df = combined_df.drop_duplicates(subset=[c for c in ["email_key", "student_key", "source_sheet"] if c in combined_df.columns], keep="first")
    combined_event_info = pd.concat(event_infos, ignore_index=True, sort=False) if event_infos else pd.DataFrame(columns=["column_name", "event_name", "event_type", "event_date", "sheet"])
    if not combined_event_info.empty:
        combined_event_info = combined_event_info.drop_duplicates(subset=["column_name", "event_name", "event_type", "event_date", "sheet"]).reset_index(drop=True)
    return combined_df, {"event_info": combined_event_info, "country_col": country_col}




def _display_batch_label(batch_val):
    b = clean_text(batch_val)
    if not b:
        return ""
    if re.fullmatch(r"\d+", b):
        return f"UG B{b}"
    if re.fullmatch(r"B\d+", b, flags=re.IGNORECASE):
        return f"UG {b.upper()}"
    if b.upper().startswith("UG"):
        return b
    return b


def build_course_event_attendance_table(data, course_label, course_df, course_ctx):
    """Course-only event table with merged same-name+date events, eligible course students, active batches and batchwise attendance."""
    base_cols = [
        "Event / Activity Name", "Event / Activity Date", "Event / Activity Type", "Attendance",
        f"Eligible {course_label} Students", "Active Batches", "Batchwise Attended Students",
    ]
    if course_df is None or course_df.empty or course_ctx is None:
        return pd.DataFrame(columns=base_cols)
    event_info = course_ctx.get("event_info", pd.DataFrame())
    if event_info is None or event_info.empty:
        return pd.DataFrame(columns=base_cols)

    course_dates = _course_dates_frame(data, course_label)
    if course_dates is None:
        course_dates = pd.DataFrame()
    if not course_dates.empty:
        cd = course_dates.copy()
        if "Batch" not in cd.columns:
            cd["Batch"] = ""
        cd["Batch Display"] = cd["Batch"].map(_display_batch_label)
        cd["_eligible_key"] = np.where(
            cd.get("email_key", pd.Series("", index=cd.index)).astype(str).str.len() > 3,
            "e:" + cd.get("email_key", pd.Series("", index=cd.index)).astype(str),
            "n:" + cd.get("student_key", pd.Series("", index=cd.index)).astype(str) + "|" + cd["Batch"].astype(str),
        )
    else:
        cd = pd.DataFrame(columns=["Batch", "Batch Display", "offered_date_parsed", "deadline_parsed", "_eligible_key"])

    rows = []
    for _, ev in event_info.iterrows():
        col = ev.get("column_name")
        if not col or col not in course_df.columns:
            continue
        ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
        ev_type = clean_text(ev.get("event_type", "")) or "Other"
        ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
        ev_date_norm = ev_date.normalize() if pd.notna(ev_date) else pd.NaT
        attended_mask = pd.to_numeric(course_df[col], errors="coerce").fillna(0).gt(0)
        attendance = int(attended_mask.sum())
        if attended_mask.any():
            adf = course_df.loc[attended_mask].copy()
            if "Batch" in adf.columns:
                batch_counts_series = adf.groupby("Batch")["student_name"].count().sort_values(ascending=False)
                batchwise = ", ".join([f"{int(v)} {_display_batch_label(k)}" for k, v in batch_counts_series.items() if clean_text(k)])
            else:
                batchwise = ""
        else:
            batchwise = ""

        # Event exists in these source batch sheets.
        sheet_txt = clean_text(ev.get("sheet", ""))
        event_sheet_batches = []
        for sh in re.split(r",|;|\|", sheet_txt):
            sh = clean_text(sh)
            if sh and "ug" in sh.lower():
                event_sheet_batches.append(sh)

        eligible_count = 0
        eligible_keys = []

        # Course eligibility for this event is based ONLY on the batch sheets where the event/activity exists.
        # Do not expand eligibility using Offered date -> Deadline ranges here.
        if not cd.empty:
            event_sheet_batches_norm = {_display_batch_label(x) for x in event_sheet_batches if clean_text(x)}
            elig = cd[cd["Batch Display"].astype(str).map(_display_batch_label).isin(event_sheet_batches_norm)].copy()
            if not elig.empty:
                eligible_keys = sorted([x for x in elig["_eligible_key"].dropna().astype(str).unique().tolist() if clean_text(x)])
                eligible_count = len(eligible_keys)

        # Active batches = only the batches where this event/activity exists in the batch sheets.
        active_batches = sorted(dict.fromkeys(event_sheet_batches))
        rows.append({
            "Event / Activity Name": ev_name,
            "Event / Activity Date": ev_date_norm,
            "Event / Activity Type": ev_type,
            "Attendance": attendance,
            f"Eligible {course_label} Students": eligible_count,
            "Active Batches": ", ".join(active_batches),
            "Batchwise Attended Students": batchwise,
            "_eligible_keys": "||".join(eligible_keys),
            "_event_key_name": normalize_name(ev_name),
            "_event_key_date": ev_date_norm,
            "_event_key_type": normalize_event_type_for_profile_graph(ev_type),
        })
    if not rows:
        return pd.DataFrame(columns=base_cols)
    raw = pd.DataFrame(rows)
    # Merge same event/activity name + date + type across multiple sheets.
    group_cols = ["_event_key_name", "_event_key_date", "_event_key_type"]
    def _merge_text(s):
        vals = []
        for x in s:
            for part in str(x).split(","):
                part = clean_text(part)
                if part and part not in vals:
                    vals.append(part)
        return ", ".join(vals)
    def _merge_key_count(s):
        vals = set()
        for x in s:
            for part in str(x).split("||"):
                part = clean_text(part)
                if part:
                    vals.add(part)
        return len(vals)

    out = raw.groupby(group_cols, dropna=False, as_index=False).agg({
        "Event / Activity Name": "first",
        "Event / Activity Date": "first",
        "Event / Activity Type": "first",
        "Attendance": "sum",
        f"Eligible {course_label} Students": "sum",
        "_eligible_keys": _merge_key_count,
        "Active Batches": _merge_text,
        "Batchwise Attended Students": _merge_text,
    })
    out[f"Eligible {course_label} Students"] = out["_eligible_keys"].astype(int)
    out = out[base_cols].sort_values("Event / Activity Date", ascending=False, na_position="last").reset_index(drop=True)
    return out

def render_courses_page(data):
    st.subheader("Courses")
    course_tabs = ["BFAI UG", "BSAI UG", "BMT UG", "IBMT UG"]
    course_labels = ["BFAI", "BSAI", "BMT", "IBMT"]
    prefixes = ["course_bfai", "course_bsai", "course_bmt", "course_ibmt"]
    tabs = st.tabs(course_tabs)
    for tab, course_label, title, prefix in zip(tabs, course_labels, course_tabs, prefixes):
        with tab:
            course_df, course_ctx = build_course_activity_context(course_label, data)
            _render_course_student_details(data, course_label, prefix)
            if course_df.empty:
                st.warning(f"No matching UG batch activity rows found for {title}. The student list above is still shown from Dates sheet if available.")
                continue
            render_sheet_detail(title, course_df, course_ctx, prefix, data=data)
            course_event_table = build_course_event_attendance_table(data, course_label, course_df, course_ctx)
            if not course_event_table.empty:
                course_event_table_display = course_event_table.copy()
                course_event_table_display["Event / Activity Date"] = pd.to_datetime(course_event_table_display["Event / Activity Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("")
                st.markdown("#### Course Event / Activity Attendance Details")
                st.caption("Merged by same event/activity name, date and type across UG batch sheets. Eligible students are counted from Dates → Course using Offered Date to Deadline.")
                st.dataframe(course_event_table_display, use_container_width=True, hide_index=True, height=420, key=f"{prefix}_course_merged_event_attendance")

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
        "In %": [pct(is_community_in_series(ug.get("community_status_value", pd.Series(dtype=object))).sum(), len(ug)), pct(is_community_in_series(pg.get("community_status_value", pd.Series(dtype=object))).sum(), len(pg))],
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
            et = pd.DataFrame(rows).groupby(["Program", "Event Type"], as_index=False)["Participation %"].mean()
            fig = px.bar(et, x="Event Type", y="Participation %", color="Program", barmode="group", title="Average Participation % by Event Type", color_discrete_map={"UG": GREEN, "PG": GREEN_3})
            st.plotly_chart(nice_layout(fig, height=420, x_tickangle=-30), use_container_width=True, key="uvspg_etype")

    def build_unique_attendee_type_df(program_label, sheets):
        rows = []
        for sheet in [s for s in sheets if s in data.get("activity_ctx", {}) and s in data.get("activities", {})]:
            info = data["activity_ctx"][sheet].get("event_info", pd.DataFrame())
            sdf = data["activities"][sheet]
            if info is None or info.empty or sdf is None or sdf.empty:
                continue
            for _, ev in info.iterrows():
                col = ev.get("column_name")
                etype = clean_text(ev.get("event_type", "Other")) or "Other"
                if not col or col not in sdf.columns:
                    continue
                att_mask = pd.to_numeric(sdf[col], errors="coerce").fillna(0) > 0
                if not att_mask.any():
                    continue
                subset = sdf.loc[att_mask].copy()
                subset["unique_student"] = subset.apply(lambda r: clean_text(r.get("email_key", "")) or clean_text(r.get("student_key", "")) or normalize_name(r.get("student_name", "")), axis=1)
                uniq = subset["unique_student"].replace("", np.nan).dropna().nunique()
                rows.append({"Program": program_label, "Event Type": etype, "Unique Attendees": int(uniq)})
        if not rows:
            return pd.DataFrame(columns=["Program", "Event Type", "Unique Attendees"])
        out = pd.DataFrame(rows)
        out["Event Type"] = out["Event Type"].apply(map_ugpg_unique_plot_event_type)
        out = out.groupby(["Program", "Event Type"], as_index=False)["Unique Attendees"].sum()
        return out

    ua = pd.concat([
        build_unique_attendee_type_df("UG", UG_BATCH_SHEETS),
        build_unique_attendee_type_df("PG", PG_BATCH_SHEETS)
    ], ignore_index=True)
    if not ua.empty:
        ua_total = ua.groupby("Event Type", as_index=False)["Unique Attendees"].sum().sort_values("Unique Attendees", ascending=False)
        u1, u2 = st.columns(2)
        with u1:
            fig = px.bar(ua, x="Event Type", y="Unique Attendees", color="Program", barmode="group", title="Unique Attendees by Event Type: UG vs PG", color_discrete_map={"UG": GREEN, "PG": GREEN_3})
            st.plotly_chart(nice_layout(fig, height=420, x_tickangle=-30), use_container_width=True, key="uvspg_unique_type")
        with u2:
            fig = px.bar(ua_total, x="Event Type", y="Unique Attendees", title="Total Unique Attendees by Event Type")
            fig.update_traces(marker_color=GREEN_2)
            st.plotly_chart(nice_layout(fig, height=420, x_tickangle=-30), use_container_width=True, key="uvspg_unique_total")
        st.dataframe(ua.sort_values(["Event Type", "Program"]), use_container_width=True, height=320, key="uvspg_unique_table")



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
    st.subheader("Conversion")
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


# ---------------- Tetr-X Analytics Additions ----------------

def _first_existing_col_like(frame: pd.DataFrame, keywords):
    if frame is None or frame.empty:
        return None
    for c in frame.columns:
        cl = clean_text(c).lower()
        if any(k in cl for k in keywords):
            return c
    return None


def _tx_scope_sheets(scope_label: str):
    if scope_label == "Tetr-X-UG":
        return ["Tetr-X-UG"]
    if scope_label == "Tetr-X-PG":
        return ["Tetr-X-PG"]
    return TX_SHEETS[:]


def _tx_master_lookup(data):
    overview = data.get("overview_df", pd.DataFrame())
    lookup = {}
    if overview is None or overview.empty:
        return lookup
    country_col = _first_existing_col_like(overview, ["country"])
    income_col = _first_existing_col_like(overview, ["income"])
    for _, r in overview.iterrows():
        sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("student_name", ""))
        if not sid or sid in lookup:
            continue
        lookup[sid] = {
            "Country": clean_text(r.get(country_col, "")) if country_col else "",
            "Income": clean_text(r.get(income_col, "")) if income_col else "",
            "Email": clean_text(r.get("email_key", "")),
        }
    return lookup


def build_tetrx_analytics_students(data, scope_label="Total"):
    """Student list + metrics for the Tetr X Analytics subsection."""
    rows = []
    master_lookup = _tx_master_lookup(data)
    today = get_today_ist()
    for sheet in _tx_scope_sheets(scope_label):
        tx_df = data.get("activities", {}).get(sheet, pd.DataFrame())
        if tx_df is None or tx_df.empty:
            continue
        program = "UG" if sheet.endswith("UG") else "PG"
        country_col = _first_existing_col_like(tx_df, ["country"])
        income_col = _first_existing_col_like(tx_df, ["income"])
        email_col = _first_existing_col_like(tx_df, ["email"])
        term_status_col = _first_existing_col_like(tx_df, [
            "tetr x/term 0 status", "tetr x term 0 status", "term 0 status",
            "term zero", "added to term 0", "community status", "admitted group"
        ])
        for _, r in tx_df.iterrows():
            sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("student_name", ""))
            if not sid:
                continue
            pay_dt = pd.to_datetime(r.get("payment_date_parsed", pd.NaT), errors="coerce")
            pay_dt = pay_dt.normalize() if pd.notna(pay_dt) else pd.NaT
            raw_status = clean_text(r.get("sheet_status_raw", ""))
            status_text = raw_status.lower().strip()
            is_refund = bool(r.get("sheet_is_refunded", False)) or ("refund" in status_text)
            # Current paid in Tetr-X Analytics must mean exactly Status == Admitted.
            is_current_paid = status_text == "admitted"
            # In Tetr-X Analytics, "in group" is based on the actual Tetr X/Term 0 Status column.
            # User rule: if the value is "Added to term 0" then the student is in group; otherwise not in group.
            term_status_raw = clean_text(r.get(term_status_col, "")) if term_status_col else ""
            term_status_lower = term_status_raw.lower().strip()
            in_group = term_status_lower == "added to term 0" or "added to term 0" in term_status_lower
            # Fallback only for older sheets where the raw Term 0 column could not be detected.
            if not term_status_col:
                in_group = clean_text(r.get("community_status_value", "")) == "Tetr X"
                term_status_raw = "Added to term 0" if in_group else clean_text(r.get("community_status_value", ""))
            days_left = ""
            if pd.notna(pay_dt):
                days_left = max(0, 60 - int((today - pay_dt).days))
            m = master_lookup.get(sid, {})
            email_val = clean_text(r.get("email_key", "")) or clean_text(r.get(email_col, "")) if email_col else clean_text(r.get("email_key", ""))
            rows.append({
                "student_id": sid,
                "Name": clean_text(r.get("student_name", "")),
                "Email": email_val,
                "UG/PG": program,
                "Country": clean_text(r.get(country_col, "")) if country_col else m.get("Country", ""),
                "Income": clean_text(r.get(income_col, "")) if income_col else m.get("Income", ""),
                "Payment Date": pay_dt,
                "Tetr X/Term 0 Status": term_status_raw or ("Added to term 0" if in_group else "Out"),
                "Refund Status": "Refunded" if is_refund else "Not Refunded",
                "Days left to ask refund": days_left,
                "source_sheet": sheet,
                "is_refunded": is_refund,
                "is_current_paid": is_current_paid,
                "in_group": in_group,
                "is_deferral": ("deferral" in status_text),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["Name", "Email", "Country", "Income", "Payment Date", "Tetr X/Term 0 Status", "Refund Status", "Days left to ask refund"]), {}
    # One row per student, latest/recent payment first.
    out = out.sort_values(["Payment Date", "Name"], ascending=[False, True], na_position="last")
    out = out.drop_duplicates("student_id", keep="first").reset_index(drop=True)
    paid_total = len(out)  # all rows/students present in the selected Tetr-X sheet(s)
    current_paid = int(out["is_current_paid"].fillna(False).astype(bool).sum())
    refunded = int(out["is_refunded"].fillna(False).astype(bool).sum())
    deferral = int(out.get("is_deferral", pd.Series(False, index=out.index)).fillna(False).astype(bool).sum())
    in_group = out["in_group"].fillna(False).astype(bool)
    current_paid_mask = out["is_current_paid"].fillna(False).astype(bool)
    refunded_mask = out["is_refunded"].fillna(False).astype(bool)
    paid_in_group = int((current_paid_mask & in_group).sum())
    refunded_in_group = int((refunded_mask & in_group).sum())
    paid_not_in_group = int((current_paid_mask & (~in_group)).sum())
    metrics = {
        "Total paid students": paid_total,
        "Current Paid Students": current_paid,
        "Refunded": refunded,
        "Deferral": deferral,
        "Paid & in group": paid_in_group,
        "Refunded & in group": refunded_in_group,
        "Paid & not in group": paid_not_in_group,
    }
    return out, metrics


def _tx_payment_lookup_by_program(data, scope_label="Total"):
    """Unique admitted Tetr-X students by program with their payment date.

    Used only for Tetr X Analytics → "Total Paid Students at the Event Date".
    User rule for this metric:
      - Total paid students at an event date = students in the Tetr-X sheet whose Status is exactly "Admitted"
      - and whose payment date is on or before that event date.

    Refunded / non-admitted rows are intentionally excluded from this denominator.
    """
    rows = []
    for sheet in _tx_scope_sheets(scope_label):
        df = data.get("activities", {}).get(sheet, pd.DataFrame())
        if df is None or df.empty:
            continue
        program = "UG" if sheet.endswith("UG") else "PG"

        # Identify the exact Status column as a fallback, but prefer parser-created flags.
        status_col = _first_existing_col_like(df, ["status", "payment status", "conversion status"])

        for _, r in df.iterrows():
            sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("student_name", ""))
            if not sid:
                continue

            # Strict paid rule: only exact Status == Admitted counts in the denominator.
            if "sheet_is_paid" in df.columns:
                is_admitted = bool(r.get("sheet_is_paid", False))
            else:
                raw_status = clean_text(r.get("sheet_status_raw", ""))
                if not raw_status and status_col:
                    raw_status = clean_text(r.get(status_col, ""))
                is_admitted = raw_status.strip().lower() == "admitted"

            if not is_admitted:
                continue

            pay_dt = pd.to_datetime(r.get("payment_date_parsed", pd.NaT), errors="coerce")
            if pd.isna(pay_dt):
                # defensive fallback: look for any payment/community-join date column on the row
                for c in df.columns:
                    cl = clean_text(c).lower()
                    if "payment" in cl or "community join" in cl:
                        pay_dt = pd.to_datetime(r.get(c, pd.NaT), errors="coerce")
                        if pd.notna(pay_dt):
                            break
            if pd.isna(pay_dt):
                continue
            rows.append({"student_id": sid, "Program": program, "Payment Date": pay_dt.normalize()})

    out = pd.DataFrame(rows)
    if out.empty:
        return {"UG": pd.DataFrame(columns=["student_id", "Payment Date"]), "PG": pd.DataFrame(columns=["student_id", "Payment Date"])}
    # If a student appears multiple times, use earliest admitted payment date for "already paid by event date".
    out = out.sort_values("Payment Date").drop_duplicates(["Program", "student_id"], keep="first")
    return {program: frame[["student_id", "Payment Date"]].copy() for program, frame in out.groupby("Program")}


def _tx_online_masterclass_rows(data, scope_label="Total"):
    """Per-event post-payment engagement for Online Events + Masterclasses in Tetr-X sheets.

    Denominator rule:
      - For Tetr-X-UG: admitted UG students whose payment date is on/before the event date.
      - For Tetr-X-PG: admitted PG students whose payment date is on/before the event date.
      - For Total: if the same event name + date exists in both UG and PG, merge it into one row
        and add both the UG and PG denominators and attendance.
    """
    paid_by_program = _tx_payment_lookup_by_program(data, scope_label)
    if not any(not frame.empty for frame in paid_by_program.values()):
        return pd.DataFrame()

    rows = []
    # Accumulate attendees by occurrence so duplicates are merged.
    occurrence_map = {}
    is_total_scope = clean_text(scope_label).lower() == "total"

    for sheet in _tx_scope_sheets(scope_label):
        df = data.get("activities", {}).get(sheet, pd.DataFrame())
        ctx = data.get("activity_ctx", {}).get(sheet, {})
        ev_info = ctx.get("event_info", pd.DataFrame()) if ctx else pd.DataFrame()
        if df is None or df.empty or ev_info is None or ev_info.empty:
            continue
        program = "UG" if sheet.endswith("UG") else "PG"
        for _, ev in ev_info.iterrows():
            ev_type_raw = clean_text(ev.get("event_type", ""))
            bucket = map_retention_bucket_event_type(ev_type_raw)
            if bucket != "Online Events & Masterclasses":
                continue
            col = ev.get("column_name")
            ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
            if not col or col not in df.columns or pd.isna(ev_date):
                continue
            ev_date = ev_date.normalize()
            ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)

            # For Total, merge UG + PG when same event name and date match.
            # For individual UG/PG sections, keep program-specific rows.
            occ_key_program_part = "TOTAL" if is_total_scope else program
            occ_key = f"{occ_key_program_part}|{normalize_name(ev_name)}|{ev_date.date()}|online_masterclass"

            if occ_key not in occurrence_map:
                occurrence_map[occ_key] = {
                    "Program": "Total" if is_total_scope else program,
                    "Event Name": ev_name,
                    "Event Date": ev_date,
                    "Month": ev_date.strftime("%b %Y"),
                    "Total Paid Students at the Event Date": 0,
                    "attendee_ids": set(),
                    "_denominator_programs_added": set(),
                }

            # Add this program's denominator once per merged occurrence. This is what makes Total
            # equal UG denominator + PG denominator for same event name/date.
            denom_added = occurrence_map[occ_key]["_denominator_programs_added"]
            if program not in denom_added:
                paid_frame = paid_by_program.get(program, pd.DataFrame())
                paid_at_event = 0
                if paid_frame is not None and not paid_frame.empty:
                    paid_dates = pd.to_datetime(paid_frame["Payment Date"], errors="coerce").dt.normalize()
                    paid_at_event = int((paid_dates.notna() & (paid_dates <= ev_date)).sum())
                occurrence_map[occ_key]["Total Paid Students at the Event Date"] += paid_at_event
                denom_added.add(program)

            attended = df[pd.to_numeric(df[col], errors="coerce").fillna(0) > 0].copy()
            if attended.empty:
                continue
            for _, r in attended.iterrows():
                sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("student_name", ""))
                if not sid:
                    continue
                pay_dt = pd.to_datetime(r.get("payment_date_parsed", pd.NaT), errors="coerce")
                if pd.notna(pay_dt) and ev_date >= pay_dt.normalize():
                    attendee_key = f"{program}|{sid}" if is_total_scope else sid
                    occurrence_map[occ_key]["attendee_ids"].add(attendee_key)

    for item in occurrence_map.values():
        attendance = len(item.pop("attendee_ids", set()))
        item.pop("_denominator_programs_added", None)
        paid_at_event = int(item.get("Total Paid Students at the Event Date", 0) or 0)
        item["Attendance"] = attendance
        item["Attendance %"] = (attendance / paid_at_event * 100) if paid_at_event else 0.0
        rows.append(item)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Event Date", "Program", "Event Name"]).reset_index(drop=True)

def _tx_top_engaged_students(data, scope_label="Total"):
    """Top Tetr-X students by attended post-payment events / eligible post-payment events."""
    student_table, _ = build_tetrx_analytics_students(data, scope_label)
    if student_table.empty:
        return pd.DataFrame()
    rows = []
    for _, stu in student_table.iterrows():
        sid = clean_text(stu.get("student_id", ""))
        program = clean_text(stu.get("UG/PG", ""))
        pay_dt = pd.to_datetime(stu.get("Payment Date", pd.NaT), errors="coerce")
        if not sid or pd.isna(pay_dt):
            continue
        pay_dt = pay_dt.normalize()
        sheet = "Tetr-X-UG" if program == "UG" else "Tetr-X-PG"
        df = data.get("activities", {}).get(sheet, pd.DataFrame())
        ctx = data.get("activity_ctx", {}).get(sheet, {})
        ev_info = ctx.get("event_info", pd.DataFrame()) if ctx else pd.DataFrame()
        if df is None or df.empty or ev_info is None or ev_info.empty:
            continue
        mask = pd.Series(False, index=df.index)
        if clean_text(stu.get("Email", "")) and "email_key" in df.columns:
            mask = mask | df["email_key"].astype(str).eq(normalize_email(stu.get("Email", "")))
        # student_id can be email or normalized name; use both defensively.
        if "student_key" in df.columns:
            mask = mask | df["student_key"].astype(str).eq(sid)
        if "email_key" in df.columns:
            mask = mask | df["email_key"].astype(str).eq(sid)
        part = df.loc[mask].copy()
        if part.empty:
            continue
        eligible = set()
        attended = set()
        type_counts = {}
        for _, ev in ev_info.iterrows():
            col = ev.get("column_name")
            ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
            if not col or col not in part.columns or pd.isna(ev_date):
                continue
            ev_date = ev_date.normalize()
            if ev_date < pay_dt:
                continue
            ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
            ev_type = map_retention_bucket_event_type(clean_text(ev.get("event_type", "")) or "Other")
            occ_key = f"{program}|{normalize_name(ev_name)}|{normalize_name(ev_type)}|{ev_date.date()}"
            eligible.add(occ_key)
            if (pd.to_numeric(part[col], errors="coerce").fillna(0) > 0).any():
                attended.add(occ_key)
                type_counts[ev_type] = type_counts.get(ev_type, 0) + 1
        eligible_count = len(eligible)
        attended_count = len(attended)
        rows.append({
            "Name": stu.get("Name", ""),
            "Email": stu.get("Email", ""),
            "UG/PG": program,
            "Payment Date": pay_dt,
            "Eligible Events": eligible_count,
            "Attended Events": attended_count,
            "Engagement %": (attended_count / eligible_count * 100) if eligible_count else 0.0,
            "Top Event Types": ", ".join([f"{k}: {v}" for k, v in sorted(type_counts.items(), key=lambda x: x[1], reverse=True)[:4]]),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["Engagement %", "Attended Events", "Name"], ascending=[False, False, True]).reset_index(drop=True)


def render_tetrx_analytics_scope(data, scope_label: str, key_prefix: str):
    students, metrics = build_tetrx_analytics_students(data, scope_label)
    st.markdown(f"### {scope_label}")
    if students.empty:
        st.info(f"No Tetr-X students found for {scope_label}.")
        return
    c = st.columns(6)
    c[0].metric("Total paid students", f"{metrics.get('Total paid students', 0):,}")
    c[1].metric("Current Paid Students", f"{metrics.get('Current Paid Students', 0):,}")
    c[2].metric("Refunded", f"{metrics.get('Refunded', 0):,}")
    c[3].metric("Paid & in group", f"{metrics.get('Paid & in group', 0):,}")
    c[4].metric("Refunded & in group", f"{metrics.get('Refunded & in group', 0):,}")
    c[5].metric("Paid & not in group", f"{metrics.get('Paid & not in group', 0):,}")

    st.markdown("#### Names of students in Tetr X")
    display = students[["Name", "Email", "Country", "Income", "Payment Date", "Tetr X/Term 0 Status", "Refund Status", "Days left to ask refund"]].copy()
    display["Payment Date"] = pd.to_datetime(display["Payment Date"], errors="coerce").dt.strftime("%d-%b-%Y")
    st.dataframe(display, use_container_width=True, height=360, hide_index=True, key=f"{key_prefix}_students")

    st.markdown("#### Post-Payment Engagement (Online Events + Masterclasses)")
    eng = _tx_online_masterclass_rows(data, scope_label)
    if eng.empty:
        st.info("No dated post-payment Online Event / Masterclass attendance found.")
    else:
        chart_df = eng.copy()
        chart_df["Date Label"] = pd.to_datetime(chart_df["Event Date"], errors="coerce").dt.strftime("%d %b")
        fig = px.bar(
            chart_df,
            x="Date Label",
            y="Attendance %",
            color="Month",
            hover_name="Event Name",
            hover_data={"Total Paid Students at the Event Date": True, "Attendance": True, "Attendance %": ':.1f', "Program": True, "Date Label": False},
            title="Post-Payment Engagement (Online Events + Masterclasses)",
        )
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        st.plotly_chart(nice_layout(fig, height=390, x_tickangle=-25), use_container_width=True, key=f"{key_prefix}_postpay_online_chart")
        table = eng.copy()
        table["Event Date"] = pd.to_datetime(table["Event Date"], errors="coerce").dt.strftime("%d-%b-%Y")
        st.dataframe(table[["Program", "Event Date", "Event Name", "Total Paid Students at the Event Date", "Attendance", "Attendance %"]], use_container_width=True, hide_index=True, height=260, key=f"{key_prefix}_postpay_online_table")

    st.markdown("#### Top engaged students in Tetr X")
    top = _tx_top_engaged_students(data, scope_label)
    if top.empty:
        st.info("No eligible post-payment event data found for top engaged students.")
    else:
        chart_top = top.head(15).copy()
        fig = px.bar(chart_top, x="Engagement %", y="Name", orientation="h", hover_data=["Attended Events", "Eligible Events", "Top Event Types"], title="Top Engaged Students by Eligible Post-Payment Attendance %")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(nice_layout(fig, height=460), use_container_width=True, key=f"{key_prefix}_top_engaged_chart")
        disp = top.copy()
        disp["Payment Date"] = pd.to_datetime(disp["Payment Date"], errors="coerce").dt.strftime("%d-%b-%Y")
        st.dataframe(disp, use_container_width=True, hide_index=True, height=360, key=f"{key_prefix}_top_engaged_table")


def render_tetrx_analytics(data):
    st.markdown("## Tetr X Analytics")
    subtabs = st.tabs(["Total", "Tetr-X-UG", "Tetr-X-PG"])
    for tab, scope in zip(subtabs, ["Total", "Tetr-X-UG", "Tetr-X-PG"]):
        with tab:
            render_tetrx_analytics_scope(data, scope, f"txanalytics_{scope.replace('-', '_').replace(' ', '_')}")

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

    tab_labels = available + ["Tetr X Analytics"]
    tabs = st.tabs(tab_labels)
    for tab, label in zip(tabs, tab_labels):
        with tab:
            if label == "Tetr X Analytics":
                render_tetrx_analytics(data)
            else:
                render_sheet_detail(label, data["activities"][label], data["activity_ctx"][label], f"tx_{label}", data=data)




def build_retention_analytics_v2(data):
    """Build Retention-tab-only cohort and post-payment Tetr-X activity data.

    Retention rules used here only:
    - Cohort comes ONLY from Tetr-X sheets.
    - Payment dates come ONLY from Tetr-X sheets.
    - Paid cohort = rows where the Tetr-X Status is exactly "Admitted" OR the Status/refund fields contain refund.
      Refund rows are included because they were paid/Tetr-X students who later churned.
    - Churn = any Tetr-X status/refund text contains "refund".
    - Retained = not churned AND completed 60 days from payment date.
    - Post-payment opportunities = Tetr-X activities/events that happened on/after each student's payment date.
    - Same event name + type + date is deduped across sheets/program within its program.
    """
    activities = data.get("activities", {})
    activity_ctx = data.get("activity_ctx", {})
    tx_rows = []
    today = get_today_ist().normalize() if "get_today_ist" in globals() else pd.Timestamp.today().normalize()

    def _status_columns(frame: pd.DataFrame, ctx: dict | None = None):
        ctx = ctx or {}
        event_cols = set(ctx.get("event_cols", []) or [])
        preferred = []
        broader = []
        for c in frame.columns:
            if c in event_cols:
                continue
            cl = clean_text(c).lower()
            if cl in {"status", "payment status"} or cl.endswith(" status"):
                preferred.append(c)
            if any(k in cl for k in ["status", "payment", "refund", "comment"]):
                broader.append(c)
        # Keep order, no duplicates
        seen = set()
        preferred = [c for c in preferred if not (c in seen or seen.add(c))]
        seen = set()
        broader = [c for c in broader if not (c in seen or seen.add(c))]
        return preferred, broader

    def _joined_text(frame: pd.DataFrame, cols) -> pd.Series:
        """Safely join text across metadata/status columns.

        Streamlit Cloud was crashing here because pandas' DataFrame.agg(" ".join, axis=1)
        can still receive non-plain-string objects when duplicate column names or extension dtypes
        are present. This helper normalizes every value explicitly and handles duplicate columns
        without changing any retention logic.
        """
        cols = [c for c in cols if c in frame.columns]
        if not cols:
            return pd.Series("", index=frame.index)

        joined = []
        # Selecting by column name can return a DataFrame when duplicate headers exist, so iterate
        # row positions after selecting the matching columns and flatten row values manually.
        selected = frame.loc[:, cols].copy()
        for _, row in selected.iterrows():
            vals = []
            for v in row.tolist():
                if pd.isna(v):
                    continue
                s = clean_text(v)
                if s:
                    vals.append(s)
            joined.append(" ".join(vals).lower().strip())
        return pd.Series(joined, index=frame.index)

    for tx_sheet in TX_SHEETS:
        tx_df = activities.get(tx_sheet, pd.DataFrame())
        if tx_df is None or tx_df.empty:
            continue
        program = "UG" if tx_sheet.endswith("UG") else "PG"
        frame = tx_df.copy()
        ctx = activity_ctx.get(tx_sheet, {})
        frame["program"] = program
        frame["student_id"] = frame.apply(student_unique_id_from_row, axis=1)
        frame["payment_date"] = pd.to_datetime(frame.get("payment_date_parsed", pd.NaT), errors="coerce")

        preferred_status_cols, broad_status_cols = _status_columns(frame, ctx)
        exact_status_text = _joined_text(frame, preferred_status_cols)
        broad_status_text = _joined_text(frame, broad_status_cols)
        if broad_status_text.eq("").all():
            # final fallback: scan non-event metadata columns only, never attendance event columns
            event_cols = set(ctx.get("event_cols", []) or [])
            meta_cols = [c for c in frame.columns if c not in event_cols]
            broad_status_text = _joined_text(frame, meta_cols)

        # Refund/churn is intentionally broad: any refund/refunded/refund confirmed keyword in status-like metadata.
        frame["retention_is_refunded"] = broad_status_text.str.contains(r"\brefund", regex=True, na=False)
        # Paid is strict admitted from Status/payment-status. Refund rows are added to the cohort separately.
        frame["retention_is_admitted"] = exact_status_text.str.strip().eq("admitted")
        if "sheet_status_raw" in frame.columns:
            raw_status = frame["sheet_status_raw"].fillna("").astype(str).str.strip().str.lower()
            frame["retention_is_admitted"] = frame["retention_is_admitted"] | raw_status.eq("admitted")
            frame["retention_is_refunded"] = frame["retention_is_refunded"] | raw_status.str.contains(r"\brefund", regex=True, na=False)
        # Use existing parser flags as fallback only; exact Admitted rule remains strict.
        if "sheet_is_refunded" in frame.columns:
            frame["retention_is_refunded"] = frame["retention_is_refunded"] | frame["sheet_is_refunded"].fillna(False).astype(bool)

        # Retention UG/PG requirement: include deferral students from Tetr-X sheets in the
        # student-level Retention tables. The cohort builder must include them here first;
        # otherwise they never reach _build_retention_student_postpayment_table(...).
        # UG: Status contains "Deferral". PG: Status contains "Admitted:Deferral" or "Deferral".
        frame["retention_is_deferral"] = broad_status_text.str.contains("deferral", case=False, na=False)
        if "sheet_status_raw" in frame.columns:
            raw_status = frame["sheet_status_raw"].fillna("").astype(str).str.strip().str.lower()
            frame["retention_is_deferral"] = frame["retention_is_deferral"] | raw_status.str.contains("deferral", na=False)

        frame["retention_in_paid_cohort"] = (
            frame["retention_is_admitted"]
            | frame["retention_is_refunded"]
            | frame["retention_is_deferral"]
        )
        tx_rows.append(frame)

    empty_paid = pd.DataFrame(columns=["student_id", "student_name", "program", "payment_date", "is_churned", "is_retained_60d"])
    empty_events = pd.DataFrame(columns=["student_id", "student_name", "program", "is_churned", "payment_date", "event_name", "event_type", "event_date", "dedupe_key", "days_after_payment"])
    empty_occ = pd.DataFrame(columns=["program", "event_name", "event_type", "event_date", "occurrence_key"])
    empty_avail = pd.DataFrame(columns=["student_id", "program", "event_name", "event_type", "event_date", "occurrence_key", "student_occurrence_key", "days_after_payment"])
    if not tx_rows:
        return empty_paid, empty_events, empty_occ, empty_avail, {}

    tx_all = pd.concat(tx_rows, ignore_index=True)
    tx_all = tx_all[tx_all["student_id"].astype(str).ne("")].copy()
    cohort_all = tx_all[tx_all["retention_in_paid_cohort"].fillna(False).astype(bool)].copy()
    if cohort_all.empty:
        return empty_paid, empty_events, empty_occ, empty_avail, {}

    # Churn/refund is any refund row for the student in Tetr-X. Keep one cohort row per student.
    refunded_ids = set(cohort_all.loc[cohort_all["retention_is_refunded"].fillna(False).astype(bool), "student_id"].astype(str).tolist())
    admitted_ids = set(cohort_all.loc[cohort_all["retention_is_admitted"].fillna(False).astype(bool), "student_id"].astype(str).tolist())
    deferral_ids = set(cohort_all.loc[cohort_all.get("retention_is_deferral", pd.Series(False, index=cohort_all.index)).fillna(False).astype(bool), "student_id"].astype(str).tolist())
    cohort_ids = admitted_ids | refunded_ids | deferral_ids
    cohort_all = cohort_all[cohort_all["student_id"].astype(str).isin(cohort_ids)].copy()
    cohort = cohort_all.sort_values(["payment_date", "student_name"], na_position="last").drop_duplicates("student_id", keep="first").reset_index(drop=True)
    cohort["is_churned"] = cohort["student_id"].astype(str).isin(refunded_ids)
    cohort["completed_60_days"] = pd.to_datetime(cohort["payment_date"], errors="coerce").apply(lambda d: bool(pd.notna(d) and (today - d.normalize()).days >= 60))
    cohort["is_retained_60d"] = (~cohort["is_churned"].astype(bool)) & cohort["completed_60_days"].astype(bool)

    event_rows = []
    occurrence_rows = []
    available_rows = []

    for _, stu in cohort.iterrows():
        sid = clean_text(stu.get("student_id", ""))
        pay_dt = pd.to_datetime(stu.get("payment_date", pd.NaT), errors="coerce")
        if not sid or pd.isna(pay_dt):
            continue
        pay_dt = pay_dt.normalize()
        program = clean_text(stu.get("program", ""))
        tx_sheet = "Tetr-X-UG" if program == "UG" else "Tetr-X-PG"
        sdf = activities.get(tx_sheet, pd.DataFrame())
        ctx = activity_ctx.get(tx_sheet, {})
        ev_info = ctx.get("event_info", pd.DataFrame()) if ctx else pd.DataFrame()
        if sdf is None or sdf.empty or ev_info is None or ev_info.empty:
            continue

        mask = pd.Series(False, index=sdf.index)
        if clean_text(stu.get("email_key", "")) and "email_key" in sdf.columns:
            mask = mask | sdf["email_key"].astype(str).eq(clean_text(stu.get("email_key", "")))
        if clean_text(stu.get("student_key", "")) and "student_key" in sdf.columns:
            mask = mask | sdf["student_key"].astype(str).eq(clean_text(stu.get("student_key", "")))
        part = sdf.loc[mask].copy()
        if part.empty:
            continue

        for _, ev in ev_info.iterrows():
            col = ev.get("column_name")
            ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
            if not col or col not in part.columns or pd.isna(ev_date):
                continue
            ev_date = ev_date.normalize()
            days_after = int((ev_date - pay_dt).days)
            if days_after < 0:
                continue
            event_type_raw = clean_text(ev.get("event_type", "")) or "Other"
            event_bucket = map_retention_bucket_event_type(event_type_raw)
            event_name = clean_text(ev.get("event_name", "")) or clean_text(col)
            occ_key = f"{program}|{event_bucket.lower()}|{normalize_name(event_name)}|{ev_date.date()}"
            student_occ_key = f"{sid}|{occ_key}"
            occurrence_rows.append({"program": program, "event_name": event_name, "event_type": event_bucket, "event_date": ev_date, "occurrence_key": occ_key})
            available_rows.append({
                "student_id": sid,
                "program": program,
                "event_name": event_name,
                "event_type": event_bucket,
                "event_date": ev_date,
                "occurrence_key": occ_key,
                "student_occurrence_key": student_occ_key,
                "days_after_payment": days_after,
            })
            if not (pd.to_numeric(part[col], errors="coerce").fillna(0) > 0).any():
                continue
            event_rows.append({
                "student_id": sid,
                "student_name": clean_text(stu.get("student_name", "")),
                "program": program,
                "is_churned": sid in refunded_ids,
                "is_retained_60d": bool(stu.get("is_retained_60d", False)),
                "payment_date": pay_dt,
                "event_name": event_name,
                "event_type": event_bucket,
                "event_date": ev_date,
                "dedupe_key": student_occ_key,
                "days_after_payment": days_after,
            })

    events_df = pd.DataFrame(event_rows) if event_rows else empty_events.copy()
    if not events_df.empty:
        events_df = events_df.drop_duplicates(subset=["dedupe_key"]).copy()
        events_df = events_df[events_df["days_after_payment"] >= 0].copy()
    occurrences_df = pd.DataFrame(occurrence_rows).drop_duplicates(subset=["occurrence_key"]) if occurrence_rows else empty_occ.copy()
    available_df = pd.DataFrame(available_rows).drop_duplicates(subset=["student_occurrence_key"]) if available_rows else empty_avail.copy()

    cohort_ids = set(cohort["student_id"].astype(str).tolist())
    refunded_cohort_ids = refunded_ids & cohort_ids
    retained_ids = set(cohort.loc[cohort["is_retained_60d"].fillna(False).astype(bool), "student_id"].astype(str).tolist())
    metrics = {
        "paid_total": len(cohort_ids),
        "paid_ug": int((cohort["program"] == "UG").sum()),
        "paid_pg": int((cohort["program"] == "PG").sum()),
        "refunded_total": len(refunded_cohort_ids),
        "retained_total": len(retained_ids),
        "eligible_60d_total": len(retained_ids | refunded_cohort_ids),
    }
    return cohort, events_df, occurrences_df, available_df, metrics


def build_retention_window_comparison(paid_df, events_df, available_df):
    if paid_df is None or paid_df.empty:
        return pd.DataFrame()
    rows = []
    for label, max_day in [("15D", 15), ("30D", 30), ("45D", 45), ("60D", 60)]:
        cohorts = []
        if "is_retained_60d" in paid_df.columns:
            cohorts.append((set(paid_df.loc[paid_df["is_retained_60d"].fillna(False).astype(bool), "student_id"].astype(str).tolist()), "Retained"))
        else:
            cohorts.append((set(), "Retained"))
        if "is_churned" in paid_df.columns:
            cohorts.append((set(paid_df.loc[paid_df["is_churned"].fillna(False).astype(bool), "student_id"].astype(str).tolist()), "Churned"))
        else:
            cohorts.append((set(), "Churned"))
        for group_ids, group_name in cohorts:
            if not group_ids:
                rows.append({"Window": label, "User Type": group_name, "Avg Activities": 0.0, "Avg Available Activities": 0.0, "Engagement %": 0.0})
                continue
            sub = events_df[(events_df["student_id"].astype(str).isin(group_ids)) & (events_df["days_after_payment"].between(0, max_day))] if events_df is not None and not events_df.empty else pd.DataFrame()
            avail_sub = available_df[(available_df["student_id"].astype(str).isin(group_ids)) & (available_df["days_after_payment"].between(0, max_day))] if available_df is not None and not available_df.empty else pd.DataFrame()
            counts = sub.groupby("student_id")["dedupe_key"].nunique() if not sub.empty else pd.Series(dtype=int)
            avail_counts = avail_sub.groupby("student_id")["student_occurrence_key"].nunique() if not avail_sub.empty else pd.Series(dtype=int)
            avg_count = float(counts.reindex(list(group_ids), fill_value=0).mean())
            avg_avail = float(avail_counts.reindex(list(group_ids), fill_value=0).mean())
            pct = (avg_count / avg_avail * 100) if avg_avail else 0.0
            rows.append({"Window": label, "User Type": group_name, "Avg Activities": avg_count, "Avg Available Activities": avg_avail, "Engagement %": pct})
    return pd.DataFrame(rows)


def _render_true_retention_overview(data):
    paid_df, events_df, occurrences_df, available_df, metrics = build_retention_analytics_v2(data)
    if paid_df.empty:
        st.warning("No paid/admitted/refunded Tetr-X students were found for retention analytics.")
        return
    paid_total = int(metrics.get("paid_total", 0))
    refunded_total = int(metrics.get("refunded_total", 0))
    retained_total = int(metrics.get("retained_total", 0))
    eligible_60d_total = int(metrics.get("eligible_60d_total", retained_total + refunded_total))
    retention_rate = (retained_total / eligible_60d_total * 100) if eligible_60d_total else 0.0

    st.markdown("### Success Metric")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("No. of students paid", f"{paid_total:,}")
    c2.metric("No. of students refunded", f"{refunded_total:,}")
    c3.metric("Retained students (60D)", f"{retained_total:,}")
    c4.metric("Retention rate", f"{retention_rate:.1f}%")

    paid_ids = set(paid_df["student_id"].astype(str).unique())
    if available_df is not None and not available_df.empty:
        avail_counts = available_df.groupby("student_id")["student_occurrence_key"].nunique()
        avg_available_total = float(avail_counts.reindex(list(paid_ids), fill_value=0).mean()) if paid_ids else 0.0
        avail_by_type_student = available_df.groupby(["student_id", "event_type"])["student_occurrence_key"].nunique().reset_index(name="Available")
        avg_avail_by_type = (
            avail_by_type_student.pivot_table(index="student_id", columns="event_type", values="Available", aggfunc="sum", fill_value=0)
            .reindex(list(paid_ids), fill_value=0)
            .mean()
            .to_dict()
        ) if paid_ids else {}
    else:
        avg_available_total = 0.0
        avg_avail_by_type = {}

    st.markdown("### Engagement Metric · Tetr-X Group")
    metric_cols = st.columns(4)
    metric_cols[0].metric("Avg activities students could attend post payment", f"{avg_available_total:.1f}")
    for idx, label in enumerate(["Online Events & Masterclasses", "Competitions & Hackathons", "General/Fun"], start=1):
        metric_cols[idx].metric(label, f"{float(avg_avail_by_type.get(label, 0.0)):.1f}")

    # Event type performance: attendance divided by actual post-payment student-level opportunities.
    if available_df is not None and not available_df.empty:
        available_type = available_df.groupby("event_type", as_index=False).agg(
            Available_Opportunities=("student_occurrence_key", "nunique"),
            Event_Occurrences=("occurrence_key", "nunique"),
        )
        attended_type = events_df.groupby("event_type", as_index=False).agg(
            Attendance_Hits=("dedupe_key", "nunique"),
            Unique_Attendees=("student_id", "nunique"),
        ) if events_df is not None and not events_df.empty else pd.DataFrame(columns=["event_type", "Attendance_Hits", "Unique_Attendees"])
        perf = available_type.merge(attended_type, on="event_type", how="left").fillna({"Attendance_Hits": 0, "Unique_Attendees": 0})
        perf["Attendance Rate %"] = np.where(perf["Available_Opportunities"].gt(0), perf["Attendance_Hits"] / perf["Available_Opportunities"] * 100, 0)
        perf["Avg Attendance per Paid Student"] = perf["Attendance_Hits"] / max(paid_total, 1)
        perf = perf.sort_values(["Attendance Rate %", "Attendance_Hits"], ascending=False)
        fig = px.bar(
            perf,
            x="event_type",
            y="Attendance Rate %",
            title="Tetr-X Post-Payment Attendance Rate by Activity Type",
            hover_data=["Attendance_Hits", "Available_Opportunities", "Event_Occurrences", "Unique_Attendees", "Avg Attendance per Paid Student"],
        )
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=340, x_tickangle=-25), use_container_width=True, key="new_ret_att_rate_by_type")
        st.dataframe(
            perf.rename(columns={
                "event_type": "Activity Type",
                "Available_Opportunities": "Available Opportunities",
                "Event_Occurrences": "Unique Event Occurrences",
                "Attendance_Hits": "Attendance Hits",
                "Unique_Attendees": "Unique Attendees",
            }),
            use_container_width=True,
            height=260,
            key="new_ret_att_rate_table",
        )

    st.markdown("### Retained vs Churn User Analytics")
    if events_df.empty and (available_df is None or available_df.empty):
        st.info("No post-payment Tetr-X attendance found for retained/churn comparison.")
    else:
        churned_ids = set(paid_df.loc[paid_df.get("is_churned", False).astype(bool), "student_id"].astype(str).unique()) if "is_churned" in paid_df.columns else set()
        retained_ids = paid_ids - churned_ids
        def avg_eng(ids):
            if not ids:
                return 0.0, 0.0, 0.0
            counts = events_df[events_df["student_id"].astype(str).isin(ids)].groupby("student_id")["dedupe_key"].nunique() if events_df is not None and not events_df.empty else pd.Series(dtype=int)
            avail_counts = available_df[available_df["student_id"].astype(str).isin(ids)].groupby("student_id")["student_occurrence_key"].nunique() if available_df is not None and not available_df.empty else pd.Series(dtype=int)
            avg = float(counts.reindex(list(ids), fill_value=0).mean())
            avg_avail = float(avail_counts.reindex(list(ids), fill_value=0).mean())
            pct = (avg / avg_avail * 100) if avg_avail else 0.0
            return avg, avg_avail, pct
        ret_avg, ret_avail, ret_pct = avg_eng(retained_ids)
        churn_avg, churn_avail, churn_pct = avg_eng(churned_ids)
        a, b = st.columns(2)
        a.metric("Average engagement by retained student", f"{ret_avg:.1f}/{ret_avail:.1f}", delta=f"{ret_pct:.1f}%")
        b.metric("Average engagement of churned student", f"{churn_avg:.1f}/{churn_avail:.1f}", delta=f"{churn_pct:.1f}%")
        comp = build_retention_window_comparison(paid_df, events_df, available_df)
        if not comp.empty:
            fig = px.bar(comp, x="Window", y="Engagement %", color="User Type", barmode="group", title="15/30/45/60 Days Comparison · Retained vs Churn", hover_data=["Avg Activities", "Avg Available Activities"])
            st.plotly_chart(nice_layout(fig, height=340), use_container_width=True, key="new_retained_churn_chart")
            st.dataframe(comp, use_container_width=True, height=220, key="new_retained_churn_table")

    st.markdown("### Top Engagement Activities/Events · Tetr-X")
    if events_df.empty:
        st.info("No post-payment Tetr-X events found.")
    else:
        top_events = events_df.groupby(["event_name", "event_type"], as_index=False).agg(
            Attendees=("student_id", "nunique"),
            Attendance=("dedupe_key", "nunique"),
            Dates=("event_date", lambda s: ", ".join([pd.to_datetime(x).strftime("%d-%b-%Y") for x in sorted(pd.to_datetime(s).dropna().unique())[:5]])),
        ).sort_values(["Attendance", "Attendees"], ascending=False).head(20)
        fig = px.bar(top_events.head(12), x="Attendance", y="event_name", color="event_type", orientation="h", title="Top Tetr-X Activities by Attendance", hover_data=["Dates", "Attendees"])
        st.plotly_chart(nice_layout(fig, height=440), use_container_width=True, key="new_ret_top_events_chart")
        st.dataframe(top_events.rename(columns={"event_name": "Event", "event_type": "Activity Type"}), use_container_width=True, height=360, key="new_ret_top_events_table")


def _retention_prepayment_activity_count_for_student(data: dict, stu: pd.Series, program: str) -> int:
    """Count deduped batch-sheet activities attended before the student's payment date.
    Used only for Retention UG/PG tables. This intentionally checks batch sheets,
    not Tetr-X sheets, and keeps other dashboard logic untouched.
    """
    pay_dt = pd.to_datetime(stu.get("payment_date", pd.NaT), errors="coerce")
    if pd.isna(pay_dt):
        return 0
    pay_dt = pay_dt.normalize()
    sid_email = clean_text(stu.get("email_key", ""))
    sid_name = clean_text(stu.get("student_key", ""))
    if not sid_email and not sid_name:
        sid_name = normalize_name(stu.get("student_name", ""))

    activities = data.get("activities", {}) if isinstance(data, dict) else {}
    contexts = data.get("activity_ctx", {}) if isinstance(data, dict) else {}
    seen = set()
    for sheet, df in activities.items():
        if sheet in {"Tetr-X-UG", "Tetr-X-PG"}:
            continue
        if infer_program_from_sheet(sheet) != program:
            continue
        if df is None or df.empty:
            continue
        match = pd.Series(False, index=df.index)
        if sid_email and "email_key" in df.columns:
            match = match | df["email_key"].astype(str).eq(sid_email)
        if sid_name and "student_key" in df.columns:
            match = match | df["student_key"].astype(str).eq(sid_name)
        if not match.any():
            continue
        ctx = contexts.get(sheet, {}) if isinstance(contexts, dict) else {}
        event_info = ctx.get("event_info", pd.DataFrame()) if isinstance(ctx, dict) else pd.DataFrame()
        if event_info is None or event_info.empty:
            continue
        sub = df.loc[match].copy()
        for _, ev in event_info.iterrows():
            col = ev.get("column_name", "")
            if not col or col not in sub.columns:
                continue
            ev_dt = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
            if pd.isna(ev_dt) or ev_dt.normalize() >= pay_dt:
                continue
            attended = pd.to_numeric(sub[col], errors="coerce").fillna(0).gt(0).any()
            if not attended:
                continue
            ev_name = clean_text(ev.get("event_name", col)).lower()
            ev_type = clean_text(ev.get("event_type", "Other")).lower()
            seen.add((ev_name, ev_type, ev_dt.normalize().strftime("%Y-%m-%d")))
    return len(seen)


def _build_retention_student_postpayment_table(paid_df: pd.DataFrame, events_df: pd.DataFrame, program: str, data: dict = None) -> pd.DataFrame:
    """Student-level post-payment retention table for Retention UG/PG tabs only."""
    columns = [
        "Name", "Email", "Payment Date", "Last Engaged Date", "Days passed since Engaged",
        "Activities done pre-payment", "Activities done post payment", "0-15 days", "16-30 days", "31-45 days", "46-60 days", "60+ days",
        "Online Event and Masterclass Attended", "Competition and hackathon Participated",
        "General/Fun/Quiz/poll/fun task Participated", "Names of Events/activities they participated in",
        "Date of Activation Post-Payment", "Activation Event/Activity Name - Post Payment",
    ]
    if paid_df is None or paid_df.empty:
        return pd.DataFrame(columns=columns)

    today = get_today_ist().normalize() if "get_today_ist" in globals() else pd.Timestamp.today().normalize()
    base = paid_df.copy()
    base["program"] = base.get("program", "").astype(str)
    base = base[base["program"].eq(program)].copy()

    # Requirement update: include Status = Admitted plus Deferral students in the UG/PG Retention tables.
    # UG: include rows whose Tetr-X-UG status contains "Deferral".
    # PG: include rows whose Tetr-X-PG status contains "Admitted:Deferral" or "Deferral".
    status_text = base.get("sheet_status_raw", pd.Series("", index=base.index)).fillna("").astype(str).str.strip().str.lower()
    if "retention_is_admitted" in base.columns:
        admitted_mask = base["retention_is_admitted"].fillna(False).astype(bool)
    else:
        admitted_mask = status_text.eq("admitted")
    deferral_mask = status_text.str.contains("deferral", na=False)
    base = base[(admitted_mask | deferral_mask)].copy()
    if base.empty:
        return pd.DataFrame(columns=columns)

    ev = events_df.copy() if events_df is not None and not events_df.empty else pd.DataFrame()
    if not ev.empty:
        ev["program"] = ev.get("program", "").astype(str)
        ev = ev[ev["program"].eq(program)].copy()
        ev["student_id"] = ev.get("student_id", "").astype(str)
        ev["days_after_payment"] = pd.to_numeric(ev.get("days_after_payment", 0), errors="coerce")
        ev["event_date"] = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
        ev = ev[ev["days_after_payment"].ge(0)].copy()
        if "dedupe_key" in ev.columns:
            ev = ev.drop_duplicates(subset=["dedupe_key"]).copy()

    rows = []
    for _, stu in base.sort_values(["payment_date", "student_name"], na_position="last").iterrows():
        sid = clean_text(stu.get("student_id", ""))
        sub = ev[ev["student_id"].eq(sid)].copy() if sid and not ev.empty else pd.DataFrame()
        if not sub.empty and "dedupe_key" in sub.columns:
            sub = sub.drop_duplicates(subset=["dedupe_key"]).copy()
        total_count = int(sub["dedupe_key"].nunique()) if (not sub.empty and "dedupe_key" in sub.columns) else int(len(sub))
        last_dt = pd.to_datetime(sub["event_date"], errors="coerce").max() if not sub.empty else pd.NaT
        days_since = int((today - last_dt.normalize()).days) if pd.notna(last_dt) else ""
        def _count_between(lo, hi=None):
            if sub.empty:
                return 0
            d = pd.to_numeric(sub["days_after_payment"], errors="coerce")
            mask = d.ge(lo)
            if hi is not None:
                mask = mask & d.le(hi)
            return int(sub.loc[mask, "dedupe_key"].nunique()) if "dedupe_key" in sub.columns else int(mask.sum())
        type_series = sub.get("event_type", pd.Series(dtype=str)).fillna("").astype(str) if not sub.empty else pd.Series(dtype=str)
        type_lower = type_series.str.lower()
        online_master = int(sub.loc[type_lower.eq("online events & masterclasses"), "dedupe_key"].nunique()) if (not sub.empty and "dedupe_key" in sub.columns) else 0
        comp_hack = int(sub.loc[type_lower.eq("competitions & hackathons"), "dedupe_key"].nunique()) if (not sub.empty and "dedupe_key" in sub.columns) else 0
        general_fun = int(sub.loc[type_lower.eq("general/fun"), "dedupe_key"].nunique()) if (not sub.empty and "dedupe_key" in sub.columns) else 0
        activation_dt = pd.NaT
        activation_event_name = ""
        if not sub.empty:
            names = []
            sub_sorted = sub.sort_values(["event_date", "event_name"], na_position="last") if "event_date" in sub.columns else sub.copy()
            for _, e in sub_sorted.iterrows():
                nm = clean_text(e.get("event_name", ""))
                dt = pd.to_datetime(e.get("event_date", pd.NaT), errors="coerce")
                if nm:
                    names.append(f"{nm} ({dt.strftime('%d-%b-%Y') if pd.notna(dt) else '-'})")
            event_names = "; ".join(dict.fromkeys(names))
            dated_sub = sub_sorted[pd.to_datetime(sub_sorted.get("event_date", pd.Series(dtype=object)), errors="coerce").notna()].copy()
            if not dated_sub.empty:
                first_event = dated_sub.iloc[0]
                activation_dt = pd.to_datetime(first_event.get("event_date", pd.NaT), errors="coerce")
                activation_event_name = clean_text(first_event.get("event_name", ""))
        else:
            event_names = ""
        pay_dt = pd.to_datetime(stu.get("payment_date", pd.NaT), errors="coerce")
        pre_count = _retention_prepayment_activity_count_for_student(data or {}, stu, program)
        email = clean_text(stu.get("email_key", ""))
        # If the normalized email is unavailable, fall back to any raw email-ish column.
        if not email:
            for c in [col for col in stu.index if "email" in str(col).lower()]:
                email = clean_text(stu.get(c, ""))
                if email:
                    break
        rows.append({
            "Name": clean_text(stu.get("student_name", "")),
            "Email": email,
            "Payment Date": pay_dt.strftime("%d-%b-%Y") if pd.notna(pay_dt) else "",
            "Last Engaged Date": last_dt.strftime("%d-%b-%Y") if pd.notna(last_dt) else "",
            "Days passed since Engaged": days_since,
            "Activities done pre-payment": pre_count,
            "Activities done post payment": total_count,
            "0-15 days": _count_between(0, 15),
            "16-30 days": _count_between(16, 30),
            "31-45 days": _count_between(31, 45),
            "46-60 days": _count_between(46, 60),
            "60+ days": _count_between(61, None),
            "Online Event and Masterclass Attended": online_master,
            "Competition and hackathon Participated": comp_hack,
            "General/Fun/Quiz/poll/fun task Participated": general_fun,
            "Names of Events/activities they participated in": event_names,
            "Date of Activation Post-Payment": activation_dt.strftime("%d-%b-%Y") if pd.notna(activation_dt) else "",
            "Activation Event/Activity Name - Post Payment": activation_event_name,
        })
    out = pd.DataFrame(rows, columns=columns)
    return out.sort_values(["Payment Date", "Activities done post payment", "Name"], ascending=[False, False, True]).reset_index(drop=True)


def _render_retention_program_student_tab(paid_df: pd.DataFrame, events_df: pd.DataFrame, program: str, key_prefix: str, data: dict = None):
    st.markdown(f"### Retention {program}")
    st.caption("Status = Admitted students only. All activity counts are deduped post-payment Tetr-X activities after each student's payment date.")
    table = _build_retention_student_postpayment_table(paid_df, events_df, program, data=data)
    if table.empty:
        st.info(f"No admitted Tetr-X {program} students with usable payment dates were found.")
        return
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Admitted Tetr-X {program} Students", f"{len(table):,}")
    m2.metric("Active Post Payment", f"{int((pd.to_numeric(table['Activities done post payment'], errors='coerce').fillna(0) > 0).sum()):,}")
    m3.metric("Avg Post-Payment Activities", f"{pd.to_numeric(table['Activities done post payment'], errors='coerce').fillna(0).mean():.1f}")

    st.dataframe(table, use_container_width=True, hide_index=True, height=560, key=f"{key_prefix}_retention_student_table")

    dist = table.groupby("Activities done post payment", as_index=False).agg(Students=("Name", "count")).sort_values("Activities done post payment")
    fig = px.bar(dist, x="Activities done post payment", y="Students", title=f"{program} Post-Payment Activity Distribution")
    fig.update_traces(marker_color=GREEN_2)
    st.plotly_chart(nice_layout(fig, height=320), use_container_width=True, key=f"{key_prefix}_retention_distribution")


def render_true_retention_page(data):
    st.subheader("Retention")
    tabs = st.tabs(["Total", "Retention UG", "Retention PG"])
    with tabs[0]:
        _render_true_retention_overview(data)
    paid_df, events_df, occurrences_df, available_df, metrics = build_retention_analytics_v2(data)
    with tabs[1]:
        _render_retention_program_student_tab(paid_df, events_df, "UG", "retention_ug", data=data)
    with tabs[2]:
        _render_retention_program_student_tab(paid_df, events_df, "PG", "retention_pg", data=data)

def render_recent_activity_page(data):
    st.subheader("Recent Activity")
    today = get_today_ist()
    default_start = today - pd.Timedelta(days=6)
    c1, c2 = st.columns(2)
    with c1:
        start_input = st.date_input("From", value=default_start.date(), key="recent_from")
    with c2:
        end_input = st.date_input("To", value=today.date(), key="recent_to")
    start_date = pd.to_datetime(start_input, errors="coerce").normalize()
    end_date = pd.to_datetime(end_input, errors="coerce").normalize()
    if pd.isna(start_date) or pd.isna(end_date):
        start_date = default_start
        end_date = today
    if start_date > end_date:
        start_date, end_date = end_date, start_date

    recent = build_recent_activity_data(data, start_date=start_date, end_date=end_date)
    st.caption(f"Showing activity from {start_date.strftime('%d %b %Y')} to {end_date.strftime('%d %b %Y')}")

    tabs = st.tabs(["Recent Payments", "Recent Active Students", "Recent Student Activity", "Recent Inactive Students", "Recent Events"])

    with tabs[0]:
        recent_payments = recent.get("recent_payments", [])
        if not recent_payments:
            st.info("No admitted/paid students with payment dates in the selected range were found.")
        else:
            for idx, item in enumerate(recent_payments):
                st.markdown(f"#### {item['student_name']}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Program", item.get("program", ""))
                c2.metric("Batch", item.get("batch", ""))
                c3.metric("Payment Date", item['payment_date'].strftime('%d-%b-%Y') if pd.notna(item.get('payment_date', pd.NaT)) else "-")
                c4.metric("Total Participation", item.get("total_participation", 0))
                if item["events_df"].empty:
                    st.info("No attended events were found for this student.")
                else:
                    st.dataframe(item["events_df"], use_container_width=True, height=min(320, 80 + 35 * len(item["events_df"])), key=f"recentpay_{idx}")
                st.markdown("---")

    with tabs[1]:
        any_active = False
        for sheet in [s for s in (UG_BATCH_SHEETS + PG_BATCH_SHEETS) if s in recent["batch_student_recent"]]:
            df = recent["batch_student_recent"][sheet]
            if df.empty:
                continue
            active = df[df["recent_attendance_count"] > 0].copy().sort_values(["recent_attendance_count", "student_name"], ascending=[False, True])
            if active.empty:
                continue
            any_active = True
            st.markdown(f"#### {sheet}")
            st.dataframe(active[["student_name", "Program", "Batch", "recent_attendance_count", "recent_events"]], use_container_width=True, height=min(360, 80 + 35 * len(active)), key=f"recentactive_{normalize_name(sheet)}")
        if not any_active:
            st.info("No active students were found in recently done events for the selected range.")

    with tabs[2]:
        def _render_recent_student_activity_program(program_label: str):
            frames = []
            for sheet in [s for s in (UG_BATCH_SHEETS + PG_BATCH_SHEETS) if s in recent["batch_student_recent"]]:
                df = recent["batch_student_recent"].get(sheet, pd.DataFrame())
                if df is None or df.empty:
                    continue
                sub = df[df.get("Program", pd.Series("", index=df.index)).astype(str).str.upper().eq(program_label)].copy()
                if sub.empty:
                    continue
                sub = sub[pd.to_numeric(sub.get("recent_attendance_count", 0), errors="coerce").fillna(0).gt(0)].copy()
                if sub.empty:
                    continue
                frames.append(sub)
            if not frames:
                st.info(f"No active {program_label} students were found for the selected range.")
                return
            combined = pd.concat(frames, ignore_index=True)
            combined["_sid"] = combined.apply(lambda r: clean_text(r.get("email_key", "")) or normalize_name(r.get("student_name", "")), axis=1)
            rows = []
            for _, grp in combined.groupby("_sid", sort=False):
                first = grp.iloc[0]
                active_dates = []
                events = []
                for _, r in grp.iterrows():
                    ev_txt = clean_text(r.get("recent_events", ""))
                    if ev_txt:
                        events.extend([clean_text(x) for x in ev_txt.split(";") if clean_text(x)])
                    date_txt = clean_text(r.get("recent_active_dates", ""))
                    if date_txt:
                        active_dates.extend([clean_text(x) for x in date_txt.split(";") if clean_text(x)])
                unique_events = list(dict.fromkeys(events))
                parsed_dates = sorted([pd.to_datetime(d, errors="coerce", dayfirst=True) for d in set(active_dates) if pd.notna(pd.to_datetime(d, errors="coerce", dayfirst=True))])
                rows.append({
                    "Name": clean_text(first.get("student_name", "")),
                    "Email": clean_text(first.get("email_key", "")),
                    "Program": program_label,
                    "Batch": clean_text(first.get("Batch", "")),
                    "Status": clean_text(first.get("status", "")),
                    "Recent activities participated count": len(unique_events),
                    "Recent activities they participated in (with event dates)": "; ".join(unique_events),
                    "Dates they were active on": "; ".join([d.strftime("%d-%b-%Y") for d in parsed_dates]),
                    "Date of reactivation": parsed_dates[0].strftime("%d-%b-%Y") if parsed_dates else clean_text(first.get("reactivation_date", "")),
                })
            out = pd.DataFrame(rows)
            if out.empty:
                st.info(f"No active {program_label} students were found for the selected range.")
                return
            out = out.sort_values(["Recent activities participated count", "Name"], ascending=[False, True]).reset_index(drop=True)
            st.markdown(f"#### {program_label} Table")
            st.dataframe(out, use_container_width=True, hide_index=True, height=min(520, 80 + 35 * len(out)), key=f"recent_student_activity_{program_label.lower()}")

        _render_recent_student_activity_program("UG")
        _render_recent_student_activity_program("PG")

    with tabs[3]:
        any_inactive = False
        for sheet in [s for s in (UG_BATCH_SHEETS + PG_BATCH_SHEETS) if s in recent["batch_student_recent"]]:
            df = recent["batch_student_recent"][sheet]
            if df.empty:
                continue
            inactive = df[df["recent_attendance_count"] <= 0].copy().sort_values(["student_name"])
            if inactive.empty:
                continue
            any_inactive = True
            st.markdown(f"#### {sheet}")
            st.dataframe(inactive[["student_name", "Program", "Batch"]], use_container_width=True, height=min(360, 80 + 35 * len(inactive)), key=f"recentinactive_{normalize_name(sheet)}")
        if not any_inactive:
            st.info("No inactive students were found for the selected range.")

    with tabs[4]:
        recent_events_df = recent.get("recent_events_df", pd.DataFrame())
        if recent_events_df.empty:
            st.info("No recent events were found in the selected range.")
        else:
            summary = recent_events_df.groupby(["event_date", "event_name", "event_type", "sheet", "batch", "program"], as_index=False)["attendance"].sum().sort_values(["event_date", "program", "batch", "attendance"], ascending=[False, True, True, False])
            disp = summary.copy()
            disp["event_date"] = pd.to_datetime(disp["event_date"], errors="coerce").dt.strftime("%d-%b-%Y")
            st.dataframe(disp.rename(columns={"event_date": "Date", "event_name": "Event Name", "event_type": "Event Type", "sheet": "Sheet", "batch": "Batch", "program": "UG/PG", "attendance": "Total Attendance"}), use_container_width=True, height=420, key="recentevents_table")
            chart_df = summary.groupby(["batch"], as_index=False)["attendance"].sum().sort_values("attendance", ascending=False)
            fig = px.bar(chart_df, x="batch", y="attendance", title="Recent Events Attendance by Batch")
            fig.update_traces(marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=320), use_container_width=True, key="recentevents_batchchart")



# ---------------- Success Metrics ----------------

def map_success_event_bucket(event_type: str) -> str:
    """Bucket event types only for Success Metrics."""
    s = clean_text(event_type).strip().lower()
    if s in {"online event", "online ama", "online amas", "ama", "masterclass", "skill bootcamp", "bootcamp"}:
        return "Online Events + Masterclasses"
    if s in {"competition", "competitions", "hackathon", "hackerthon"}:
        return "Competition & Hackathon"
    return "Other"


def _fmt_count_pct(count, denom):
    count = int(count or 0)
    denom = int(denom or 0)
    pct = (count / denom * 100) if denom else 0.0
    return f"{count:,} ({pct:.1f}%)"


def _student_id_from_values(email_key="", student_key="", student_name=""):
    return clean_text(email_key) or clean_text(student_key) or normalize_name(student_name)


def _program_denominators_for_success(data):
    overview = data.get("overview_df", pd.DataFrame())
    out = {"Total": set(), "UG": set(), "PG": set()}
    if overview is None or overview.empty:
        return out
    for _, r in overview.iterrows():
        sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("student_name", ""))
        if not sid:
            continue
        program = clean_text(r.get("Program", ""))
        out["Total"].add(sid)
        if program in {"UG", "PG"}:
            out[program].add(sid)
    return out


def _paid_students_from_tetrx_for_success(data):
    rows = []
    activities = data.get("activities", {})
    for tx_sheet in TX_SHEETS:
        tx_df = activities.get(tx_sheet, pd.DataFrame())
        if tx_df is None or tx_df.empty:
            continue
        program = "UG" if tx_sheet.endswith("UG") else "PG"
        frame = tx_df.copy()
        if "sheet_is_paid" in frame.columns:
            frame = frame[frame["sheet_is_paid"].fillna(False).astype(bool)].copy()
        elif "sheet_status_raw" in frame.columns:
            frame = frame[frame["sheet_status_raw"].astype(str).str.strip().str.lower().eq("admitted")].copy()
        else:
            continue
        for _, r in frame.iterrows():
            sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("student_name", ""))
            if not sid:
                continue
            pay_dt = pd.to_datetime(r.get("payment_date_parsed", pd.NaT), errors="coerce")
            rows.append({
                "student_id": sid,
                "student_name": clean_text(r.get("student_name", "")),
                "program": program,
                "payment_date": pay_dt.normalize() if pd.notna(pay_dt) else pd.NaT,
            })
    paid = pd.DataFrame(rows)
    if paid.empty:
        return pd.DataFrame(columns=["student_id", "student_name", "program", "payment_date"])
    paid = paid.sort_values(["payment_date", "student_name"], na_position="last").drop_duplicates("student_id", keep="first").reset_index(drop=True)
    return paid


def _winner_ids_for_success(data):
    winner_df = data.get("winner_df", pd.DataFrame())
    if winner_df is None or winner_df.empty:
        return set()
    w = winner_df.copy()
    if "is_winner" in w.columns:
        w = w[w["is_winner"].fillna(False).astype(bool)].copy()
    ids = set()
    for _, r in w.iterrows():
        sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("winner_name", ""))
        if sid:
            ids.add(sid)
    return ids


def _success_activity_events_from_batch_sheets(data):
    """Return attended student-event rows and deduped event occurrences from UG/PG batch sheets only."""
    activities = data.get("activities", {})
    activity_ctx = data.get("activity_ctx", {})
    attendee_rows = []
    occurrence_rows = []
    for sheet in UG_BATCH_SHEETS + PG_BATCH_SHEETS:
        df = activities.get(sheet, pd.DataFrame())
        ctx = activity_ctx.get(sheet, {})
        if df is None or df.empty or not ctx:
            continue
        event_info = ctx.get("event_info", pd.DataFrame())
        if event_info is None or event_info.empty:
            continue
        program = infer_program_from_sheet(sheet)
        batch = infer_batch_group_from_sheet_name(sheet)
        for _, ev in event_info.iterrows():
            col = ev.get("column_name")
            if not col or col not in df.columns:
                continue
            ev_type_raw = clean_text(ev.get("event_type", "")) or "Other"
            bucket = map_success_event_bucket(ev_type_raw)
            if bucket == "Other":
                continue
            ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
            if pd.isna(ev_date):
                continue
            ev_date = ev_date.normalize()
            ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
            occurrence_key = "|".join([program, bucket.lower(), normalize_name(ev_name), ev_date.strftime("%Y-%m-%d")])
            occurrence_rows.append({
                "program": program,
                "batch": batch,
                "sheet": sheet,
                "event_name": ev_name,
                "event_type": bucket,
                "event_type_raw": ev_type_raw,
                "event_date": ev_date,
                "occurrence_key": occurrence_key,
            })
            attended_mask = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
            if not attended_mask.any():
                continue
            for _, r in df.loc[attended_mask].iterrows():
                sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("student_name", ""))
                if not sid:
                    continue
                dedupe_key = "|".join([sid, occurrence_key])
                attendee_rows.append({
                    "student_id": sid,
                    "student_name": clean_text(r.get("student_name", "")),
                    "program": program,
                    "batch": clean_text(r.get("Batch", "")) or batch,
                    "sheet": sheet,
                    "event_name": ev_name,
                    "event_type": bucket,
                    "event_type_raw": ev_type_raw,
                    "event_date": ev_date,
                    "occurrence_key": occurrence_key,
                    "dedupe_key": dedupe_key,
                })
    attendees = pd.DataFrame(attendee_rows)
    if not attendees.empty:
        attendees = attendees.drop_duplicates("dedupe_key").reset_index(drop=True)
    else:
        attendees = pd.DataFrame(columns=["student_id", "student_name", "program", "batch", "sheet", "event_name", "event_type", "event_type_raw", "event_date", "occurrence_key", "dedupe_key"])
    occurrences = pd.DataFrame(occurrence_rows)
    if not occurrences.empty:
        occurrences = occurrences.drop_duplicates("occurrence_key").reset_index(drop=True)
    else:
        occurrences = pd.DataFrame(columns=["program", "batch", "sheet", "event_name", "event_type", "event_type_raw", "event_date", "occurrence_key"])
    return attendees, occurrences


def _count_by_group(ids_by_program, denominators_by_program=None):
    rows = []
    for group in ["Total", "UG", "PG"]:
        ids = set(ids_by_program.get(group, set()))
        denom = len(set(denominators_by_program.get(group, set()))) if denominators_by_program else 0
        rows.append({"Group": group, "Count": len(ids), "%": (len(ids) / denom * 100) if denom else 0.0})
    return pd.DataFrame(rows)


def _group_sets_from_frame(frame: pd.DataFrame, id_col="student_id"):
    out = {"Total": set(), "UG": set(), "PG": set()}
    if frame is None or frame.empty or id_col not in frame.columns:
        return out
    for _, r in frame.iterrows():
        sid = clean_text(r.get(id_col, ""))
        if not sid:
            continue
        program = clean_text(r.get("program", ""))
        out["Total"].add(sid)
        if program in {"UG", "PG"}:
            out[program].add(sid)
    return out


def _pct_value(count, denom):
    count = int(count or 0)
    denom = int(denom or 0)
    return round((count / denom * 100), 1) if denom else 0.0


def _group_counts_from_sets(ids_by_group):
    return {g: len(set(ids_by_group.get(g, set()))) for g in ["Total", "UG", "PG"]}


def _share_denominator_from_metric(ids_by_group):
    """Use total metric count as denominator, useful for UG/PG split of event occurrences."""
    total = len(set(ids_by_group.get("Total", set())))
    return {"Total": set(range(total)), "UG": set(range(total)), "PG": set(range(total))}


def _success_metric_table(metrics, denominators):
    """
    Standard Success Metrics table.
    Every row shows separate Total/UG/PG counts and percentages.

    Denominator modes:
      - students: all unique students by group
      - paid: all paid/Tetr-X admitted students by group
      - share_total: percentage contribution to that row's Total count
      - any custom key added to denominators, e.g. comp_paid_attendees
    """
    rows = []
    default_empty = {"Total": set(), "UG": set(), "PG": set()}
    for label, ids_by_group, denom_mode in metrics:
        ids_by_group = ids_by_group or default_empty
        counts = _group_counts_from_sets(ids_by_group)

        if denom_mode == "share_total":
            total_denom = max(counts.get("Total", 0), 0)
            denom_counts = {"Total": total_denom, "UG": total_denom, "PG": total_denom}
        else:
            denom_sets = denominators.get(denom_mode, denominators.get("students", default_empty))
            denom_counts = {g: len(set(denom_sets.get(g, set()))) for g in ["Total", "UG", "PG"]}

        rows.append({
            "Metric": label,
            "Total": counts["Total"],
            "Total %": _pct_value(counts["Total"], denom_counts["Total"]),
            "UG": counts["UG"],
            "UG %": _pct_value(counts["UG"], denom_counts["UG"]),
            "PG": counts["PG"],
            "PG %": _pct_value(counts["PG"], denom_counts["PG"]),
        })
    return pd.DataFrame(rows)


def build_success_metrics_data(data):
    student_denoms = _program_denominators_for_success(data)
    paid_df = _paid_students_from_tetrx_for_success(data)
    paid_sets = _group_sets_from_frame(paid_df)
    paid_lookup = paid_df.set_index("student_id")["payment_date"].to_dict() if not paid_df.empty else {}
    winner_ids = _winner_ids_for_success(data)
    attendees, occurrences = _success_activity_events_from_batch_sheets(data)

    if not attendees.empty:
        attendees = attendees.copy()
        attendees["payment_date"] = attendees["student_id"].map(paid_lookup)
        attendees["is_paid"] = attendees["student_id"].isin(set(paid_lookup.keys()))
        attendees["event_date"] = pd.to_datetime(attendees["event_date"], errors="coerce").dt.normalize()
        attendees["payment_date"] = pd.to_datetime(attendees["payment_date"], errors="coerce").dt.normalize()
        attendees["before_payment"] = attendees["is_paid"] & attendees["payment_date"].notna() & (attendees["event_date"] <= attendees["payment_date"])
        attendees["before_7_payment"] = attendees["is_paid"] & attendees["payment_date"].notna() & attendees["event_date"].between(attendees["payment_date"] - pd.Timedelta(days=7), attendees["payment_date"], inclusive="both")
        attendees["is_winner"] = attendees["student_id"].isin(winner_ids)

    return {
        "student_denoms": student_denoms,
        "paid_df": paid_df,
        "paid_sets": paid_sets,
        "winner_ids": winner_ids,
        "attendees": attendees,
        "occurrences": occurrences,
    }


def _render_success_summary_table(title, rows, key):
    st.markdown(f"#### {title}")
    table = pd.DataFrame(rows)
    if table.empty:
        st.info("No matching data found for this section.")
        return
    st.dataframe(table, use_container_width=True, hide_index=True, key=key)




def _winner_challenge_lookup(data):
    winner_df = data.get("winner_df", pd.DataFrame())
    out = {}
    if winner_df is None or winner_df.empty:
        return out
    w = winner_df[winner_df.get("is_winner", False).fillna(False).astype(bool)].copy() if "is_winner" in winner_df.columns else winner_df.copy()
    for _, r in w.iterrows():
        sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("winner_name", ""))
        if not sid:
            continue
        out.setdefault(sid, set()).add(clean_text(r.get("challenge_name", "")))
    return {k: sorted([x for x in v if x]) for k, v in out.items()}


def render_success_comp_t7_student_list(comp_before7: pd.DataFrame, data):
    st.markdown("##### Paid Students Attended Competition in T-7 to T · Student List")
    if comp_before7 is None or comp_before7.empty:
        st.info("No paid students attended Competition/Hackathon in T-7 to T.")
        return
    win_lookup = _winner_challenge_lookup(data)
    rows = []
    for sid, g in comp_before7.groupby("student_id"):
        event_names = sorted(dict.fromkeys([clean_text(x) for x in g.get("event_name", pd.Series(dtype=str)).tolist() if clean_text(x)]))
        winner_challenges = win_lookup.get(sid, [])
        first = g.iloc[0]
        rows.append({
            "Student Name": clean_text(first.get("student_name", "")),
            "UG/PG": clean_text(first.get("program", "")),
            "Batch": clean_text(first.get("batch", "")),
            "Events Attended in T-7 to T": len(set(g.get("dedupe_key", pd.Series(dtype=str)).astype(str).tolist())),
            "Event Names": ", ".join(event_names),
            "Winner Status": "Won" if winner_challenges else "Not Won",
            "Winner Challenge(s)": ", ".join(winner_challenges),
        })
    out = pd.DataFrame(rows).sort_values(["Winner Status", "Events Attended in T-7 to T", "Student Name"], ascending=[True, False, True])
    st.dataframe(out, use_container_width=True, hide_index=True, height=300, key="success_comp_t7_student_list")



def _success_report_activity_bucket(event_type: str) -> str:
    """Event buckets for Success Metrics -> Report only."""
    typ = clean_text(event_type).strip().lower()
    if typ in {"online event", "online ama", "online amas", "ama", "masterclass", "skill bootcamp", "bootcamp"}:
        return "Online Events + Masterclasses"
    if typ in {"competition", "competitions", "hackathon", "hackerthon"}:
        return "Competition & Hackathon"
    if typ in {"general", "fun", "fun task", "quiz", "poll"}:
        return "General/Fun"
    return "Other"


def _success_report_is_joined_wa_value(x) -> bool:
    s = clean_text(x).strip().lower().replace("-", " ")
    s = " ".join(s.split())
    return s in {"in", "tetrx", "tetr x", "added to term 0", "left"}


def _success_report_student_base(data):
    """One row per Master UG/PG student, enriched with Dates sheet course/offer data."""
    overview = data.get("overview_df", pd.DataFrame())
    dates_df = data.get("dates_df", pd.DataFrame())
    cols = [
        "student_id", "student_name", "email_key", "program", "batch", "batch_key",
        "course", "offered_date", "community_raw"
    ]
    if overview is None or overview.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, r in overview.iterrows():
        sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("student_name", ""))
        if not sid:
            continue
        program = clean_text(r.get("Program", "")).upper()
        batch = clean_text(r.get("Batch", ""))
        offered = pd.to_datetime(r.get("offered_date_parsed", pd.NaT), errors="coerce")
        course = ""
        drow = None
        if dates_df is not None and not dates_df.empty:
            drow = find_student_dates_row(
                dates_df,
                r.get("student_name", ""),
                r.get("email_key", ""),
                r.get("student_key", ""),
                program,
                batch,
            )
        if drow is not None:
            course = clean_text(drow.get("Course", ""))
            if pd.isna(offered):
                offered = pd.to_datetime(drow.get("offered_date_parsed", pd.NaT), errors="coerce")
            if not batch:
                batch = clean_text(drow.get("Batch", ""))
        rows.append({
            "student_id": sid,
            "student_name": clean_text(r.get("student_name", "")),
            "email_key": clean_text(r.get("email_key", "")),
            "program": program,
            "batch": batch,
            "batch_key": normalize_batch_token(batch),
            "course": course,
            "offered_date": offered.normalize() if pd.notna(offered) else pd.NaT,
            "community_raw": clean_text(r.get("admitted_group_batch_onwards_raw", r.get("community_status_value", ""))),
        })

    base = pd.DataFrame(rows, columns=cols)
    if base.empty:
        return base
    base = base.drop_duplicates("student_id", keep="first").reset_index(drop=True)
    return base


def _success_report_payment_rows(data):
    """Tetr-X rows with payment dates. For Report: paid = has payment date in Tetr-X."""
    rows = []
    activities = data.get("activities", {})
    for tx_sheet in TX_SHEETS:
        df = activities.get(tx_sheet, pd.DataFrame())
        if df is None or df.empty:
            continue
        program = "UG" if tx_sheet.endswith("UG") else "PG"
        for _, r in df.iterrows():
            sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("student_name", ""))
            if not sid:
                continue
            pay_dt = pd.to_datetime(r.get("payment_date_parsed", pd.NaT), errors="coerce")
            if pd.isna(pay_dt):
                continue
            status = clean_text(r.get("sheet_status_raw", ""))
            rows.append({
                "student_id": sid,
                "program": program,
                "student_name": clean_text(r.get("student_name", "")),
                "status": status,
                "status_lower": status.lower().strip(),
                "payment_date": pay_dt.normalize(),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=["student_id", "program", "student_name", "status", "status_lower", "payment_date"])
    # If duplicated, keep earliest payment date but preserve a non-empty/current status where possible.
    out = out.sort_values(["payment_date", "student_name"], na_position="last").drop_duplicates("student_id", keep="first").reset_index(drop=True)
    return out


def _success_report_all_activity_attendees(data):
    """All attended student-event rows across batch + Tetr-X sheets for the Success Report."""
    activities = data.get("activities", {})
    activity_ctx = data.get("activity_ctx", {})
    rows = []
    for sheet in UG_BATCH_SHEETS + PG_BATCH_SHEETS + TX_SHEETS:
        df = activities.get(sheet, pd.DataFrame())
        ctx = activity_ctx.get(sheet, {})
        if df is None or df.empty or not ctx:
            continue
        event_info = ctx.get("event_info", pd.DataFrame())
        if event_info is None or event_info.empty:
            continue
        program = infer_program_from_sheet(sheet)
        sheet_batch = infer_batch_group_from_sheet_name(sheet)
        for _, ev in event_info.iterrows():
            col = ev.get("column_name")
            if not col or col not in df.columns:
                continue
            ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
            if pd.isna(ev_date):
                continue
            ev_date = ev_date.normalize()
            ev_type_raw = clean_text(ev.get("event_type", "")) or "Other"
            bucket = _success_report_activity_bucket(ev_type_raw)
            ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
            event_key = "|".join([program, normalize_name(ev_name), bucket.lower(), ev_date.strftime("%Y-%m-%d")])
            attended_mask = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
            if not attended_mask.any():
                continue
            for _, r in df.loc[attended_mask].iterrows():
                sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("student_name", ""))
                if not sid:
                    continue
                rows.append({
                    "student_id": sid,
                    "student_name": clean_text(r.get("student_name", "")),
                    "program": program,
                    "batch": clean_text(r.get("Batch", "")) or sheet_batch,
                    "source_sheet": sheet,
                    "event_name": ev_name,
                    "event_type_raw": ev_type_raw,
                    "event_bucket": bucket,
                    "event_date": ev_date,
                    "event_key": event_key,
                    "dedupe_key": "|".join([sid, event_key]),
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=[
            "student_id", "student_name", "program", "batch", "source_sheet", "event_name",
            "event_type_raw", "event_bucket", "event_date", "event_key", "dedupe_key"
        ])
    return out.drop_duplicates("dedupe_key").reset_index(drop=True)


def _success_report_winner_rows(data):
    winner_df = data.get("winner_df", pd.DataFrame())
    if winner_df is None or winner_df.empty:
        return pd.DataFrame(columns=["student_id", "program", "announcement_date"])
    rows = []
    w = winner_df.copy()
    if "is_winner" in w.columns or "is_spotlight" in w.columns:
        w = w[w.get("is_winner", False).fillna(False).astype(bool) | w.get("is_spotlight", False).fillna(False).astype(bool)].copy()
    for _, r in w.iterrows():
        sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("winner_name", ""))
        if not sid:
            continue
        ann = pd.to_datetime(r.get("announcement_date", pd.NaT), errors="coerce")
        rows.append({
            "student_id": sid,
            "announcement_date": ann.normalize() if pd.notna(ann) else pd.NaT,
            "challenge_name": clean_text(r.get("challenge_name", "")),
            "entry_type": clean_text(r.get("entry_type", "")),
        })
    return pd.DataFrame(rows)


def _success_report_group_sets(base: pd.DataFrame, ids: set):
    out = {"Total": set(), "UG": set(), "PG": set()}
    if base is None or base.empty:
        return out
    ids = set(ids or set())
    sub = base[base["student_id"].isin(ids)].copy()
    for _, r in sub.iterrows():
        sid = clean_text(r.get("student_id", ""))
        program = clean_text(r.get("program", ""))
        if not sid:
            continue
        out["Total"].add(sid)
        if program in {"UG", "PG"}:
            out[program].add(sid)
    return out


def _success_report_table_from_metrics(metric_sets, offered_sets, visible_groups):
    rows = []
    for metric_name, ids_by_group in metric_sets:
        row = {"Metric": metric_name}
        for group in visible_groups:
            cnt = len(set(ids_by_group.get(group, set())))
            denom = len(set(offered_sets.get(group, set())))
            row[group] = cnt
            row[f"{group} %"] = round((cnt / denom * 100), 1) if denom else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def render_success_metrics_report(data):
    st.markdown("### Report")
    st.caption("Report metrics are calculated as of the selected date. Percentages use Offered Students in the selected filters as the denominator.")

    base_all = _success_report_student_base(data)
    if base_all.empty:
        st.info("No Master UG / Master PG students found for this report.")
        return

    activities_all = _success_report_all_activity_attendees(data)
    payments_all = _success_report_payment_rows(data)
    winners_all = _success_report_winner_rows(data)

    date_candidates = []
    for series in [base_all.get("offered_date"), activities_all.get("event_date"), payments_all.get("payment_date"), winners_all.get("announcement_date")]:
        if series is not None:
            vals = pd.to_datetime(series, errors="coerce").dropna()
            if not vals.empty:
                date_candidates.append(vals.max())
    default_as_of = max(date_candidates).date() if date_candidates else pd.Timestamp.today().date()

    batch_options = sorted([x for x in base_all["batch"].map(clean_text).dropna().unique().tolist() if x])
    course_options = sorted([x for x in base_all["course"].map(clean_text).dropna().unique().tolist() if x])

    applied_key = "success_report_applied_filters"
    default_filters = {
        "as_of_date": default_as_of,
        "view_choice": "Total + UG + PG",
        "selected_batches": [],
        "selected_courses": [],
    }
    if applied_key not in st.session_state:
        st.session_state[applied_key] = default_filters.copy()
    applied_filters = st.session_state.get(applied_key, default_filters.copy()) or default_filters.copy()

    current_as_of = applied_filters.get("as_of_date", default_as_of)
    try:
        current_as_of = pd.to_datetime(current_as_of).date()
    except Exception:
        current_as_of = default_as_of
    current_view = applied_filters.get("view_choice", "Total + UG + PG")
    if current_view not in {"Total + UG + PG", "UG", "PG"}:
        current_view = "Total + UG + PG"
    current_batches = [x for x in applied_filters.get("selected_batches", []) if x in batch_options]
    current_courses = [x for x in applied_filters.get("selected_courses", []) if x in course_options]

    with st.form("success_report_filters_form"):
        f1, f2, f3, f4, f5 = st.columns([1.05, 1.05, 1.25, 1.25, 0.7])
        with f1:
            pending_as_of_date = st.date_input("Report as of date", value=current_as_of, key="success_report_as_of_date_pending")
        with f2:
            pending_view_choice = st.radio("UG / PG view", ["Total + UG + PG", "UG", "PG"], index=["Total + UG + PG", "UG", "PG"].index(current_view), horizontal=False, key="success_report_program_view_pending")
        with f3:
            pending_batches = st.multiselect("Batches (blank = all)", batch_options, default=current_batches, key="success_report_batches_pending")
        with f4:
            pending_courses = st.multiselect("Courses (blank = all)", course_options, default=current_courses, key="success_report_courses_pending")
        with f5:
            st.markdown("<div style='height: 1.85rem;'></div>", unsafe_allow_html=True)
            submitted_filters = st.form_submit_button("Show", use_container_width=True)

    if submitted_filters:
        st.session_state[applied_key] = {
            "as_of_date": pending_as_of_date,
            "view_choice": pending_view_choice,
            "selected_batches": list(pending_batches or []),
            "selected_courses": list(pending_courses or []),
        }

    applied_filters = st.session_state.get(applied_key, default_filters.copy()) or default_filters.copy()
    as_of_date = applied_filters.get("as_of_date", default_as_of)
    as_of = pd.to_datetime(as_of_date).normalize()
    view_choice = applied_filters.get("view_choice", "Total + UG + PG")
    if view_choice not in {"Total + UG + PG", "UG", "PG"}:
        view_choice = "Total + UG + PG"
    selected_batches = [x for x in applied_filters.get("selected_batches", []) if x in batch_options]
    selected_courses = [x for x in applied_filters.get("selected_courses", []) if x in course_options]

    active_filter_bits = [f"As of: {pd.to_datetime(as_of).strftime('%d-%b-%Y')}", f"View: {view_choice}"]
    if selected_batches:
        active_filter_bits.append("Batches: " + ", ".join(selected_batches))
    if selected_courses:
        active_filter_bits.append("Courses: " + ", ".join(selected_courses))
    st.caption("Applied filters · " + " | ".join(active_filter_bits))

    base = base_all.copy()
    if view_choice in {"UG", "PG"}:
        base = base[base["program"].eq(view_choice)].copy()
    if selected_batches:
        wanted_batches = set(clean_text(x) for x in selected_batches)
        wanted_batch_keys = set(normalize_batch_token(x) for x in selected_batches)
        base = base[base["batch"].map(clean_text).isin(wanted_batches) | base["batch_key"].isin(wanted_batch_keys)].copy()
    if selected_courses:
        wanted_courses = set(clean_text(x) for x in selected_courses)
        base = base[base["course"].map(clean_text).isin(wanted_courses)].copy()

    # Offered-as-of denominator. Offered date comes from Dates sheet; rows without offered date are only kept for current/latest views.
    offered_mask = pd.to_datetime(base["offered_date"], errors="coerce").notna() & (pd.to_datetime(base["offered_date"], errors="coerce") <= as_of)
    base = base.loc[offered_mask].copy()
    if base.empty:
        st.info("No offered students match the selected date/filter combination.")
        return

    base_ids = set(base["student_id"].astype(str))
    offered_sets = _success_report_group_sets(base, base_ids)

    acts = activities_all.copy()
    if not acts.empty:
        acts["event_date"] = pd.to_datetime(acts["event_date"], errors="coerce").dt.normalize()
        acts = acts[acts["student_id"].isin(base_ids) & acts["event_date"].notna() & (acts["event_date"] <= as_of)].copy()

    payments = payments_all.copy()
    if not payments.empty:
        payments["payment_date"] = pd.to_datetime(payments["payment_date"], errors="coerce").dt.normalize()
        payments = payments[payments["student_id"].isin(base_ids) & payments["payment_date"].notna() & (payments["payment_date"] <= as_of)].copy()
    paid_ids = set(payments["student_id"].astype(str).tolist()) if not payments.empty else set()

    refund_ids = set()
    final_admitted_ids = set()
    if not payments.empty:
        pay_status = payments["status_lower"].astype(str)
        refund_mask = pay_status.str.contains("refund", na=False)
        refund_ids = set(payments.loc[refund_mask, "student_id"].astype(str).tolist())
        prog = payments["program"].astype(str).str.upper()
        admitted_exact = pay_status.str.strip().eq("admitted")
        ug_final = prog.eq("UG") & (admitted_exact | pay_status.str.contains("deferral", na=False))
        pg_final = prog.eq("PG") & (admitted_exact | pay_status.str.contains(r"admitted\s*:\s*deferral", regex=True, na=False))
        final_mask = (ug_final | pg_final) & (~refund_mask)
        final_admitted_ids = set(payments.loc[final_mask, "student_id"].astype(str).tolist())

    joined_ids = set(base.loc[base["community_raw"].map(_success_report_is_joined_wa_value).fillna(False).astype(bool), "student_id"].astype(str).tolist())
    active_ids = set(acts["student_id"].astype(str).tolist()) if not acts.empty else set()
    general_ids = set(acts.loc[acts["event_bucket"].eq("General/Fun"), "student_id"].astype(str).tolist()) if not acts.empty else set()
    event_ids = set(acts.loc[acts["event_bucket"].eq("Online Events + Masterclasses"), "student_id"].astype(str).tolist()) if not acts.empty else set()
    comp_ids = set(acts.loc[acts["event_bucket"].eq("Competition & Hackathon"), "student_id"].astype(str).tolist()) if not acts.empty else set()

    win_ids = set()
    if not winners_all.empty:
        wins = winners_all.copy()
        wins["announcement_date"] = pd.to_datetime(wins["announcement_date"], errors="coerce").dt.normalize()
        wins = wins[wins["student_id"].isin(base_ids)].copy()
        # With a report date selected, count only Winner/Spotlight announcements on or before that date.
        wins = wins[wins["announcement_date"].notna() & (wins["announcement_date"] <= as_of)].copy()
        win_ids = set(wins["student_id"].astype(str).tolist()) if not wins.empty else set()

    # First-30-days engagement bands after offer date, capped by report date.
    eng_counts = pd.Series(dtype="int64")
    if not acts.empty:
        offer_lookup = base.set_index("student_id")["offered_date"].to_dict()
        tmp = acts.copy()
        tmp["offered_date"] = pd.to_datetime(tmp["student_id"].map(offer_lookup), errors="coerce").dt.normalize()
        tmp = tmp[tmp["offered_date"].notna()].copy()
        tmp = tmp[(tmp["event_date"] >= tmp["offered_date"]) & (tmp["event_date"] <= tmp["offered_date"] + pd.Timedelta(days=30))].copy()
        if not tmp.empty:
            eng_counts = tmp.groupby("student_id")["dedupe_key"].nunique()
    low_ids = set(eng_counts[(eng_counts >= 1) & (eng_counts <= 3)].index.astype(str).tolist()) if not eng_counts.empty else set()
    med_ids = set(eng_counts[(eng_counts >= 4) & (eng_counts <= 7)].index.astype(str).tolist()) if not eng_counts.empty else set()
    high_ids = set(eng_counts[eng_counts >= 8].index.astype(str).tolist()) if not eng_counts.empty else set()
    med_high_ids = med_ids | high_ids

    def gs(ids):
        return _success_report_group_sets(base, set(ids))

    metric_sets = [
        ("Offered", offered_sets),
        ("Onboarded (WA)", gs(joined_ids)),
        ("Activated (1+ touchpoint)", gs(active_ids)),
        ("Medium and high engaged (4+ touchpoints)", gs(med_high_ids)),
        ("Paid", gs(paid_ids)),
        ("Refund", gs(refund_ids)),
        ("Final admitted", gs(final_admitted_ids)),
        ("Joined WhatsApp", gs(joined_ids)),
        ("General activated (at least 1 touchpoint)", gs(general_ids)),
        ("General activated and paid", gs(general_ids & paid_ids)),
        ("Events activated (at least 1 online event)", gs(event_ids)),
        ("Events activated and paid", gs(event_ids & paid_ids)),
        ("Competitions (at least 1 competition)", gs(comp_ids)),
        ("Competitions activated and paid", gs(comp_ids & paid_ids)),
        ("Winner/Spotlight activated (at least 1 winner/spotlight)", gs(win_ids)),
        ("Winner/spotlight and paid", gs(win_ids & paid_ids)),
        ("Total engagement = {1,3} - Low engagement", gs(low_ids)),
        ("Low engagement and paid", gs(low_ids & paid_ids)),
        ("Total engagement = {4,7} - Medium engagement", gs(med_ids)),
        ("Medium engagement and paid", gs(med_ids & paid_ids)),
        ("Total engagement = {8,+} - High engagement", gs(high_ids)),
        ("High engagement and paid", gs(high_ids & paid_ids)),
        ("Overall paid", gs(paid_ids)),
    ]

    visible_groups = ["Total", "UG", "PG"] if view_choice == "Total + UG + PG" else [view_choice]
    report_table = _success_report_table_from_metrics(metric_sets, offered_sets, visible_groups)
    st.dataframe(report_table, use_container_width=True, hide_index=True, height=820, key="success_report_table")

    with st.expander("Report definitions used"):
        st.markdown(
            """
- **Offered denominator:** Master UG + Master PG students with Offered Date on/before the selected report date.
- **Joined WA / Onboarded / In Community:** `In`, `TetrX`, `Tetr X`, `Added to Term 0`, `Left` from `Admitted Group (Batch onwards)`.
- **Paid / Overall paid:** student has a payment date in Tetr-X UG/PG on/before the selected report date.
- **Final admitted:** UG = Admitted or status contains Deferral; PG = Admitted or status contains Admitted: Deferral; refunds excluded.
- **Events activated:** Online Events + Masterclasses. **Competitions:** Competition + Hackathon. **General activated:** General/Fun/Quiz/Poll/Fun Task.
- **Low/Medium/High engagement:** touchpoints in the first 30 days after offer date, capped by the selected report date.
            """
        )

def render_success_metrics_page(data):
    st.subheader("Success Metrics")
    st.caption("All event/activity attendance is deduped by same student + same event name + same event type + same date. Event occurrences are deduped by program + event name + event type + date, so duplicated events across batch sheets count once.")
    success_summary_tab, success_report_tab = st.tabs(["Summary", "Report"])
    with success_report_tab:
        render_success_metrics_report(data)
    with success_summary_tab:
        sm = build_success_metrics_data(data)
        attendees = sm["attendees"]
        occurrences = sm["occurrences"]
        denoms = {"students": sm["student_denoms"], "paid": sm["paid_sets"]}

        total_students = len(sm["student_denoms"].get("Total", set()))
        total_paid = len(sm["paid_sets"].get("Total", set()))
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Unique Students", f"{total_students:,}")
        c2.metric("Paid / Admitted Students", f"{total_paid:,}")
        c3.metric("Winner Students", f"{len(sm['winner_ids']):,}")

        st.markdown("---")
        st.markdown("### Online Events + Masterclasses")
        online_occ = occurrences[occurrences["event_type"].eq("Online Events + Masterclasses")].copy() if not occurrences.empty else pd.DataFrame()
        online_att = attendees[attendees["event_type"].eq("Online Events + Masterclasses")].copy() if not attendees.empty else pd.DataFrame()

        occ_by_group = {"Total": set(online_occ.get("occurrence_key", pd.Series(dtype=str)).astype(str).tolist()) if not online_occ.empty else set(),
                        "UG": set(online_occ.loc[online_occ.get("program", pd.Series(dtype=str)).eq("UG"), "occurrence_key"].astype(str).tolist()) if not online_occ.empty else set(),
                        "PG": set(online_occ.loc[online_occ.get("program", pd.Series(dtype=str)).eq("PG"), "occurrence_key"].astype(str).tolist()) if not online_occ.empty else set()}
        online_att_sets = _group_sets_from_frame(online_att)
        online_paid_att = online_att[online_att.get("is_paid", False).fillna(False).astype(bool)].copy() if not online_att.empty else pd.DataFrame()
        online_paid_att_sets = _group_sets_from_frame(online_paid_att)
        online_before_payment = online_att[online_att.get("before_payment", False).fillna(False).astype(bool)].copy() if not online_att.empty else pd.DataFrame()
        online_before_payment_sets = _group_sets_from_frame(online_before_payment)
        online_before7 = online_att[online_att.get("before_7_payment", False).fillna(False).astype(bool)].copy() if not online_att.empty else pd.DataFrame()
        online_before7_sets = _group_sets_from_frame(online_before7)

        metric_rows = [
            ("Events / Masterclasses Done", occ_by_group, "share_total"),
            ("Total Unique Students Attended", online_att_sets, "students"),
            ("At least 1 Before Payment", online_before_payment_sets, "paid"),
            ("Paid Students Attended", online_paid_att_sets, "paid"),
            ("Paid Students Attended in T-7 to T", online_before7_sets, "paid"),
        ]
        st.dataframe(_success_metric_table(metric_rows, denoms), use_container_width=True, hide_index=True, key="success_online_summary")

        # at least 1 / 3 / 5 online events before payment
        thresholds = []
        if not online_before_payment.empty:
            per_student = online_before_payment.groupby(["student_id", "program"])["dedupe_key"].nunique().reset_index(name="attended_count")
            for th in [1, 3, 5]:
                sub = per_student[per_student["attended_count"] >= th]
                thresholds.append((f"Paid Students with ≥{th} Online/Masterclass Before Payment", _group_sets_from_frame(sub), "paid"))
        if thresholds:
            st.markdown("##### Before-payment attendance depth")
            st.dataframe(_success_metric_table(thresholds, denoms), use_container_width=True, hide_index=True, key="success_online_thresholds")

        if not online_occ.empty:
            chart_df = online_occ.groupby(["program", "event_type"], as_index=False)["occurrence_key"].nunique().rename(columns={"occurrence_key": "Event Occurrences"})
            fig = px.bar(chart_df, x="program", y="Event Occurrences", color="program", title="Online Events + Masterclasses Done by Program")
            st.plotly_chart(nice_layout(fig, height=320), use_container_width=True, key="success_online_occ_chart")

        st.markdown("---")
        st.markdown("### Competition & Hackathon")
        comp_occ = occurrences[occurrences["event_type"].eq("Competition & Hackathon")].copy() if not occurrences.empty else pd.DataFrame()
        comp_att = attendees[attendees["event_type"].eq("Competition & Hackathon")].copy() if not attendees.empty else pd.DataFrame()
        comp_att_sets = _group_sets_from_frame(comp_att)
        comp_paid_att = comp_att[comp_att.get("is_paid", False).fillna(False).astype(bool)].copy() if not comp_att.empty else pd.DataFrame()
        comp_paid_att_sets = _group_sets_from_frame(comp_paid_att)
        comp_paid_winners = comp_paid_att[comp_paid_att.get("is_winner", False).fillna(False).astype(bool)].copy() if not comp_paid_att.empty else pd.DataFrame()
        comp_paid_winner_sets = _group_sets_from_frame(comp_paid_winners)
        comp_before7 = comp_att[comp_att.get("before_7_payment", False).fillna(False).astype(bool)].copy() if not comp_att.empty else pd.DataFrame()
        comp_before7_sets = _group_sets_from_frame(comp_before7)
        comp_before7_winners = comp_before7[comp_before7.get("is_winner", False).fillna(False).astype(bool)].copy() if not comp_before7.empty else pd.DataFrame()
        comp_before7_winner_sets = _group_sets_from_frame(comp_before7_winners)
        denoms["comp_paid_attendees"] = comp_paid_att_sets
        denoms["comp_before7_attendees"] = comp_before7_sets
        comp_rows = [
            ("Unique Students Participated", comp_att_sets, "students"),
            ("Unique Students Participated and Paid", comp_paid_att_sets, "paid"),
            ("Paid Students Participated and Won", comp_paid_winner_sets, "comp_paid_attendees"),
            ("Paid Students Attended Competition in T-7 to T", comp_before7_sets, "paid"),
            ("Paid Students Attended and Won Competition in T-7 to T", comp_before7_winner_sets, "comp_before7_attendees"),
        ]
        st.dataframe(_success_metric_table(comp_rows, denoms), use_container_width=True, hide_index=True, key="success_comp_summary")
        render_success_comp_t7_student_list(comp_before7, data)

        # Before-payment attendance depth for Competition & Hackathon.
        # "Before payment" means all attended competition/hackathon activities on or before the payment date,
        # not only the T-7 window. Stop displaying thresholds once the count becomes zero.
        comp_before_payment = comp_att[comp_att.get("before_payment", False).fillna(False).astype(bool)].copy() if not comp_att.empty else pd.DataFrame()
        comp_thresholds = []
        if not comp_before_payment.empty:
            comp_per_student = (
                comp_before_payment.groupby(["student_id", "program"], as_index=False)["dedupe_key"]
                .nunique()
                .rename(columns={"dedupe_key": "attended_count"})
            )
            th = 1
            while True:
                sub = comp_per_student[comp_per_student["attended_count"] >= th]
                if sub.empty:
                    break
                comp_thresholds.append((f"Paid Students with ≥{th} Competition/Hackathon Before Payment", _group_sets_from_frame(sub), "paid"))
                th += 1
                if th > 100:  # safety guard for unexpected data issues
                    break
        if comp_thresholds:
            st.markdown("##### Before-payment attendance depth")
            st.dataframe(_success_metric_table(comp_thresholds, denoms), use_container_width=True, hide_index=True, key="success_comp_thresholds")

        if not comp_att.empty:
            chart_df = comp_att.groupby(["program", "event_type"], as_index=False)["student_id"].nunique().rename(columns={"student_id": "Unique Students"})
            fig = px.bar(chart_df, x="program", y="Unique Students", color="program", title="Competition & Hackathon Unique Participants")
            st.plotly_chart(nice_layout(fig, height=320), use_container_width=True, key="success_comp_participants_chart")

        with st.expander("Audit tables: deduped event occurrences and attendees"):
            a, b = st.tabs(["Event Occurrences", "Attended Student Events"])
            with a:
                disp = occurrences.copy()
                if not disp.empty:
                    disp["event_date"] = pd.to_datetime(disp["event_date"], errors="coerce").dt.strftime("%d-%b-%Y")
                st.dataframe(disp, use_container_width=True, height=320, key="success_occ_audit")
            with b:
                disp = attendees.copy()
                if not disp.empty:
                    disp["event_date"] = pd.to_datetime(disp["event_date"], errors="coerce").dt.strftime("%d-%b-%Y")
                st.dataframe(disp[[c for c in ["student_name", "program", "batch", "event_name", "event_type", "event_date", "is_paid", "before_payment", "before_7_payment", "is_winner"] if c in disp.columns]], use_container_width=True, height=360, key="success_att_audit")


# ---------------- Refund Analytics ----------------

def _student_id_from_row(row):
    email = clean_text(row.get("email_key", ""))
    name = clean_text(row.get("student_key", "")) or normalize_name(row.get("student_name", ""))
    return email or name


def build_tetrx_student_base_for_refunds(data):
    """One row per unique Tetr-X student for refund analytics."""
    rows = []
    activities = data.get("activities", {})
    for sheet in TX_SHEETS:
        df = activities.get(sheet, pd.DataFrame())
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            sid = _student_id_from_row(r)
            if not sid:
                continue
            status_raw = clean_text(r.get("sheet_status_raw", ""))
            pay_dt = pd.to_datetime(r.get("payment_date_parsed", pd.NaT), errors="coerce")
            rows.append({
                "student_id": sid,
                "Student Name": clean_text(r.get("student_name", "")),
                "Email": clean_text(r.get("email_key", "")),
                "UG/PG": clean_text(r.get("Program", infer_program_from_sheet(sheet))) or infer_program_from_sheet(sheet),
                "Batch": clean_text(r.get("Batch", "")),
                "Country": clean_text(r.get("Country", r.get("country", ""))),
                "Income": clean_text(r.get("Income", r.get("income", ""))),
                "Payment Date": pay_dt,
                "Status": status_raw,
                "is_refunded": "refund" in status_raw.lower(),
                "is_admitted": status_raw.strip().lower() == "admitted",
                "source_sheet": sheet,
                "email_key": clean_text(r.get("email_key", "")),
                "student_key": clean_text(r.get("student_key", "")),
            })
    base = pd.DataFrame(rows)
    if base.empty:
        return base

    # Collapse duplicate rows safely. Refund wins if any matching row is refunded; admitted wins if any is admitted.
    base = base.sort_values(["Payment Date", "Student Name"], ascending=[False, True], na_position="last")
    grouped = []
    for sid, g in base.groupby("student_id", dropna=False):
        g = g.copy()
        first = g.iloc[0].to_dict()
        pay_dates = pd.to_datetime(g["Payment Date"], errors="coerce").dropna()
        first["Payment Date"] = pay_dates.min() if not pay_dates.empty else pd.NaT
        first["is_refunded"] = bool(g["is_refunded"].fillna(False).any())
        first["is_admitted"] = bool(g["is_admitted"].fillna(False).any())
        statuses = [clean_text(x) for x in g["Status"].tolist() if clean_text(x)]
        first["Status"] = ", ".join(sorted(dict.fromkeys(statuses)))
        sheets = [clean_text(x) for x in g["source_sheet"].tolist() if clean_text(x)]
        first["source_sheet"] = ", ".join(sorted(dict.fromkeys(sheets)))
        # Prefer non-empty stable fields from any row.
        for col in ["Student Name", "Email", "UG/PG", "Batch", "Country", "Income", "email_key", "student_key"]:
            vals = [clean_text(x) for x in g[col].tolist() if clean_text(x)] if col in g.columns else []
            if vals:
                first[col] = vals[0]
        grouped.append(first)
    out = pd.DataFrame(grouped)
    if not out.empty:
        out["Payment Date"] = pd.to_datetime(out["Payment Date"], errors="coerce")
    return out


def collect_tetrx_events_for_refund_student(data, student_row, window_days=60):
    """Collect deduped refund-analytics activity for one Tetr-X student.

    Accuracy rule used for Refund Analytics:
    - Batch sheets: count attendance before/on payment date.
    - Tetr-X sheet: count attendance after/on payment date, up to window_days after payment.
    - Same student + event name + event type + event date is merged as 1 across sheets.
    """
    sid = clean_text(student_row.get("student_id", ""))
    email_key = clean_text(student_row.get("email_key", "")) or normalize_email(student_row.get("Email", ""))
    student_key = clean_text(student_row.get("student_key", "")) or normalize_name(student_row.get("Student Name", ""))
    program = clean_text(student_row.get("UG/PG", ""))
    pay_dt = pd.to_datetime(student_row.get("Payment Date", pd.NaT), errors="coerce")
    base_cols = ["student_id", "Student Name", "UG/PG", "Batch", "event_date", "event_type", "event_name", "source_sheet", "source_group", "dedupe_key"]
    rows = []
    if pd.isna(pay_dt):
        return pd.DataFrame(columns=base_cols)
    pay_dt = pay_dt.normalize()
    max_dt = pay_dt + pd.Timedelta(days=window_days) if window_days is not None else pd.NaT

    if program == "UG":
        candidate_sheets = [s for s in UG_BATCH_SHEETS if s in data.get("activities", {})] + ["Tetr-X-UG"]
    elif program == "PG":
        candidate_sheets = [s for s in PG_BATCH_SHEETS if s in data.get("activities", {})] + ["Tetr-X-PG"]
    else:
        candidate_sheets = [s for s in (UG_BATCH_SHEETS + PG_BATCH_SHEETS + TX_SHEETS) if s in data.get("activities", {})]

    for sheet in candidate_sheets:
        if sheet not in data.get("activities", {}) or sheet not in data.get("activity_ctx", {}):
            continue
        if program and infer_program_from_sheet(sheet) != program:
            continue
        sdf = data.get("activities", {}).get(sheet, pd.DataFrame())
        ctx = data.get("activity_ctx", {}).get(sheet, {})
        event_info = ctx.get("event_info", pd.DataFrame()) if isinstance(ctx, dict) else pd.DataFrame()
        if sdf is None or sdf.empty or event_info is None or event_info.empty:
            continue

        mask = pd.Series(False, index=sdf.index)
        if email_key and "email_key" in sdf.columns:
            mask = mask | sdf["email_key"].astype(str).eq(email_key)
        if student_key and "student_key" in sdf.columns:
            mask = mask | sdf["student_key"].astype(str).eq(student_key)
        part = sdf.loc[mask]
        if part.empty:
            continue

        is_tx = sheet in TX_SHEETS
        source_group = "Tetr-X post-payment" if is_tx else "Batch pre-payment"
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

                if is_tx:
                    if ev_date < pay_dt:
                        continue
                    if pd.notna(max_dt) and ev_date > max_dt:
                        continue
                else:
                    # Batch-sheet attendance is used for the pre-payment part of the refund journey.
                    if ev_date > pay_dt:
                        continue

                ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
                ev_type = clean_text(ev.get("event_type", "Other")) or "Other"
                date_key = ev_date.strftime("%Y-%m-%d")
                dedupe_key = "|".join([sid or email_key or student_key, normalize_name(ev_name), normalize_name(ev_type), date_key])
                rows.append({
                    "student_id": sid,
                    "Student Name": clean_text(student_row.get("Student Name", "")),
                    "UG/PG": program,
                    "Batch": clean_text(student_row.get("Batch", "")),
                    "event_date": ev_date,
                    "event_type": ev_type,
                    "event_name": ev_name,
                    "source_sheet": sheet,
                    "source_group": source_group,
                    "dedupe_key": dedupe_key,
                })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=base_cols)
    out = out.sort_values(["event_date", "event_name", "source_sheet"]).drop_duplicates(subset=["dedupe_key"]).reset_index(drop=True)
    return out

def build_refund_activity_summary(data, tx_students, window_days=60):
    summary_rows = []
    event_rows = []
    for _, stu in tx_students.iterrows():
        ev = collect_tetrx_events_for_refund_student(data, stu, window_days=window_days)
        count = int(ev["dedupe_key"].nunique()) if not ev.empty else 0
        event_types = ", ".join(ev.groupby("event_type")["dedupe_key"].nunique().sort_values(ascending=False).index.tolist()) if not ev.empty else ""
        event_names = ", ".join(ev.sort_values("event_date")["event_name"].dropna().astype(str).drop_duplicates().head(30).tolist()) if not ev.empty else ""
        summary_rows.append({
            "Student Name": clean_text(stu.get("Student Name", "")),
            "Email": clean_text(stu.get("Email", "")),
            "UG/PG": clean_text(stu.get("UG/PG", "")),
            "Batch": clean_text(stu.get("Batch", "")),
            "Payment Date": pd.to_datetime(stu.get("Payment Date", pd.NaT), errors="coerce"),
            "Status": clean_text(stu.get("Status", "")),
            "Activities in first 60 days after payment": count,
            "Event Types": event_types,
            "Event Names": event_names,
        })
        if not ev.empty:
            event_rows.append(ev.assign(Status=clean_text(stu.get("Status", ""))))
    summary = pd.DataFrame(summary_rows)
    events = pd.concat(event_rows, ignore_index=True) if event_rows else pd.DataFrame(columns=["student_id", "Student Name", "UG/PG", "Batch", "event_date", "event_type", "event_name", "source_sheet", "dedupe_key", "Status"])
    return summary, events


def _activity_distribution_table(summary_df, label):
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(columns=[label, "Students"])
    counts = pd.to_numeric(summary_df["Activities in first 60 days after payment"], errors="coerce").fillna(0).astype(int)
    dist = counts.value_counts().sort_index().reset_index()
    dist.columns = ["Activities", "Students"]
    dist[label] = dist["Activities"].apply(lambda n: f"did {int(n)} activity but didn't refund in 60 days" if "didn't refund" in label.lower() else f"did {int(n)} activity still refunded")
    return dist[[label, "Students"]]


def render_refund_analytics_page(data):
    st.subheader("Refund Analytics")
    tx_students = build_tetrx_student_base_for_refunds(data)
    if tx_students.empty:
        st.warning("No Tetr-X students found for refund analytics.")
        return

    refunded = tx_students[tx_students["is_refunded"]].copy()
    today = get_today_ist()
    tx_students["days_since_payment"] = (today - pd.to_datetime(tx_students["Payment Date"], errors="coerce").dt.normalize()).dt.days
    retained_pool = tx_students[(~tx_students["is_refunded"]) & (tx_students["is_admitted"]) & (tx_students["days_since_payment"].ge(60))].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Students Refunded", f"{refunded['student_id'].nunique():,}")
    c2.metric("UG Refunds", f"{refunded.loc[refunded['UG/PG'].eq('UG'), 'student_id'].nunique():,}")
    c3.metric("PG Refunds", f"{refunded.loc[refunded['UG/PG'].eq('PG'), 'student_id'].nunique():,}")

    st.markdown("---")
    st.markdown("### Refunded Student Activity")
    ug_tab, pg_tab = st.tabs(["UG Refunds", "PG Refunds"])
    for tab, prog in [(ug_tab, "UG"), (pg_tab, "PG")]:
        with tab:
            sub = refunded[refunded["UG/PG"].eq(prog)].copy()
            st.caption(f"Refunded students from Tetr-X-{prog}. Activity counts use deduped batch-sheet pre-payment events plus Tetr-X events in the first 60 days after payment.")
            if sub.empty:
                st.info(f"No {prog} refunded students found.")
                continue
            summary, events = build_refund_activity_summary(data, sub, window_days=60)
            show = summary.copy()
            if "Payment Date" in show.columns:
                show["Payment Date"] = pd.to_datetime(show["Payment Date"], errors="coerce").dt.strftime("%d-%b-%Y")
            st.dataframe(show.sort_values(["Payment Date", "Student Name"], ascending=[False, True], na_position="last"), use_container_width=True, height=360, key=f"refund_{prog}_summary")
            with st.expander(f"{prog} refunded students · detailed event list"):
                evshow = events.copy()
                if not evshow.empty:
                    evshow["event_date"] = pd.to_datetime(evshow["event_date"], errors="coerce").dt.strftime("%d-%b-%Y")
                cols = [c for c in ["Student Name", "UG/PG", "Batch", "event_date", "event_type", "event_name", "source_sheet"] if c in evshow.columns]
                st.dataframe(evshow[cols] if cols else evshow, use_container_width=True, height=420, key=f"refund_{prog}_events")

    st.markdown("---")
    st.markdown("### 60-Day Non-Refunded Activity Distribution")
    st.caption("Admitted Tetr-X students who completed 60 days after payment and did not refund. Counts use deduped batch-sheet pre-payment activities plus Tetr-X activities in the first 60 days after payment. Shown separately for UG and PG.")
    nr_ug_tab, nr_pg_tab = st.tabs(["UG Non-Refunded", "PG Non-Refunded"])
    for tab, prog in [(nr_ug_tab, "UG"), (nr_pg_tab, "PG")]:
        with tab:
            retained_sub = retained_pool[retained_pool["UG/PG"].eq(prog)].copy()
            retained_summary, _ = build_refund_activity_summary(data, retained_sub, window_days=60)
            dist_non_refund = pd.DataFrame()
            if not retained_summary.empty:
                counts = pd.to_numeric(retained_summary["Activities in first 60 days after payment"], errors="coerce").fillna(0).astype(int)
                max_count = int(counts.max()) if not counts.empty else 0
                dist_rows = []
                for n in range(0, max_count + 1):
                    cnt = int((counts == n).sum())
                    if cnt > 0:
                        dist_rows.append({"Activity Bucket": f"did {n} activity but didn't refund in 60 days", "Students": cnt})
                dist_non_refund = pd.DataFrame(dist_rows)
            if dist_non_refund.empty:
                st.info(f"No {prog} admitted students with completed 60-day non-refund windows found yet.")
            else:
                a, b = st.columns([1, 1])
                with a:
                    st.dataframe(dist_non_refund, use_container_width=True, hide_index=True, key=f"refund_non_refund_dist_{prog}")
                with b:
                    fig = px.bar(dist_non_refund, x="Activity Bucket", y="Students", title=f"{prog} Non-Refunded Students by Activity Count")
                    fig.update_traces(marker_color=GREEN_2)
                    st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-25), use_container_width=True, key=f"refund_non_refund_chart_{prog}")

    st.markdown("---")
    st.markdown("### Refunded Students by Activity Count")
    st.caption("Refunded Tetr-X students. Counts use deduped batch-sheet pre-payment activities plus Tetr-X activities in the first 60 days after payment. Shown separately for UG and PG.")
    r_ug_tab, r_pg_tab = st.tabs(["UG Refunded", "PG Refunded"])
    for tab, prog in [(r_ug_tab, "UG"), (r_pg_tab, "PG")]:
        with tab:
            refunded_sub = refunded[refunded["UG/PG"].eq(prog)].copy()
            refund_summary, _ = build_refund_activity_summary(data, refunded_sub, window_days=60)
            dist_refund = pd.DataFrame()
            if not refund_summary.empty:
                refund_summary = refund_summary.copy()
                refund_summary["_activity_count_bucket"] = pd.to_numeric(
                    refund_summary["Activities in first 60 days after payment"], errors="coerce"
                ).fillna(0).astype(int)
                counts = refund_summary["_activity_count_bucket"]
                max_count = int(counts.max()) if not counts.empty else 0
                dist_rows = []
                for n in range(max_count, -1, -1):
                    bucket_students = refund_summary.loc[counts == n, "Student Name"].dropna().astype(str).map(clean_text)
                    bucket_students = [x for x in bucket_students.tolist() if x]
                    cnt = len(bucket_students)
                    if cnt > 0:
                        dist_rows.append({
                            "Activity Bucket": f"did {n} activity still refunded",
                            "Students": cnt,
                            "Student Names": ", ".join(sorted(dict.fromkeys(bucket_students))),
                        })
                dist_refund = pd.DataFrame(dist_rows)
            if dist_refund.empty:
                st.info(f"No {prog} refunded students found for this distribution.")
            else:
                a, b = st.columns([1, 1])
                with a:
                    st.dataframe(dist_refund, use_container_width=True, hide_index=True, key=f"refund_refund_dist_{prog}")
                with b:
                    fig = px.bar(dist_refund, x="Activity Bucket", y="Students", title=f"{prog} Refunded Students by Activity Count")
                    fig.update_traces(marker_color=RED)
                    st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-25), use_container_width=True, key=f"refund_refund_chart_{prog}")


# ---------------- Targeted update overrides (fix52) ----------------

def is_online_masterclass_type(ev_type):
    b = map_retention_bucket_event_type(clean_text(ev_type))
    return b == "Online Events & Masterclasses" or any(k in clean_text(ev_type).lower() for k in ["online", "masterclass", "skill bootcamp", "ama"])


def _activity_count_for_row_from_event_info(row, event_info):
    if event_info is None or event_info.empty:
        return 0
    cnt = 0
    for _, ev in event_info.iterrows():
        col = ev.get("column_name")
        if col and col in row.index:
            cnt += int(pd.to_numeric(pd.Series([row.get(col, 0)]), errors="coerce").fillna(0).iloc[0] > 0)
    return cnt


def build_online_masterclass_attendees(df, ctx):
    event_info = ctx.get("event_info", pd.DataFrame()) if ctx else pd.DataFrame()
    cols = []
    if event_info is not None and not event_info.empty:
        for _, ev in event_info.iterrows():
            col = ev.get("column_name")
            if col in df.columns and is_online_masterclass_type(ev.get("event_type", "")):
                cols.append(col)
    if not cols or df is None or df.empty:
        return pd.DataFrame()
    mask = pd.Series(False, index=df.index)
    for c in cols:
        mask = mask | (pd.to_numeric(df[c], errors="coerce").fillna(0) > 0)
    out = df.loc[mask].copy()
    return out.drop_duplicates(subset=[c for c in ["email_key", "student_key"] if c in out.columns])


def render_om_attendance_section(df, ctx, prefix):
    if prefix.startswith("tx_") or df is None or df.empty:
        return
    attendees = build_online_masterclass_attendees(df, ctx)
    total_students = len(df)
    in_mask = is_community_in_series(df.get("community_status_value", pd.Series("", index=df.index)))
    in_total = int(in_mask.sum())
    not_in_total = int((~in_mask).sum())
    if attendees.empty:
        total_att = in_att = not_in_att = 0
    else:
        att_ids = set(attendees.apply(student_unique_id_from_row, axis=1).astype(str))
        df_ids = df.apply(student_unique_id_from_row, axis=1).astype(str)
        total_att = len(att_ids - {""})
        in_att = int(df.loc[in_mask & df_ids.isin(att_ids)].apply(student_unique_id_from_row, axis=1).astype(str).replace("", np.nan).dropna().nunique())
        not_in_att = int(df.loc[(~in_mask) & df_ids.isin(att_ids)].apply(student_unique_id_from_row, axis=1).astype(str).replace("", np.nan).dropna().nunique())

    st.markdown("#### Online Events + Masterclasses Unique Attendance")
    a,b,c = st.columns(3)
    a.metric("Unique Attendees", f"{total_att:,}", delta=f"{(total_att/total_students*100 if total_students else 0):.1f}% of students")
    b.metric("In Community Attendees", f"{in_att:,}", delta=f"{(in_att/in_total*100 if in_total else 0):.1f}% of In")
    c.metric("Not-In Community Attendees", f"{not_in_att:,}", delta=f"{(not_in_att/not_in_total*100 if not_in_total else 0):.1f}% of not In")
    if not attendees.empty:
        detail_cols = [col for col in ["student_name", "email_key", "Program", "Batch", "community_status_value", "sheet_status_raw"] if col in attendees.columns]
        st.dataframe(attendees[detail_cols].sort_values([c for c in ["Batch", "student_name"] if c in detail_cols]), use_container_width=True, height=250, key=f"{prefix}_om_attendees")


def render_sheet_detail(sheet_name, df, ctx, prefix, data=None):
    st.markdown(f"#### {sheet_name}")
    if df is None or df.empty:
        st.warning(f"No data available for {sheet_name}.")
        return

    total_students = int(len(df))
    active_students = int(df["is_active"].sum()) if "is_active" in df else int((pd.to_numeric(df.get("engagement_score", pd.Series(0, index=df.index)), errors="coerce").fillna(0) > 0).sum())
    paid_students = int(df["sheet_is_paid"].sum()) if "sheet_is_paid" in df else 0
    refunded_students = int(df["sheet_is_refunded"].sum()) if "sheet_is_refunded" in df else 0
    comm_series = df.get("community_status_value", pd.Series("", index=df.index)).astype(str)
    in_mask_metric = is_community_in_series(comm_series)
    in_community = int(in_mask_metric.sum())
    active_in_comm = int((df.get("is_active", pd.Series(False, index=df.index)).astype(bool) & in_mask_metric).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Students", f"{total_students:,}")
    k2.metric("Joined WA", f"{in_community:,}", delta=f"{(in_community/total_students*100 if total_students else 0):.1f}%")
    k3.metric("Active", f"{active_students:,}", delta=f"{(active_students/total_students*100 if total_students else 0):.1f}% overall, {(active_in_comm/in_community*100 if in_community else 0):.1f}% in Community")
    k4.metric("Admitted / Paid", f"{paid_students:,}", delta=f"{(paid_students/total_students*100 if total_students else 0):.1f}%")
    k5.metric("Refunded", f"{refunded_students:,}", delta=f"{(refunded_students/total_students*100 if total_students else 0):.1f}%")

    event_info = ctx.get("event_info", pd.DataFrame()) if ctx else pd.DataFrame()
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
        fig = px.pie(active_circle, names="Status", values="Students", hole=0.58, title="Active vs Non-Active", color="Status", color_discrete_map={"Active": GREEN, "Non-Active": GREEN_4})
        st.plotly_chart(nice_layout(fig, height=340), use_container_width=True, key=f"{prefix}_active_circle")

    comm = comm_series.replace("", np.nan).dropna()
    if not comm.empty:
        st.markdown("#### Community Status")
        comm_col, nonin_col = st.columns([1, 1.25])
        with comm_col:
            community_plot = comm.value_counts().reset_index()
            community_plot.columns = ["Community Status", "Students"]
            fig = px.pie(community_plot, names="Community Status", values="Students", hole=0.58, title="Community Status", color="Community Status", color_discrete_map={"Tetr X": GREEN, "In": GREEN_3, "Out": GREEN_4})
            st.plotly_chart(nice_layout(fig, height=340), use_container_width=True, key=f"{prefix}_community")
        with nonin_col:
            non_in = df[~is_community_in_series(comm_series)].copy()
            st.markdown("##### Students Not In Community")
            if non_in.empty:
                st.info("All listed students are marked In community.")
            else:
                cols = [c for c in ["student_name", "email_key", "community_status_value", "Batch", "Program"] if c in non_in.columns]
                st.dataframe(non_in[cols].sort_values([c for c in ["Batch", "student_name"] if c in cols]), use_container_width=True, height=320, key=f"{prefix}_non_in_students")

    d1, d2 = st.columns(2)
    with d1:
        if event_info is not None and not event_info.empty:
            participants = []
            for _, r in event_info.iterrows():
                col = r["column_name"]
                participants.append(int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum()) if col in df.columns else 0)
            event_counts = event_info.assign(Participants=participants).sort_values("Participants", ascending=False).head(12)
            fig = px.bar(event_counts, x="Participants", y="event_name", orientation="h", color="event_type", title="Top Events by Participation", hover_name="event_name", hover_data={"event_name": False, "event_type": True, "event_date": True, "Participants": True})
            st.plotly_chart(nice_layout(fig, height=460), use_container_width=True, key=f"{prefix}_events")
    with d2:
        country_col = ctx.get("country_col") if ctx else None
        if country_col and country_col in df.columns:
            top_country = df.groupby(country_col)["student_name"].count().reset_index(name="Students").sort_values("Students", ascending=False).head(10)
            fig = px.bar(top_country, x=country_col, y="Students", title="Country Split")
            fig.update_traces(marker_color=GREEN_3)
            st.plotly_chart(nice_layout(fig, height=430, x_tickangle=-30), use_container_width=True, key=f"{prefix}_country")

    t1, t2 = st.columns(2)
    with t1:
        top_col_order = ["student_name"]
        if prefix.startswith("course_") and "Batch" in df.columns:
            top_col_order.append("Batch")
        top_col_order += ["sheet_status_raw", "engagement_pct", "engagement_score", "community_status_value"]
        top_cols = [c for c in top_col_order if c in df.columns]
        students = df[top_cols].sort_values([c for c in ["engagement_pct", "engagement_score"] if c in top_cols], ascending=False).head(20) if top_cols else pd.DataFrame()
        if "sheet_status_raw" in students.columns:
            students = students.rename(columns={"sheet_status_raw": "Status"})
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
            target = df[(~df.get("sheet_is_paid", pd.Series(False, index=df.index))) & (~df.get("sheet_is_refunded", pd.Series(False, index=df.index))) & (df.get("is_active", pd.Series(False, index=df.index)))]
            target_col_order = ["student_name"]
            if prefix.startswith("course_") and "Batch" in target.columns:
                target_col_order.append("Batch")
            target_col_order += ["sheet_status_raw", "engagement_pct", "engagement_score", "community_status_value"]
            cols = [c for c in target_col_order if c in target.columns]
            target = target[cols].sort_values([c for c in ["engagement_pct", "engagement_score"] if c in cols], ascending=False).head(20) if cols else pd.DataFrame()
            if "sheet_status_raw" in target.columns:
                target = target.rename(columns={"sheet_status_raw": "Status"})
            st.markdown("#### Best Upgrade Targets")
            st.dataframe(target, use_container_width=True, height=390, key=f"{prefix}_upgrade_df")

    if not prefix.startswith("tx_"):
        render_paid_students_section(df, ctx, prefix, data=data)
        render_om_attendance_section(df, ctx, prefix)

    if event_info is not None and not event_info.empty and event_info["event_date"].notna().any():
        timeline = build_timeline_from_event_info(df, event_info)
        if not timeline.empty:
            fig = px.line(timeline, x="event_date", y="Participants", markers=True, title="Participation Timeline", hover_name="Event Names", hover_data={"Event Names": False, "Event Types": True, "event_date": True, "Participants": True})
            fig.update_traces(line_color=GREEN, marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{prefix}_timeline")
            event_table = build_event_attendance_table(df, event_info)
            if not event_table.empty:
                event_table_display = event_table.copy()
                event_table_display["Event / Activity Date"] = pd.to_datetime(event_table_display["Event / Activity Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("")
                st.markdown("##### Event / Activity Attendance Table")
                st.dataframe(event_table_display, use_container_width=True, hide_index=True, height=320, key=f"{prefix}_event_attendance_table")


def build_tx_risky_students(data, scope_label="Total"):
    top = _tx_top_engaged_students(data, scope_label)
    if top is None or top.empty:
        return pd.DataFrame()
    out = top.copy()
    attended = pd.to_numeric(out.get("Attended Events", 0), errors="coerce").fillna(0).astype(int)
    out["Risk Level"] = np.select([attended <= 2, attended.between(3, 5), attended >= 6], ["High Risk", "Moderate Risk", "Low Risk"], default="High Risk")
    return out.sort_values(["Risk Level", "Attended Events", "Name"], ascending=[True, True, True])


def render_tx_risky_students_section(data, scope_label, key_prefix):
    st.markdown("#### Risky Students")
    st.caption("Risk is based on Tetr-X post-payment activity attendance: High Risk = 0–2, Moderate Risk = 3–5, Low Risk = 6+ attended activities.")
    risky = build_tx_risky_students(data, scope_label)
    if risky.empty:
        st.info("No risk data available for this scope.")
        return
    cols = [c for c in ["Risk Level", "Name", "Email", "UG/PG", "Payment Date", "Attended Events", "Eligible Events", "Engagement %", "Top Event Types"] if c in risky.columns]
    if "Payment Date" in risky.columns:
        risky = risky.copy(); risky["Payment Date"] = pd.to_datetime(risky["Payment Date"], errors="coerce").dt.strftime("%d-%b-%Y")
    st.dataframe(risky[cols], use_container_width=True, hide_index=True, height=360, key=f"{key_prefix}_risky_students")
    chart = risky.groupby("Risk Level")["Name"].count().reset_index(name="Students")
    fig = px.bar(chart, x="Risk Level", y="Students", title="Risk Buckets")
    fig.update_traces(marker_color=AMBER)
    st.plotly_chart(nice_layout(fig, height=300), use_container_width=True, key=f"{key_prefix}_risky_chart")


def render_tetrx_analytics_scope(data, scope_label: str, key_prefix: str):
    students, metrics = build_tetrx_analytics_students(data, scope_label)
    st.markdown(f"### {scope_label}")
    if students.empty:
        st.info(f"No Tetr-X students found for {scope_label}.")
        return
    c = st.columns(7)
    c[0].metric("Total paid students", f"{metrics.get('Total paid students', 0):,}")
    c[1].metric("Current Paid Students", f"{metrics.get('Current Paid Students', 0):,}")
    c[2].metric("Refunded", f"{metrics.get('Refunded', 0):,}")
    c[3].metric("Deferral", f"{metrics.get('Deferral', 0):,}")
    c[4].metric("Paid & in group", f"{metrics.get('Paid & in group', 0):,}")
    c[5].metric("Refunded & in group", f"{metrics.get('Refunded & in group', 0):,}")
    c[6].metric("Paid & not in group", f"{metrics.get('Paid & not in group', 0):,}")

    st.markdown("#### Names of students in Tetr X")
    display = students[[col for col in ["Name", "Email", "Country", "Income", "Payment Date", "Tetr X/Term 0 Status", "Refund Status", "Days left to ask refund"] if col in students.columns]].copy()
    if "Payment Date" in display.columns:
        display["Payment Date"] = pd.to_datetime(display["Payment Date"], errors="coerce").dt.strftime("%d-%b-%Y")
    st.dataframe(display, use_container_width=True, height=360, hide_index=True, key=f"{key_prefix}_students")

    st.markdown("#### Post-Payment Engagement (Online Events + Masterclasses)")
    eng = _tx_online_masterclass_rows(data, scope_label)
    if eng.empty:
        st.info("No dated post-payment Online Event / Masterclass attendance found.")
    else:
        chart_df = eng.copy()
        chart_df["Date Label"] = pd.to_datetime(chart_df["Event Date"], errors="coerce").dt.strftime("%d %b")
        fig = px.bar(chart_df, x="Date Label", y="Attendance %", color="Month", hover_name="Event Name", hover_data={"Total Paid Students at the Event Date": True, "Attendance": True, "Attendance %": ':.1f', "Program": True, "Date Label": False}, title="Post-Payment Engagement (Online Events + Masterclasses)")
        fig.update_traces(texttemplate="%{y:.1f}%", textposition="outside")
        st.plotly_chart(nice_layout(fig, height=390, x_tickangle=-25), use_container_width=True, key=f"{key_prefix}_postpay_online_chart")
        table = eng.copy(); table["Event Date"] = pd.to_datetime(table["Event Date"], errors="coerce").dt.strftime("%d-%b-%Y")
        st.dataframe(table[["Program", "Event Date", "Event Name", "Total Paid Students at the Event Date", "Attendance", "Attendance %"]], use_container_width=True, hide_index=True, height=260, key=f"{key_prefix}_postpay_online_table")

    st.markdown("#### Top engaged students in Tetr X")
    top = _tx_top_engaged_students(data, scope_label)
    if top.empty:
        st.info("No eligible post-payment event data found for top engaged students.")
    else:
        chart_top = top.head(15).copy()
        fig = px.bar(chart_top, x="Engagement %", y="Name", orientation="h", hover_data=["Attended Events", "Eligible Events", "Top Event Types"], title="Top Engaged Students by Eligible Post-Payment Attendance %")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(nice_layout(fig, height=460), use_container_width=True, key=f"{key_prefix}_top_engaged_chart")
        disp = top.copy(); disp["Payment Date"] = pd.to_datetime(disp["Payment Date"], errors="coerce").dt.strftime("%d-%b-%Y")
        st.dataframe(disp, use_container_width=True, hide_index=True, height=360, key=f"{key_prefix}_top_engaged_table")

    render_tx_risky_students_section(data, scope_label, key_prefix)


def _build_conversion_extra_metrics(data):
    summary_df, events_df = build_retention_data(data)
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
    rows=[]
    for prog_label, sub in [("Total", summary_df), ("UG", summary_df[summary_df.get("program", "").eq("UG")]), ("PG", summary_df[summary_df.get("program", "").eq("PG")])]:
        paid_n=len(sub)
        if paid_n==0:
            avg_total=avg_before=avg_first30=0
        else:
            avg_total=float(pd.to_numeric(sub.get("total_count", sub.get("post_payment_count", 0)), errors="coerce").fillna(0).mean()) if "total_count" in sub or "post_payment_count" in sub else 0
            avg_before=float(pd.to_numeric(sub.get("before_payment_count", 0), errors="coerce").fillna(0).mean()) if "before_payment_count" in sub else 0
            avg_first30=float(pd.to_numeric(sub.get("first30_count", 0), errors="coerce").fillna(0).mean()) if "first30_count" in sub else 0
        rows += [
            {"Program": prog_label, "Metric": "Average session attended per admitted student", "Value": round(avg_total,2)},
            {"Program": prog_label, "Metric": "Average events attended before payment per admitted student", "Value": round(avg_before,2)},
            {"Program": prog_label, "Metric": "Average sessions attended per admitted student in 30 days", "Value": round(avg_first30,2)},
        ]
    return pd.DataFrame(rows)


_base_render_retention_page = render_retention_page

def render_retention_page(data):
    _base_render_retention_page(data)
    st.markdown("---")
    st.markdown("### Additional Conversion Averages")
    st.caption("These metrics use admitted / paid students and the same deduped activity logic as the Conversion page.")
    extra = _build_conversion_extra_metrics(data)
    if extra.empty:
        st.info("No admitted / paid conversion averages available.")
    else:
        fig = px.bar(extra, x="Metric", y="Value", color="Program", barmode="group", title="Average Sessions / Events per Admitted Student")
        st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-20), use_container_width=True, key="conversion_extra_avg_chart")
        st.dataframe(extra, use_container_width=True, hide_index=True, key="conversion_extra_avg_table")



def render_tetrx_page(data):
    st.subheader("Tetr-X")
    available = [s for s in TX_SHEETS if s in data.get("activities", {})]
    if not available:
        st.warning("Tetr-X sheets not available.")
        return
    tx_all = pd.concat([data["activities"][s] for s in available], ignore_index=True)
    tx_students = int(len(tx_all))
    tx_active = int(tx_all["is_active"].sum()) if "is_active" in tx_all else 0
    tx_paid = int(tx_all["sheet_is_paid"].sum()) if "sheet_is_paid" in tx_all else 0
    tx_refunded = int(tx_all["sheet_is_refunded"].sum()) if "sheet_is_refunded" in tx_all else 0
    tx_deferral = int(tx_all.get("sheet_status_raw", pd.Series("", index=tx_all.index)).astype(str).str.lower().str.contains("deferral", na=False).sum())
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tetr-X Students", f"{tx_students:,}")
    k2.metric("Active Students", f"{tx_active:,}", delta=f"{(tx_active/tx_students*100 if tx_students else 0):.1f}%")
    k3.metric("Admitted / Paid", f"{tx_paid:,}", delta=f"{(tx_paid/tx_students*100 if tx_students else 0):.1f}%")
    k4.metric("Refunded", f"{tx_refunded:,}", delta=f"{(tx_refunded/tx_students*100 if tx_students else 0):.1f}%")
    k5.metric("Deferral", f"{tx_deferral:,}", delta=f"{(tx_deferral/tx_students*100 if tx_students else 0):.1f}%")

    tab_labels = available + ["Tetr X Analytics"]
    tabs = st.tabs(tab_labels)
    for tab, label in zip(tabs, tab_labels):
        with tab:
            if label == "Tetr X Analytics":
                render_tetrx_analytics(data)
            else:
                render_sheet_detail(label, data["activities"][label], data["activity_ctx"][label], f"tx_{label}", data=data)



# ---------------- Targeted fixes v4.7 fix54 ----------------
# These overrides keep existing features intact and only change:
# 1) Conversion pre-payment averages, 2) Tetr-X risky-window filtering,
# 3) batch community/counsellor layout, and 4) PG B7 support via constant above.

def _student_match_mask(frame: pd.DataFrame, email_key: str, student_key: str) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=bool)
    mask = pd.Series(False, index=frame.index)
    if email_key and "email_key" in frame.columns:
        mask = mask | frame["email_key"].astype(str).eq(email_key)
    if student_key and "student_key" in frame.columns:
        mask = mask | frame["student_key"].astype(str).eq(student_key)
    return mask


def _paid_tetrx_students_for_conversion(data: dict) -> pd.DataFrame:
    rows = []
    for tx_sheet in TX_SHEETS:
        tx = data.get("activities", {}).get(tx_sheet, pd.DataFrame())
        if tx is None or tx.empty:
            continue
        program = "UG" if tx_sheet.endswith("UG") else "PG"
        paid = tx[tx.get("sheet_is_paid", pd.Series(False, index=tx.index)).fillna(False).astype(bool)].copy()
        for _, r in paid.iterrows():
            sid = clean_text(r.get("email_key", "")) or clean_text(r.get("student_key", "")) or normalize_name(r.get("student_name", ""))
            pay_dt = pd.to_datetime(r.get("payment_date_parsed", pd.NaT), errors="coerce")
            if not sid or pd.isna(pay_dt):
                continue
            rows.append({
                "student_id": sid,
                "student_name": clean_text(r.get("student_name", "")),
                "email_key": clean_text(r.get("email_key", "")),
                "student_key": clean_text(r.get("student_key", "")),
                "program": program,
                "batch": clean_text(r.get("Batch", "")),
                "payment_date": pay_dt.normalize(),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values("payment_date").drop_duplicates("student_id", keep="first").reset_index(drop=True)


def _conversion_prepayment_counts_for_student(data: dict, stu: pd.Series) -> tuple[int, int, int]:
    """Return (overall_before_payment, before_payment, first30) using batch sheets only."""
    program = clean_text(stu.get("program", ""))
    pay_dt = pd.to_datetime(stu.get("payment_date", pd.NaT), errors="coerce")
    if pd.isna(pay_dt):
        return 0, 0, 0
    pay_dt = pay_dt.normalize()
    email_key = clean_text(stu.get("email_key", ""))
    student_key = clean_text(stu.get("student_key", ""))
    student_name = clean_text(stu.get("student_name", ""))
    batch = clean_text(stu.get("batch", ""))

    dates_row = find_student_dates_row(data.get("dates_df", pd.DataFrame()), student_name, email_key, student_key, program, batch)
    offered_dt = pd.to_datetime(dates_row.get("offered_date_parsed", pd.NaT), errors="coerce") if dates_row is not None else pd.NaT
    deadline_dt = pd.to_datetime(dates_row.get("deadline_parsed", pd.NaT), errors="coerce") if dates_row is not None else pd.NaT
    if pd.notna(offered_dt):
        offered_dt = offered_dt.normalize()
    if pd.notna(deadline_dt):
        deadline_dt = deadline_dt.normalize()

    sheets = UG_BATCH_SHEETS if program == "UG" else PG_BATCH_SHEETS
    before_keys, first30_keys = set(), set()
    for sheet in sheets:
        df = data.get("activities", {}).get(sheet, pd.DataFrame())
        ctx = data.get("activity_ctx", {}).get(sheet, {})
        ev_info = ctx.get("event_info", pd.DataFrame()) if ctx else pd.DataFrame()
        if df is None or df.empty or ev_info is None or ev_info.empty:
            continue
        mask = _student_match_mask(df, email_key, student_key)
        if not mask.any():
            continue
        part = df.loc[mask]
        for _, prow in part.iterrows():
            for _, ev in ev_info.iterrows():
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
                ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
                ev_type = clean_text(ev.get("event_type", "Other")) or "Other"
                key = "|".join([student_key or email_key or normalize_name(student_name), normalize_name(ev_name), normalize_name(ev_type), ev_date.strftime("%Y-%m-%d")])
                if ev_date <= pay_dt:
                    before_keys.add(key)
                if pd.notna(offered_dt) and pd.notna(deadline_dt) and offered_dt <= ev_date <= deadline_dt:
                    # Dates sheet first-30 velocity must also use batch-sheet pre-payment attendance.
                    if ev_date <= pay_dt:
                        first30_keys.add(key)
    return len(before_keys), len(before_keys), len(first30_keys)


def _build_conversion_extra_metrics(data):
    paid = _paid_tetrx_students_for_conversion(data)
    if paid.empty:
        return pd.DataFrame(columns=["Program", "Metric", "Value"])
    rows = []
    count_rows = []
    for _, stu in paid.iterrows():
        overall, before, first30 = _conversion_prepayment_counts_for_student(data, stu)
        count_rows.append({"program": stu.get("program", ""), "overall": overall, "before": before, "first30": first30})
    counts = pd.DataFrame(count_rows)
    for prog_label, sub in [("Total", counts), ("UG", counts[counts["program"].eq("UG")]), ("PG", counts[counts["program"].eq("PG")])]:
        if sub.empty:
            avg_total = avg_before = avg_first30 = 0.0
        else:
            avg_total = float(pd.to_numeric(sub["overall"], errors="coerce").fillna(0).mean())
            avg_before = float(pd.to_numeric(sub["before"], errors="coerce").fillna(0).mean())
            avg_first30 = float(pd.to_numeric(sub["first30"], errors="coerce").fillna(0).mean())
        rows += [
            {"Program": prog_label, "Metric": "Average session attended per admitted student", "Value": round(avg_total, 2)},
            {"Program": prog_label, "Metric": "Average events attended before payment per admitted student", "Value": round(avg_before, 2)},
            {"Program": prog_label, "Metric": "Average sessions attended per admitted student in 30 days", "Value": round(avg_first30, 2)},
        ]
    return pd.DataFrame(rows)



def _conversion_scoped_data(data: dict, scope_label: str) -> dict:
    """Return a shallow copy of dashboard data filtered for Conversion scope only.

    This is intentionally used only by the Conversion page tabs so no other
    dashboard sections are affected. Total keeps the full data. UG keeps UG
    batch sheets + Tetr-X-UG. PG keeps PG batch sheets + Tetr-X-PG.
    """
    if scope_label == "Total":
        return data
    keep_sheets = set(UG_BATCH_SHEETS if scope_label == "UG" else PG_BATCH_SHEETS)
    keep_sheets.add("Tetr-X-UG" if scope_label == "UG" else "Tetr-X-PG")
    scoped = dict(data)
    scoped["activities"] = {k: v for k, v in data.get("activities", {}).items() if k in keep_sheets}
    scoped["activity_ctx"] = {k: v for k, v in data.get("activity_ctx", {}).items() if k in keep_sheets}
    return scoped


def _render_conversion_core_for_scope(data: dict, scope_label: str, key_prefix: str):
    scoped_data = _conversion_scoped_data(data, scope_label)
    summary_df, events_df = build_retention_data(scoped_data)

    if summary_df is None or summary_df.empty:
        st.warning(f"No admitted/paid students with usable payment dates were found for {scope_label}.")
        return

    # Safety filter in case a future helper starts retaining broader data.
    if scope_label in {"UG", "PG"} and "program" in summary_df.columns:
        summary_df = summary_df[summary_df["program"].astype(str).eq(scope_label)].copy()
    if scope_label in {"UG", "PG"} and events_df is not None and not events_df.empty and "program" in events_df.columns:
        events_df = events_df[events_df["program"].astype(str).eq(scope_label)].copy()

    total = int(len(summary_df))
    ug_total = int((summary_df.get("program", pd.Series("", index=summary_df.index)).astype(str) == "UG").sum())
    pg_total = int((summary_df.get("program", pd.Series("", index=summary_df.index)).astype(str) == "PG").sum())

    k1, k2, k3 = st.columns(3)
    k1.metric("Admitted / Paid / Tetr X Students", f"{total:,}")
    k2.metric("UG", f"{ug_total:,}")
    k3.metric("PG", f"{pg_total:,}")

    render_distribution_block("First 30 Days", summary_df["first30_count"], f"{key_prefix}_first30")
    render_event_type_block("First 30 Days Event Type Performance", events_df, "in_first30", f"{key_prefix}_first30", eligible_students=total)

    render_distribution_block("T-7", summary_df["tminus7_count"], f"{key_prefix}_tminus7")
    render_event_type_block("T-7 Event Type Performance", events_df, "in_tminus7", f"{key_prefix}_tminus7", eligible_students=total)

    render_distribution_block("T+7", summary_df["tplus7_count"], f"{key_prefix}_tplus7")
    render_event_type_block("T+7 Event Type Performance", events_df, "in_tplus7", f"{key_prefix}_tplus7", eligible_students=total)

    render_distribution_block("Post Payment Journey", summary_df["post_payment_count"], f"{key_prefix}_post")
    c1, c2 = st.columns(2)
    c1.metric("Students Active Post Payment", f"{int((summary_df['post_payment_count'] >= 1).sum()):,}")
    c2.metric("Students with 0 Post Payment Activities", f"{int((summary_df['post_payment_count'] == 0).sum()):,}")
    render_event_type_block("Post Payment Journey Event Type Performance", events_df, "post_payment", f"{key_prefix}_post", eligible_students=total)


def _render_conversion_averages_for_scope(data: dict, scope_label: str, key_prefix: str):
    st.markdown("---")
    st.markdown("### Additional Conversion Averages")
    st.caption("Pre-payment behavioural metrics use admitted / paid students from Tetr-X and attendance from their respective batch sheets.")
    extra = _build_conversion_extra_metrics(data)
    if extra is None or extra.empty:
        st.info("No admitted / paid conversion averages available.")
        return
    if scope_label in {"UG", "PG"}:
        extra = extra[extra["Program"].astype(str).eq(scope_label)].copy()
    if extra.empty:
        st.info(f"No admitted / paid conversion averages available for {scope_label}.")
        return
    fig = px.bar(extra, x="Metric", y="Value", color="Program", barmode="group", title="Pre-Payment Behavioural Data")
    st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-20), use_container_width=True, key=f"{key_prefix}_extra_avg_chart")
    st.dataframe(extra, use_container_width=True, hide_index=True, key=f"{key_prefix}_extra_avg_table")


def render_retention_page(data):
    st.subheader("Conversion")
    st.caption("Conversion analytics split into Total, UG, and PG views. Total keeps the combined view; UG and PG use their respective Tetr-X and batch sheets only.")
    tab_total, tab_ug, tab_pg = st.tabs(["Total", "UG", "PG"])
    with tab_total:
        _render_conversion_core_for_scope(data, "Total", "conversion_total")
        _render_conversion_averages_for_scope(data, "Total", "conversion_total")
    with tab_ug:
        _render_conversion_core_for_scope(data, "UG", "conversion_ug")
        _render_conversion_averages_for_scope(data, "UG", "conversion_ug")
    with tab_pg:
        _render_conversion_core_for_scope(data, "PG", "conversion_pg")
        _render_conversion_averages_for_scope(data, "PG", "conversion_pg")

def build_tx_risky_students(data, scope_label="Total"):
    top = _tx_top_engaged_students(data, scope_label)
    if top is None or top.empty:
        return pd.DataFrame()
    out = top.copy()
    today = get_today_ist()
    out["Payment Date"] = pd.to_datetime(out.get("Payment Date", pd.NaT), errors="coerce")
    out["Days left to refund"] = out["Payment Date"].apply(lambda d: max(0, 60 - int((today - d.normalize()).days)) if pd.notna(d) else np.nan)
    # Risk list should only show students still inside the 60-day refund window.
    out = out[out["Days left to refund"].fillna(-1).between(0, 60)].copy()
    if out.empty:
        return out
    attended = pd.to_numeric(out.get("Attended Events", 0), errors="coerce").fillna(0).astype(int)
    out["Risk Level"] = np.select([attended <= 2, attended.between(3, 5), attended >= 6], ["High Risk", "Moderate Risk", "Low Risk"], default="High Risk")
    return out.sort_values(["Risk Level", "Attended Events", "Name"], ascending=[True, True, True])


def render_tx_risky_students_section(data, scope_label, key_prefix):
    st.markdown("#### Risky Students")
    st.caption("Only students still inside the 60-day refund window are shown. Risk is based on Tetr-X post-payment attendance: High Risk = 0–2, Moderate Risk = 3–5, Low Risk = 6+ attended activities.")
    risky = build_tx_risky_students(data, scope_label)
    if risky.empty:
        st.info("No students inside the 60-day refund window for this risk view.")
        return
    cols = [c for c in ["Risk Level", "Name", "Email", "UG/PG", "Payment Date", "Days left to refund", "Attended Events", "Eligible Events", "Engagement %", "Top Event Types"] if c in risky.columns]
    disp = risky.copy()
    if "Payment Date" in disp.columns:
        disp["Payment Date"] = pd.to_datetime(disp["Payment Date"], errors="coerce").dt.strftime("%d-%b-%Y")
    st.dataframe(disp[cols], use_container_width=True, hide_index=True, height=360, key=f"{key_prefix}_risky_students")
    chart = risky.groupby("Risk Level")["Name"].count().reset_index(name="Students")
    fig = px.bar(chart, x="Risk Level", y="Students", title="Risk Buckets")
    fig.update_traces(marker_color=AMBER)
    st.plotly_chart(nice_layout(fig, height=300), use_container_width=True, key=f"{key_prefix}_risky_chart")


def _resolve_counsellor_series_for_frame(df: pd.DataFrame, ctx: dict | None = None) -> pd.Series:
    if df is None or df.empty:
        return pd.Series(dtype=str)
    if "counsellor_name" in df.columns:
        return df["counsellor_name"].map(clean_text).replace("", "Unknown")
    ctx = ctx or {}
    c1 = ctx.get("counsellor1_col")
    c2 = ctx.get("counsellor2_col")
    c = ctx.get("counsellor_col")
    if c1 and c1 in df.columns:
        s1 = df[c1].map(clean_text)
        s2 = df[c2].map(clean_text) if c2 and c2 in df.columns else pd.Series("", index=df.index)
        return s1.where(~s1.str.lower().eq("not required"), s2).replace("", "Unknown")
    if c and c in df.columns:
        return df[c].map(clean_text).replace("", "Unknown")
    return pd.Series("Unknown", index=df.index)


def render_sheet_detail(sheet_name, df, ctx, prefix, data=None):
    st.markdown(f"#### {sheet_name}")
    if df is None or df.empty:
        st.warning(f"No data available for {sheet_name}.")
        return

    total_students = int(len(df))
    active_students = int(df["is_active"].sum()) if "is_active" in df else int((pd.to_numeric(df.get("engagement_score", pd.Series(0, index=df.index)), errors="coerce").fillna(0) > 0).sum())
    paid_students = int(df["sheet_is_paid"].sum()) if "sheet_is_paid" in df else 0
    refunded_students = int(df["sheet_is_refunded"].sum()) if "sheet_is_refunded" in df else 0
    comm_series = df.get("community_status_value", pd.Series("", index=df.index)).astype(str)
    in_mask_metric = is_community_in_series(comm_series)
    in_community = int(in_mask_metric.sum())
    active_in_comm = int((df.get("is_active", pd.Series(False, index=df.index)).astype(bool) & in_mask_metric).sum())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Students", f"{total_students:,}")
    k2.metric("Joined WA", f"{in_community:,}", delta=f"{(in_community/total_students*100 if total_students else 0):.1f}%")
    k3.metric("Active", f"{active_students:,}", delta=f"{(active_students/total_students*100 if total_students else 0):.1f}% overall, {(active_in_comm/in_community*100 if in_community else 0):.1f}% in Community")
    k4.metric("Admitted / Paid", f"{paid_students:,}", delta=f"{(paid_students/total_students*100 if total_students else 0):.1f}%")
    k5.metric("Refunded", f"{refunded_students:,}", delta=f"{(refunded_students/total_students*100 if total_students else 0):.1f}%")

    event_info = ctx.get("event_info", pd.DataFrame()) if ctx else pd.DataFrame()
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
        fig = px.pie(active_circle, names="Status", values="Students", hole=0.58, title="Active vs Non-Active", color="Status", color_discrete_map={"Active": GREEN, "Non-Active": GREEN_4})
        st.plotly_chart(nice_layout(fig, height=340), use_container_width=True, key=f"{prefix}_active_circle")

    comm = comm_series.replace("", np.nan).dropna()
    if not comm.empty:
        st.markdown("#### Community Status")
        comm_col, counsellor_col = st.columns([1, 1.25])
        non_in_mask = ~is_community_in_series(comm_series)
        with comm_col:
            community_plot = comm.value_counts().reset_index()
            community_plot.columns = ["Community Status", "Students"]
            fig = px.pie(community_plot, names="Community Status", values="Students", hole=0.58, title="Community Status", color="Community Status", color_discrete_map={"Tetr X": GREEN, "In": GREEN_3, "Out": GREEN_4})
            st.plotly_chart(nice_layout(fig, height=340), use_container_width=True, key=f"{prefix}_community")
        with counsellor_col:
            out_df = df[non_in_mask].copy()
            out_df["Counsellor"] = _resolve_counsellor_series_for_frame(out_df, ctx)
            if out_df.empty:
                st.info("No Out community students for counsellor breakdown.")
            else:
                counsellor_plot = out_df.groupby("Counsellor")["student_name"].count().reset_index(name="Out Community Students").sort_values("Out Community Students", ascending=False)
                fig = px.bar(counsellor_plot, x="Counsellor", y="Out Community Students", title="Out Community Students by Counsellor")
                fig.update_traces(marker_color=AMBER)
                st.plotly_chart(nice_layout(fig, height=340, x_tickangle=-25), use_container_width=True, key=f"{prefix}_out_counsellor")
        non_in = df[non_in_mask].copy()
        st.markdown("##### Students Not In Community")
        if non_in.empty:
            st.info("All listed students are marked In community.")
        else:
            non_in["Counsellor"] = _resolve_counsellor_series_for_frame(non_in, ctx)
            cols = [c for c in ["student_name", "email_key", "community_status_value", "Counsellor", "Batch", "Program"] if c in non_in.columns]
            st.dataframe(non_in[cols].sort_values([c for c in ["Batch", "Counsellor", "student_name"] if c in cols]), use_container_width=True, height=320, key=f"{prefix}_non_in_students")

    d1, d2 = st.columns(2)
    with d1:
        if event_info is not None and not event_info.empty:
            participants = []
            for _, r in event_info.iterrows():
                col = r["column_name"]
                participants.append(int(pd.to_numeric(df[col], errors="coerce").fillna(0).sum()) if col in df.columns else 0)
            event_counts = event_info.assign(Participants=participants).sort_values("Participants", ascending=False).head(12)
            fig = px.bar(event_counts, x="Participants", y="event_name", orientation="h", color="event_type", title="Top Events by Participation", hover_name="event_name", hover_data={"event_name": False, "event_type": True, "event_date": True, "Participants": True})
            st.plotly_chart(nice_layout(fig, height=460), use_container_width=True, key=f"{prefix}_events")
    with d2:
        country_col = ctx.get("country_col") if ctx else None
        if country_col and country_col in df.columns:
            top_country = df.groupby(country_col)["student_name"].count().reset_index(name="Students").sort_values("Students", ascending=False).head(10)
            fig = px.bar(top_country, x=country_col, y="Students", title="Country Split")
            fig.update_traces(marker_color=GREEN_3)
            st.plotly_chart(nice_layout(fig, height=430, x_tickangle=-30), use_container_width=True, key=f"{prefix}_country")

    t1, t2 = st.columns(2)
    with t1:
        top_col_order = ["student_name"]
        if prefix.startswith("course_") and "Batch" in df.columns:
            top_col_order.append("Batch")
        top_col_order += ["sheet_status_raw", "engagement_pct", "engagement_score", "community_status_value"]
        top_cols = [c for c in top_col_order if c in df.columns]
        students = df[top_cols].sort_values([c for c in ["engagement_pct", "engagement_score"] if c in top_cols], ascending=False).head(20) if top_cols else pd.DataFrame()
        if "sheet_status_raw" in students.columns:
            students = students.rename(columns={"sheet_status_raw": "Status"})
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
            target = df[(~df.get("sheet_is_paid", pd.Series(False, index=df.index))) & (~df.get("sheet_is_refunded", pd.Series(False, index=df.index))) & (df.get("is_active", pd.Series(False, index=df.index)))]
            target_col_order = ["student_name"]
            if prefix.startswith("course_") and "Batch" in target.columns:
                target_col_order.append("Batch")
            target_col_order += ["sheet_status_raw", "engagement_pct", "engagement_score", "community_status_value"]
            cols = [c for c in target_col_order if c in target.columns]
            target = target[cols].sort_values([c for c in ["engagement_pct", "engagement_score"] if c in cols], ascending=False).head(20) if cols else pd.DataFrame()
            if "sheet_status_raw" in target.columns:
                target = target.rename(columns={"sheet_status_raw": "Status"})
            st.markdown("#### Best Upgrade Targets")
            st.dataframe(target, use_container_width=True, height=390, key=f"{prefix}_upgrade_df")

    if not prefix.startswith("tx_"):
        render_paid_students_section(df, ctx, prefix, data=data)
        render_om_attendance_section(df, ctx, prefix)

    if event_info is not None and not event_info.empty and event_info["event_date"].notna().any():
        timeline = build_timeline_from_event_info(df, event_info)
        if not timeline.empty:
            fig = px.line(timeline, x="event_date", y="Participants", markers=True, title="Participation Timeline", hover_name="Event Names", hover_data={"Event Names": False, "Event Types": True, "event_date": True, "Participants": True})
            fig.update_traces(line_color=GREEN, marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{prefix}_timeline")
            event_table = build_event_attendance_table(df, event_info)
            if not event_table.empty:
                event_table_display = event_table.copy()
                event_table_display["Event / Activity Date"] = pd.to_datetime(event_table_display["Event / Activity Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("")
                st.markdown("##### Event / Activity Attendance Table")
                st.dataframe(event_table_display, use_container_width=True, hide_index=True, height=320, key=f"{prefix}_event_attendance_table")



# ---------------- Activities Page ----------------

def _activity_event_type_norm(x):
    s = clean_text(x).lower()
    if "online" in s or "ama" in s:
        return "Online Event"
    if "masterclass" in s or "skill bootcamp" in s or "bootcamp" in s:
        return "Masterclass"
    if "competition" in s:
        return "Competition"
    if "hackathon" in s or "hackerthon" in s:
        return "Hackathon"
    if "fun" in s:
        return "Fun"
    if "poll" in s:
        return "Poll"
    return clean_text(x) or "Other"


def _event_name_key(x):
    s = normalize_name(x)
    # Light near-match cleanup while staying deterministic.
    s = re.sub(r"\b(session|event|ama|online|masterclass|webinar)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or normalize_name(x)


def _event_program_from_sheet(sheet):
    if sheet in UG_BATCH_SHEETS:
        return "UG"
    if sheet in PG_BATCH_SHEETS:
        return "PG"
    if sheet == "Tetr-X-UG":
        return "TetrX-UG"
    if sheet == "Tetr-X-PG":
        return "TetrX-PG"
    return infer_program_from_sheet(sheet) or ""


def _event_audience_label(sheet):
    if sheet in UG_BATCH_SHEETS:
        return "UG"
    if sheet in PG_BATCH_SHEETS:
        return "PG"
    if sheet in TX_SHEETS:
        return "TetrX"
    return infer_program_from_sheet(sheet) or ""


def _student_id_from_row_basic(row):
    return _student_id_from_values(row.get("email_key", ""), row.get("student_key", ""), row.get("student_name", ""))


def _paid_at_event_mask(df, event_date):
    if df is None or df.empty:
        return pd.Series(False, index=df.index if df is not None else [])
    status = df.get("sheet_status_raw", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()
    paid = status.eq("admitted") | df.get("sheet_is_paid", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    if pd.isna(event_date):
        return paid
    pay = pd.to_datetime(df.get("payment_date_parsed", pd.NaT), errors="coerce").dt.normalize()
    return paid & pay.notna() & (pay <= pd.to_datetime(event_date).normalize())


def _batch_distribution_for_ids(df, attended_mask, sheet):
    """Return a dict like {'UG B7': 4} for attended students in a source sheet."""
    if df is None or df.empty or not attended_mask.any():
        return {}
    sub = df.loc[attended_mask].copy()
    if "Batch" in sub.columns:
        labels = sub["Batch"].astype(str).map(clean_text)
        fallback = infer_batch_group_from_sheet_name(sheet) if 'infer_batch_group_from_sheet_name' in globals() else sheet
        labels = labels.replace("", fallback)
    else:
        fallback = infer_batch_group_from_sheet_name(sheet) if 'infer_batch_group_from_sheet_name' in globals() else sheet
        labels = pd.Series(fallback, index=sub.index)
    counts = labels.value_counts().to_dict()
    return {clean_text(k): int(v) for k, v in counts.items() if clean_text(k)}


def _merge_dist_dicts(*dicts):
    out = {}
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for k, v in d.items():
            k = clean_text(k)
            if not k:
                continue
            out[k] = out.get(k, 0) + int(v or 0)
    return out


def _format_dist_dict(d):
    if not isinstance(d, dict) or not d:
        return ""
    def sort_key(x):
        m = re.search(r"(\d+)", str(x[0]))
        return (str(x[0])[:2], int(m.group(1)) if m else 999, str(x[0]))
    return ", ".join([f"{v} {k}" for k, v in sorted(d.items(), key=sort_key)])


def build_activities_all_events(data, speaker_filter=None):
    """Build all Online Event + Masterclass occurrence table with deduped event name/date.

    Attendance is counted separately for UG, PG, Tetr-X UG and Tetr-X PG. Eligibility is the count of students in the
    audience sheet(s); for Tetr-X it is admitted students whose payment date is on/before the event date.
    """
    rows = []
    activities = data.get("activities", {})
    activity_ctx = data.get("activity_ctx", {})
    for sheet, df in activities.items():
        if df is None or df.empty:
            continue
        ctx = activity_ctx.get(sheet, {})
        event_info = ctx.get("event_info", pd.DataFrame())
        if event_info is None or event_info.empty:
            continue
        audience = _event_audience_label(sheet)
        if sheet == "Tetr-X-UG":
            program_group = "TetrX-UG"
        elif sheet == "Tetr-X-PG":
            program_group = "TetrX-PG"
        else:
            program_group = infer_program_from_sheet(sheet)
        for _, ev in event_info.iterrows():
            col = ev.get("column_name")
            if col not in df.columns:
                continue
            event_name = clean_text(ev.get("event_name", ""))
            event_type = _activity_event_type_norm(ev.get("event_type", ""))
            event_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
            if event_type not in {"Online Event", "Masterclass"}:
                continue
            if speaker_filter and speaker_filter.lower() not in event_name.lower():
                continue
            date_key = event_date.normalize().strftime("%Y-%m-%d") if pd.notna(event_date) else ""
            event_key = f"{_event_name_key(event_name)}|{date_key}"
            attended_mask = pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0)
            if sheet in TX_SHEETS:
                eligible_mask = _paid_at_event_mask(df, event_date)
                attended_mask = attended_mask & eligible_mask
            else:
                eligible_mask = pd.Series(True, index=df.index)
            eligible_ids = set(df.loc[eligible_mask].apply(_student_id_from_row_basic, axis=1).astype(str))
            eligible_ids.discard("")
            attended_ids = set(df.loc[attended_mask].apply(_student_id_from_row_basic, axis=1).astype(str))
            attended_ids.discard("")
            ug_dist = _batch_distribution_for_ids(df, attended_mask, sheet) if program_group == "UG" else {}
            pg_dist = _batch_distribution_for_ids(df, attended_mask, sheet) if program_group == "PG" else {}
            rows.append({
                "event_key": event_key,
                "Event Name": event_name,
                "Date": event_date.normalize() if pd.notna(event_date) else pd.NaT,
                "Event Type": event_type,
                "Audience": audience,
                "source_sheet": sheet,
                "UG Attended IDs": attended_ids if program_group == "UG" else set(),
                "PG Attended IDs": attended_ids if program_group == "PG" else set(),
                "TetrX UG Attended IDs": attended_ids if program_group == "TetrX-UG" else set(),
                "TetrX PG Attended IDs": attended_ids if program_group == "TetrX-PG" else set(),
                "Eligible IDs": eligible_ids,
                "UG Distribution Dict": ug_dist,
                "PG Distribution Dict": pg_dist,
            })
    if not rows:
        return pd.DataFrame(columns=["Event Name", "Date", "Audience", "Audience Size", "UG Attended", "PG Attended", "Tetr X UG Attended", "Tetr X PG Attended", "Total Attendance", "Attend to Eligible Ratio", "UG Attendance Distribution", "PG Attendance Distribution"])
    raw = pd.DataFrame(rows)
    out_rows = []
    for key, g in raw.groupby("event_key", dropna=False):
        ug_ids, pg_ids, txug_ids, txpg_ids, eligible_ids = set(), set(), set(), set(), set()
        audiences, sheets = [], []
        ug_dist, pg_dist = {}, {}
        for _, r in g.iterrows():
            ug_ids |= set(r.get("UG Attended IDs", set()))
            pg_ids |= set(r.get("PG Attended IDs", set()))
            txug_ids |= set(r.get("TetrX UG Attended IDs", set()))
            txpg_ids |= set(r.get("TetrX PG Attended IDs", set()))
            eligible_ids |= set(r.get("Eligible IDs", set()))
            ug_dist = _merge_dist_dicts(ug_dist, r.get("UG Distribution Dict", {}))
            pg_dist = _merge_dist_dicts(pg_dist, r.get("PG Distribution Dict", {}))
            if clean_text(r.get("Audience", "")):
                audiences.append(clean_text(r.get("Audience", "")))
            if clean_text(r.get("source_sheet", "")):
                sheets.append(clean_text(r.get("source_sheet", "")))
        first = g.iloc[0]
        audience_size = len(eligible_ids)
        total_att = len(ug_ids | pg_ids | txug_ids | txpg_ids)
        out_rows.append({
            "Event Name": clean_text(first.get("Event Name", "")),
            "Date": first.get("Date", pd.NaT),
            "Event Type": clean_text(first.get("Event Type", "")),
            "Audience": ", ".join(sorted(set(audiences))),
            "Source Sheets": ", ".join(sorted(set(sheets))),
            "Audience Size": audience_size,
            "UG Attended": len(ug_ids),
            "PG Attended": len(pg_ids),
            "Tetr X UG Attended": len(txug_ids),
            "Tetr X PG Attended": len(txpg_ids),
            "TetrX Attended": len(txug_ids | txpg_ids),
            "Total Attendance": total_att,
            "Attend to Eligible Ratio": (total_att / audience_size * 100) if audience_size else 0.0,
            "UG Attendance Distribution": _format_dist_dict(ug_dist),
            "PG Attendance Distribution": _format_dist_dict(pg_dist),
        })
    out = pd.DataFrame(out_rows).sort_values(["Date", "Event Name"], ascending=[False, True], na_position="last")
    return out.reset_index(drop=True)


def _student_id_from_activity_row(row):
    return _student_id_from_values(row.get("email_key", ""), row.get("student_key", ""), row.get("student_name", ""))


def _tetrx_student_frame(data, program="UG"):
    tx_sheet = "Tetr-X-UG" if str(program).upper() == "UG" else "Tetr-X-PG"
    tx = data.get("activities", {}).get(tx_sheet, pd.DataFrame())
    if tx is None or tx.empty:
        return pd.DataFrame()
    status = tx.get("sheet_status_raw", pd.Series("", index=tx.index)).astype(str).map(clean_text)
    out = tx.copy()
    out["student_id"] = out.apply(_student_id_from_activity_row, axis=1)
    out["Status"] = status
    out = out[out["student_id"].astype(str).str.len().gt(0)].copy()
    out = out.drop_duplicates("student_id", keep="first").reset_index(drop=True)
    return out


def _tetrx_ug_student_frame(data):
    return _tetrx_student_frame(data, "UG")


def _categorize_activities_ama(event_name, event_type):
    name = clean_text(event_name).lower()
    typ = _activity_event_type_norm(event_type)
    if typ == "Online Event":
        if any(x in name for x in ["shahrose", "welcome webinar", "harshit"]):
            return "AMA Welcome Webinar"
        if "pratham" in name:
            return "AMA Pratham"
        if "tarun" in name:
            return "AMA Tarun"
        if "amitoj" in name:
            return "AMA Amitoj"
        if "garima" in name:
            return "AMA Garima"
        if any(x in name for x in ["kritee", "ayush", "saarthak"]):
            return "AMA Capstone"
        if any(x in name for x in ["jessica", "yuliia", "yulia"]):
            return "AMA Life at Tetr"
    if typ == "Masterclass":
        return "Masterclass"
    if typ == "Competition":
        return "Competition"
    if typ == "Hackathon":
        return "Hackathon"
    return None


def _event_is_eligible_for_student(mode, sheet, tx_sheet, ev_date, pay, offered, deadline):
    """Eligibility for a Tetr-X activity-matrix student/category cell.

    Pre-Payment: only batch-sheet events on/before payment date.
    First 30 Days: all students are eligible for the view; attendance is included inside Offered date -> Deadline.
    Post-Payment: events on/after payment date.
    All: pre-payment batch events plus post-payment events, with duplicate event rows merged later.
    """
    if pd.isna(ev_date):
        return False
    if mode == "Pre-Payment":
        return sheet != tx_sheet and pd.notna(pay) and ev_date <= pay
    if mode == "First 30 Days":
        return True
    if mode == "Post-Payment":
        return pd.notna(pay) and ev_date >= pay
    # All
    if pd.isna(pay):
        return False
    if sheet != tx_sheet and ev_date <= pay:
        return True
    if ev_date >= pay:
        return True
    return False


def _event_include_attendance_for_student(mode, sheet, tx_sheet, ev_date, pay, offered, deadline):
    if pd.isna(ev_date):
        return False
    if mode == "Pre-Payment":
        return sheet != tx_sheet and pd.notna(pay) and ev_date <= pay
    if mode == "First 30 Days":
        return pd.notna(offered) and pd.notna(deadline) and offered <= ev_date <= deadline
    if mode == "Post-Payment":
        return pd.notna(pay) and ev_date >= pay
    # All: count every attended event the student was eligible for before or after payment.
    return _event_is_eligible_for_student(mode, sheet, tx_sheet, ev_date, pay, offered, deadline)


AMA_ACTIVITY_COLUMNS = [
    "AMA Welcome Webinar", "AMA Pratham", "AMA Tarun", "AMA Amitoj",
    "AMA Garima", "AMA Capstone", "AMA Life at Tetr"
]
AMA_MASTERCLASS_COLUMN = "AMAs & Masterclasses"


def _combine_activity_columns_for_row(row, cols):
    """Sum activity columns while preserving N/A when the student was not eligible for any source column."""
    total = 0
    has_numeric = False
    for col in cols:
        val = row.get(col, "N/A") if hasattr(row, "get") else "N/A"
        parsed = pd.to_numeric(pd.Series([val]), errors="coerce").iloc[0]
        if pd.notna(parsed):
            total += parsed
            has_numeric = True
    return int(total) if has_numeric else "N/A"


def _combined_activity_summary_row(table, eligible_student_ids_by_category, source_cols, label):
    """Build a summary row for a combined activity bucket using unique eligible students."""
    if table is None or table.empty or label not in table.columns:
        return {
            "Activity Column": label,
            "Total Attendance": 0,
            "Unique Students Attended": 0,
            "Eligible Unique Students": 0,
            "% of Activity-Type Eligible Students": 0.0,
        }
    combined_counts = pd.to_numeric(table[label], errors="coerce")
    eligible_ids = set()
    for col in source_cols:
        eligible_ids |= set(eligible_student_ids_by_category.get(col, set()))
    unique_attended = int(combined_counts.fillna(0).gt(0).sum()) if len(combined_counts) else 0
    eligible_count = len(eligible_ids)
    return {
        "Activity Column": label,
        "Total Attendance": int(combined_counts.fillna(0).sum()) if len(combined_counts) else 0,
        "Unique Students Attended": unique_attended,
        "Eligible Unique Students": eligible_count,
        "% of Activity-Type Eligible Students": (unique_attended / eligible_count * 100) if eligible_count else 0.0,
    }


def _tetrx_event_matrix(data, mode, program="UG"):
    program = str(program).upper()
    tx_sheet = "Tetr-X-UG" if program == "UG" else "Tetr-X-PG"
    batch_sheets = UG_BATCH_SHEETS if program == "UG" else PG_BATCH_SHEETS
    students = _tetrx_student_frame(data, program)
    categories = ["AMA Welcome Webinar", "AMA Pratham", "AMA Tarun", "AMA Amitoj", "AMA Garima", "AMA Capstone", "AMA Life at Tetr", "Masterclass", "Competition", "Hackathon"]
    if students.empty:
        return pd.DataFrame(columns=["Student Name", "Email", "Status"] + categories), pd.DataFrame()

    student_info = {}
    dates_df = data.get("dates_df", pd.DataFrame())
    for _, r in students.iterrows():
        sid = r.get("student_id", "")
        pay_dt = pd.to_datetime(r.get("payment_date_parsed", pd.NaT), errors="coerce")
        pay_dt = pay_dt.normalize() if pd.notna(pay_dt) else pd.NaT
        drow = find_student_dates_row(dates_df, r.get("student_name", ""), r.get("email_key", ""), r.get("student_key", ""), program, r.get("Batch", "")) if dates_df is not None and not dates_df.empty else None
        offered = pd.to_datetime(drow.get("offered_date_parsed", pd.NaT), errors="coerce") if drow is not None else pd.NaT
        deadline = pd.to_datetime(drow.get("deadline_parsed", pd.NaT), errors="coerce") if drow is not None else pd.NaT
        student_info[sid] = {
            "name": clean_text(r.get("student_name", "")),
            "email": clean_text(r.get("email_key", "")),
            "status": clean_text(r.get("Status", r.get("sheet_status_raw", ""))),
            "payment_date": pay_dt,
            "offered": offered.normalize() if pd.notna(offered) else pd.NaT,
            "deadline": deadline.normalize() if pd.notna(deadline) else pd.NaT,
        }

    event_rows = []
    eligible_student_ids_by_category = {c: set() for c in categories}
    eligible_student_category = {sid: {c: False for c in categories} for sid in student_info.keys()}

    for sheet in list(batch_sheets) + [tx_sheet]:
        df = data.get("activities", {}).get(sheet, pd.DataFrame())
        ctx = data.get("activity_ctx", {}).get(sheet, {})
        ev_info = ctx.get("event_info", pd.DataFrame())
        if df is None or df.empty or ev_info is None or ev_info.empty:
            continue
        for _, ev in ev_info.iterrows():
            col = ev.get("column_name")
            if col not in df.columns:
                continue
            category = _categorize_activities_ama(ev.get("event_name", ""), ev.get("event_type", ""))
            if not category:
                continue
            ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
            ev_date = ev_date.normalize() if pd.notna(ev_date) else pd.NaT

            # Mark eligible students for this activity bucket. For First 30 Days, every
            # Tetr-X student is eligible for every bucket that exists; the date window only
            # controls counted attendance.
            if pd.notna(ev_date):
                for sid, info in student_info.items():
                    pay = info.get("payment_date", pd.NaT)
                    offered = info.get("offered", pd.NaT)
                    deadline = info.get("deadline", pd.NaT)
                    eligible_for_event = _event_is_eligible_for_student(mode, sheet, tx_sheet, ev_date, pay, offered, deadline)
                    if eligible_for_event:
                        eligible_student_ids_by_category[category].add(sid)
                        eligible_student_category[sid][category] = True

            attended = pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0)
            for _, sr in df.loc[attended].iterrows():
                sid = _student_id_from_values(sr.get("email_key", ""), sr.get("student_key", ""), sr.get("student_name", ""))
                if sid not in student_info:
                    continue
                info = student_info[sid]
                pay = info.get("payment_date", pd.NaT)
                offered = info.get("offered", pd.NaT)
                deadline = info.get("deadline", pd.NaT)
                include = _event_include_attendance_for_student(mode, sheet, tx_sheet, ev_date, pay, offered, deadline)
                if not include:
                    continue
                # If an attended event exists, keep that student/category eligible too, so
                # counts and denominators cannot become inconsistent because of missing dates.
                eligible_student_ids_by_category[category].add(sid)
                eligible_student_category[sid][category] = True
                dedupe = f"{sid}|{_event_name_key(ev.get('event_name',''))}|{category}|{ev_date.strftime('%Y-%m-%d') if pd.notna(ev_date) else ''}"
                event_rows.append({
                    "student_id": sid,
                    "category": category,
                    "event_date": ev_date,
                    "event_name": clean_text(ev.get("event_name", "")),
                    "source_sheet": sheet,
                    "dedupe_key": dedupe,
                })

    events = pd.DataFrame(event_rows)
    if not events.empty:
        events = events.drop_duplicates("dedupe_key")

    # Winner counts for each Tetr-X student, shown only in Activities matrices.
    winner_lookup = {}
    winner_df = data.get("winner_df", pd.DataFrame())
    if winner_df is not None and not winner_df.empty:
        w = winner_df.copy()
        if "is_winner" in w.columns:
            w = w[w["is_winner"].fillna(False).astype(bool)].copy()
        for _, wr in w.iterrows():
            wid = _student_id_from_values(wr.get("email_key", ""), wr.get("student_key", ""), wr.get("winner_name", ""))
            if wid:
                winner_lookup[wid] = winner_lookup.get(wid, 0) + 1

    table_rows = []
    for sid, info in student_info.items():
        row = {"Student Name": info["name"], "Email": info["email"], "Status": info["status"], "Winner": int(winner_lookup.get(sid, 0))}
        if events.empty:
            counts = pd.Series(dtype=int)
        else:
            counts = events[events["student_id"].eq(sid)]["category"].value_counts()
        for c in categories:
            if mode in {"Pre-Payment", "Post-Payment", "All"} and not eligible_student_category.get(sid, {}).get(c, False):
                row[c] = "N/A"
            else:
                row[c] = int(counts.get(c, 0))
        table_rows.append(row)
    table = pd.DataFrame(table_rows).sort_values("Student Name").reset_index(drop=True)

    # Combined column requested only for Activities matrices. It preserves N/A when
    # the student was not eligible for either Competition or Hackathon in the selected view.
    def _combine_activity_cells(a, b):
        av = pd.to_numeric(pd.Series([a]), errors="coerce").iloc[0]
        bv = pd.to_numeric(pd.Series([b]), errors="coerce").iloc[0]
        if pd.isna(av) and pd.isna(bv):
            return "N/A"
        return int((0 if pd.isna(av) else av) + (0 if pd.isna(bv) else bv))

    if "Competition" in table.columns and "Hackathon" in table.columns:
        table["Competition & Hackathon"] = [
            _combine_activity_cells(a, b) for a, b in zip(table["Competition"], table["Hackathon"])
        ]

    ama_masterclass_cols = [c for c in AMA_ACTIVITY_COLUMNS + ["Masterclass"] if c in table.columns]
    if ama_masterclass_cols:
        table[AMA_MASTERCLASS_COLUMN] = table.apply(
            lambda row: _combine_activity_columns_for_row(row, ama_masterclass_cols), axis=1
        )

    eligibility = {c: len(eligible_student_ids_by_category.get(c, set())) for c in categories}
    summary_rows = []
    for c in categories:
        numeric_counts = pd.to_numeric(table[c], errors="coerce") if c in table.columns else pd.Series(dtype=float)
        total_att = int(numeric_counts.fillna(0).sum()) if len(numeric_counts) else 0
        eligible = int(eligibility.get(c, 0))
        unique_attended = int(numeric_counts.fillna(0).gt(0).sum()) if len(numeric_counts) else 0
        summary_rows.append({
            "Activity Column": c,
            "Total Attendance": total_att,
            "Unique Students Attended": unique_attended,
            "Eligible Unique Students": eligible,
            "% of Activity-Type Eligible Students": (unique_attended / eligible * 100) if eligible else 0.0,
        })
    if "Competition & Hackathon" in table.columns:
        summary_rows.append(_combined_activity_summary_row(
            table, eligible_student_ids_by_category, ["Competition", "Hackathon"], "Competition & Hackathon"
        ))
    if AMA_MASTERCLASS_COLUMN in table.columns:
        summary_rows.append(_combined_activity_summary_row(
            table, eligible_student_ids_by_category, [c for c in AMA_ACTIVITY_COLUMNS + ["Masterclass"] if c in categories], AMA_MASTERCLASS_COLUMN
        ))
    summary = pd.DataFrame(summary_rows)
    return table, summary


def _tetrx_ug_event_matrix(data, mode):
    return _tetrx_event_matrix(data, mode, "UG")


def _render_tetrx_activity_matrix(data, program, title, key_prefix):
    st.markdown(f"### {title} Activity Matrix")
    mode = st.radio("Select View", ["Pre-Payment", "First 30 Days", "Post-Payment", "All"], horizontal=True, key=f"{key_prefix}_mode")
    matrix, summary = _tetrx_event_matrix(data, mode, program)
    if matrix.empty:
        st.info(f"No {title} student data found.")
        return
    total_students = len(matrix)
    paid_students = int(matrix["Status"].astype(str).str.lower().eq("admitted").sum())
    winner_students = int(pd.to_numeric(matrix.get("Winner", pd.Series(0, index=matrix.index)), errors="coerce").fillna(0).gt(0).sum()) if "Winner" in matrix.columns else 0
    winner_pct = (winner_students / total_students * 100) if total_students else 0.0
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Students", f"{total_students:,}")
    m2.metric("Total Paid / Admitted Students", f"{paid_students:,}")
    m3.metric("Winners", f"{winner_students:,}", delta=f"{winner_pct:.1f}% of students")
    attendance_cols = [c for c in ["AMA Welcome Webinar", "AMA Pratham", "AMA Tarun", "AMA Amitoj", "AMA Garima", "AMA Capstone", "AMA Life at Tetr", "Masterclass", "AMAs & Masterclasses", "Competition", "Hackathon", "Competition & Hackathon"] if c in matrix.columns]

    def _highlight_activity_nonzero(val):
        try:
            return "background-color: #dff3e7; color: #0b3d2e; font-weight: 700;" if float(val) > 0 else ""
        except Exception:
            return ""

    try:
        styled_matrix = matrix.style.applymap(_highlight_activity_nonzero, subset=attendance_cols)
        st.dataframe(styled_matrix, use_container_width=True, hide_index=True, height=420, key=f"{key_prefix}_matrix_{mode}")
    except Exception:
        st.dataframe(matrix, use_container_width=True, hide_index=True, height=420, key=f"{key_prefix}_matrix_{mode}")

    st.markdown("#### Attendance Summary")
    sdisp = summary.copy()
    if not sdisp.empty:
        if "% of Activity-Type Eligible Students" in sdisp.columns:
            sdisp["% of Activity-Type Eligible Students"] = sdisp["% of Activity-Type Eligible Students"].map(lambda x: f"{x:.1f}%")
        st.dataframe(sdisp, use_container_width=True, hide_index=True, key=f"{key_prefix}_summary_{mode}")
    ama_cols = ["AMA Welcome Webinar", "AMA Pratham", "AMA Tarun", "AMA Amitoj", "AMA Garima", "AMA Capstone", "AMA Life at Tetr"]
    numeric_matrix = matrix.copy()
    for col in attendance_cols:
        numeric_matrix[col] = pd.to_numeric(numeric_matrix[col], errors="coerce").fillna(0)
    avg_ama = numeric_matrix[ama_cols].sum(axis=1).mean() if all(c in numeric_matrix.columns for c in ama_cols) else 0
    avg_masterclass = numeric_matrix['Masterclass'].mean() if "Masterclass" in numeric_matrix else 0.0
    avg_ama_masterclass = numeric_matrix[AMA_MASTERCLASS_COLUMN].mean() if AMA_MASTERCLASS_COLUMN in numeric_matrix.columns else ((numeric_matrix[ama_cols].sum(axis=1) + (numeric_matrix['Masterclass'] if "Masterclass" in numeric_matrix else 0)).mean() if all(c in numeric_matrix.columns for c in ama_cols) else avg_masterclass)
    avg_comp_masterclass = numeric_matrix["Competition & Hackathon"].mean() if "Competition & Hackathon" in numeric_matrix else ((numeric_matrix["Competition"] if "Competition" in numeric_matrix else 0) + (numeric_matrix["Hackathon"] if "Hackathon" in numeric_matrix else 0)).mean()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Average AMA's Attended", f"{avg_ama:.2f}")
    c2.metric("Average AMAs & Masterclass Attended", f"{avg_ama_masterclass:.2f}")
    c3.metric("Average Competition & Hackathon", f"{avg_comp_masterclass:.2f}")
    c4.metric("Average Masterclass Attended", f"{avg_masterclass:.2f}")
    c5.metric("Average Competition Attended", f"{numeric_matrix['Competition'].mean():.2f}" if "Competition" in numeric_matrix else "0.00")
    c6.metric("Average Hackathon Attended", f"{numeric_matrix['Hackathon'].mean():.2f}" if "Hackathon" in numeric_matrix else "0.00")
    chart_df = summary.copy()
    if not chart_df.empty:
        pct_col = "% of Activity-Type Eligible Students"
        if pct_col in chart_df.columns:
            chart_df[pct_col] = pd.to_numeric(chart_df[pct_col], errors="coerce").fillna(0)
            chart_df["Percentage Label"] = chart_df[pct_col].map(lambda x: f"{x:.1f}%")
            fig = px.bar(
                chart_df,
                x="Activity Column",
                y=pct_col,
                text="Percentage Label",
                title=f"{title} Attendance % by Activity · {mode}",
                hover_data={
                    "Activity Column": False,
                    pct_col: ":.1f",
                    "Unique Students Attended": True,
                    "Eligible Unique Students": True,
                    "Total Attendance": True,
                    "Percentage Label": False,
                },
            )
            fig.update_yaxes(title="% of Activity-Type Eligible Students")
        else:
            fig = px.bar(chart_df, x="Activity Column", y="Total Attendance", text="Total Attendance", title=f"{title} Attendance by Activity · {mode}")
        fig.update_traces(marker_color=GREEN, textposition="outside")
        st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-25), use_container_width=True, key=f"{key_prefix}_chart_{mode}")




def _unpaid_student_frame(data, program="UG"):
    """Unique unpaid/non-admitted students from batch sheets only for Activities → Unpaid UG/PG."""
    program = str(program).upper()
    batch_sheets = UG_BATCH_SHEETS if program == "UG" else PG_BATCH_SHEETS
    # Exclude anybody who is already present in the relevant Tetr-X sheet as admitted/paid.
    tx_students = _tetrx_student_frame(data, program)
    tx_paid_ids = set()
    if tx_students is not None and not tx_students.empty:
        tx_status = tx_students.get("Status", pd.Series("", index=tx_students.index)).astype(str).str.strip().str.lower()
        tx_paid_ids = set(tx_students.loc[tx_status.eq("admitted"), "student_id"].astype(str))

    rows = []
    seen = set()
    for sheet in batch_sheets:
        df = data.get("activities", {}).get(sheet, pd.DataFrame())
        if df is None or df.empty:
            continue
        for _, r in df.iterrows():
            sid = _student_id_from_values(r.get("email_key", ""), r.get("student_key", ""), r.get("student_name", ""))
            if not sid or sid in seen or sid in tx_paid_ids:
                continue
            status = clean_text(r.get("sheet_status_raw", r.get("paid_label", "")))
            if status.lower() == "admitted" or bool(r.get("sheet_is_paid", False)):
                continue
            rows.append({
                "student_id": sid,
                "Student Name": clean_text(r.get("student_name", "")),
                "Email": clean_text(r.get("email_key", "")),
                "Status": status,
                "Batch": clean_text(r.get("Batch", infer_batch_group_from_sheet_name(sheet))),
                "Program": program,
                "student_key": clean_text(r.get("student_key", "")),
                "email_key": clean_text(r.get("email_key", "")),
            })
            seen.add(sid)
    return pd.DataFrame(rows)


def _unpaid_event_matrix(data, mode, program="UG"):
    program = str(program).upper()
    batch_sheets = UG_BATCH_SHEETS if program == "UG" else PG_BATCH_SHEETS
    students = _unpaid_student_frame(data, program)
    categories = ["AMA Welcome Webinar", "AMA Pratham", "AMA Tarun", "AMA Amitoj", "AMA Garima", "AMA Capstone", "AMA Life at Tetr", "Masterclass", "Competition", "Hackathon"]
    if students.empty:
        return pd.DataFrame(columns=["Student Name", "Email", "Status"] + categories), pd.DataFrame()

    dates_df = data.get("dates_df", pd.DataFrame())
    student_info = {}
    for _, r in students.iterrows():
        drow = find_student_dates_row(dates_df, r.get("Student Name", ""), r.get("email_key", ""), r.get("student_key", ""), program, r.get("Batch", "")) if dates_df is not None and not dates_df.empty else None
        offered = pd.to_datetime(drow.get("offered_date_parsed", pd.NaT), errors="coerce") if drow is not None else pd.NaT
        deadline = pd.to_datetime(drow.get("deadline_parsed", pd.NaT), errors="coerce") if drow is not None else pd.NaT
        student_info[r["student_id"]] = {
            "name": r.get("Student Name", ""),
            "email": r.get("Email", ""),
            "status": r.get("Status", ""),
            "offered": offered.normalize() if pd.notna(offered) else pd.NaT,
            "deadline": deadline.normalize() if pd.notna(deadline) else pd.NaT,
        }

    event_rows = []
    eligible_student_ids_by_category = {c: set() for c in categories}
    eligible_student_category = {sid: {c: False for c in categories} for sid in student_info.keys()}

    for sheet in batch_sheets:
        df = data.get("activities", {}).get(sheet, pd.DataFrame())
        ctx = data.get("activity_ctx", {}).get(sheet, {})
        ev_info = ctx.get("event_info", pd.DataFrame())
        if df is None or df.empty or ev_info is None or ev_info.empty:
            continue
        for _, ev in ev_info.iterrows():
            col = ev.get("column_name")
            if col not in df.columns:
                continue
            category = _categorize_activities_ama(ev.get("event_name", ""), ev.get("event_type", ""))
            if not category:
                continue
            ev_date = pd.to_datetime(ev.get("event_date", pd.NaT), errors="coerce")
            ev_date = ev_date.normalize() if pd.notna(ev_date) else pd.NaT

            for sid, info in student_info.items():
                if pd.isna(ev_date):
                    eligible = False
                elif mode == "First 30 Days":
                    offered, deadline = info.get("offered", pd.NaT), info.get("deadline", pd.NaT)
                    eligible = pd.notna(offered) and pd.notna(deadline) and offered <= ev_date <= deadline
                else:  # All
                    eligible = True
                if eligible:
                    eligible_student_ids_by_category[category].add(sid)
                    eligible_student_category[sid][category] = True

            attended = pd.to_numeric(df[col], errors="coerce").fillna(0).gt(0)
            for _, sr in df.loc[attended].iterrows():
                sid = _student_id_from_values(sr.get("email_key", ""), sr.get("student_key", ""), sr.get("student_name", ""))
                if sid not in student_info:
                    continue
                info = student_info[sid]
                if pd.isna(ev_date):
                    include = False
                elif mode == "First 30 Days":
                    offered, deadline = info.get("offered", pd.NaT), info.get("deadline", pd.NaT)
                    include = pd.notna(offered) and pd.notna(deadline) and offered <= ev_date <= deadline
                else:
                    include = True
                if not include:
                    continue
                eligible_student_ids_by_category[category].add(sid)
                eligible_student_category[sid][category] = True
                dedupe = f"{sid}|{_event_name_key(ev.get('event_name',''))}|{category}|{ev_date.strftime('%Y-%m-%d') if pd.notna(ev_date) else ''}"
                event_rows.append({"student_id": sid, "category": category, "dedupe_key": dedupe})

    events = pd.DataFrame(event_rows)
    if not events.empty:
        events = events.drop_duplicates("dedupe_key")

    rows = []
    for sid, info in student_info.items():
        row = {"Student Name": info["name"], "Email": info["email"], "Status": info["status"]}
        counts = events[events["student_id"].eq(sid)]["category"].value_counts() if not events.empty else pd.Series(dtype=int)
        for c in categories:
            row[c] = int(counts.get(c, 0)) if eligible_student_category.get(sid, {}).get(c, False) else "N/A"
        if "Competition" in row and "Hackathon" in row:
            av = pd.to_numeric(pd.Series([row["Competition"]]), errors="coerce").iloc[0]
            bv = pd.to_numeric(pd.Series([row["Hackathon"]]), errors="coerce").iloc[0]
            row["Competition & Hackathon"] = "N/A" if pd.isna(av) and pd.isna(bv) else int((0 if pd.isna(av) else av) + (0 if pd.isna(bv) else bv))
        ama_masterclass_cols = [c for c in AMA_ACTIVITY_COLUMNS + ["Masterclass"] if c in row]
        if ama_masterclass_cols:
            row[AMA_MASTERCLASS_COLUMN] = _combine_activity_columns_for_row(row, ama_masterclass_cols)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("Student Name").reset_index(drop=True)

    summary_rows = []
    for c in categories:
        numeric_counts = pd.to_numeric(table[c], errors="coerce") if c in table.columns else pd.Series(dtype=float)
        eligible = len(eligible_student_ids_by_category.get(c, set()))
        unique_attended = int(numeric_counts.fillna(0).gt(0).sum()) if len(numeric_counts) else 0
        summary_rows.append({
            "Activity Column": c,
            "Total Attendance": int(numeric_counts.fillna(0).sum()) if len(numeric_counts) else 0,
            "Unique Students Attended": unique_attended,
            "Eligible Unique Students": eligible,
            "% of Activity-Type Eligible Students": (unique_attended / eligible * 100) if eligible else 0.0,
        })
    if "Competition & Hackathon" in table.columns:
        summary_rows.append(_combined_activity_summary_row(
            table, eligible_student_ids_by_category, ["Competition", "Hackathon"], "Competition & Hackathon"
        ))
    if AMA_MASTERCLASS_COLUMN in table.columns:
        summary_rows.append(_combined_activity_summary_row(
            table, eligible_student_ids_by_category, [c for c in AMA_ACTIVITY_COLUMNS + ["Masterclass"] if c in categories], AMA_MASTERCLASS_COLUMN
        ))
    return table, pd.DataFrame(summary_rows)


def _render_unpaid_activity_matrix(data, program, title, key_prefix):
    st.markdown(f"### {title} Activity Matrix")
    mode = st.radio("Select View", ["First 30 Days", "All"], horizontal=True, key=f"{key_prefix}_mode")
    matrix, summary = _unpaid_event_matrix(data, mode, program)
    if matrix.empty:
        st.info(f"No {title} student data found.")
        return
    st.metric("Total Unpaid / Non-Admitted Students", f"{len(matrix):,}")
    attendance_cols = [c for c in ["AMA Welcome Webinar", "AMA Pratham", "AMA Tarun", "AMA Amitoj", "AMA Garima", "AMA Capstone", "AMA Life at Tetr", "Masterclass", "AMAs & Masterclasses", "Competition", "Hackathon", "Competition & Hackathon"] if c in matrix.columns]

    def _highlight_activity_nonzero(val):
        try:
            return "background-color: #dff3e7; color: #0b3d2e; font-weight: 700;" if float(val) > 0 else ""
        except Exception:
            return ""

    try:
        st.dataframe(matrix.style.applymap(_highlight_activity_nonzero, subset=attendance_cols), use_container_width=True, hide_index=True, height=420, key=f"{key_prefix}_matrix_{mode}")
    except Exception:
        st.dataframe(matrix, use_container_width=True, hide_index=True, height=420, key=f"{key_prefix}_matrix_{mode}")

    st.markdown("#### Attendance Summary")
    sdisp = summary.copy()
    if not sdisp.empty and "% of Activity-Type Eligible Students" in sdisp.columns:
        sdisp["% of Activity-Type Eligible Students"] = sdisp["% of Activity-Type Eligible Students"].map(lambda x: f"{x:.1f}%")
    st.dataframe(sdisp, use_container_width=True, hide_index=True, key=f"{key_prefix}_summary_{mode}")

    numeric_matrix = matrix.copy()
    for col in attendance_cols:
        numeric_matrix[col] = pd.to_numeric(numeric_matrix[col], errors="coerce").fillna(0)
    ama_cols = [c for c in ["AMA Welcome Webinar", "AMA Pratham", "AMA Tarun", "AMA Amitoj", "AMA Garima", "AMA Capstone", "AMA Life at Tetr"] if c in numeric_matrix.columns]
    avg_ama = numeric_matrix[ama_cols].sum(axis=1).mean() if ama_cols else 0.0
    avg_masterclass = numeric_matrix["Masterclass"].mean() if "Masterclass" in numeric_matrix else 0.0
    avg_ama_masterclass = numeric_matrix[AMA_MASTERCLASS_COLUMN].mean() if AMA_MASTERCLASS_COLUMN in numeric_matrix.columns else ((numeric_matrix[ama_cols].sum(axis=1) + (numeric_matrix["Masterclass"] if "Masterclass" in numeric_matrix else 0)).mean() if ama_cols else avg_masterclass)
    avg_comp_masterclass = numeric_matrix["Competition & Hackathon"].mean() if "Competition & Hackathon" in numeric_matrix else 0.0
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Average AMA's Attended", f"{avg_ama:.2f}")
    c2.metric("Average AMAs & Masterclass Attended", f"{avg_ama_masterclass:.2f}")
    c3.metric("Average Competition & Hackathon", f"{avg_comp_masterclass:.2f}")
    c4.metric("Average Masterclass Attended", f"{avg_masterclass:.2f}")
    c5.metric("Average Competition Attended", f"{numeric_matrix['Competition'].mean():.2f}" if "Competition" in numeric_matrix else "0.00")
    c6.metric("Average Hackathon Attended", f"{numeric_matrix['Hackathon'].mean():.2f}" if "Hackathon" in numeric_matrix else "0.00")

    if not summary.empty:
        chart_df = summary.copy()
        pct_col = "% of Activity-Type Eligible Students"
        if pct_col in chart_df.columns:
            chart_df[pct_col] = pd.to_numeric(chart_df[pct_col], errors="coerce").fillna(0)
            chart_df["Percentage Label"] = chart_df[pct_col].map(lambda x: f"{x:.1f}%")
            fig = px.bar(
                chart_df,
                x="Activity Column",
                y=pct_col,
                text="Percentage Label",
                title=f"{title} Attendance % by Activity · {mode}",
                hover_data={
                    "Activity Column": False,
                    pct_col: ":.1f",
                    "Unique Students Attended": True,
                    "Eligible Unique Students": True,
                    "Total Attendance": True,
                    "Percentage Label": False,
                },
            )
            fig.update_yaxes(title="% of Activity-Type Eligible Students")
        else:
            fig = px.bar(chart_df, x="Activity Column", y="Total Attendance", text="Total Attendance", title=f"{title} Attendance by Activity · {mode}")
        fig.update_traces(marker_color=GREEN, textposition="outside")
        st.plotly_chart(nice_layout(fig, height=360, x_tickangle=-25), use_container_width=True, key=f"{key_prefix}_chart_{mode}")


def _render_attendance_distribution_toggle(df, key_prefix):
    if df is None or df.empty:
        return
    dist_cols = ["Event Name", "Date", "UG Attendance Distribution", "PG Attendance Distribution"]
    if not all(c in df.columns for c in dist_cols):
        return
    if st.toggle("Show UG/PG attended batch distribution", value=False, key=f"{key_prefix}_dist_toggle"):
        d = df[dist_cols].copy()
        d["Date"] = pd.to_datetime(d["Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("")
        st.dataframe(d, use_container_width=True, hide_index=True, height=260, key=f"{key_prefix}_dist_table")



def _winner_impact_students_for_scope(data, scope="Total", entry_kind="winner"):
    """Return denominator Tetr-X students and Winner/Spotlight-announced-in-T-7 rows.

    Uses the Winner sheet's Date of Winner Announcement and each Tetr-X student's payment date.
    Window: T-7 through T, inclusive. T is the payment date.
    entry_kind: "winner" or "spotlight".
    """
    scope = clean_text(scope).upper()
    entry_kind = clean_text(entry_kind).lower()
    frames = []
    if scope in {"TOTAL", "UG"}:
        ug = _tetrx_student_frame(data, "UG")
        if ug is not None and not ug.empty:
            ug = ug.copy()
            ug["Program"] = "UG"
            frames.append(ug)
    if scope in {"TOTAL", "PG"}:
        pg = _tetrx_student_frame(data, "PG")
        if pg is not None and not pg.empty:
            pg = pg.copy()
            pg["Program"] = "PG"
            frames.append(pg)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    students = pd.concat(frames, ignore_index=True).drop_duplicates("student_id", keep="first")
    if "payment_date_parsed" in students.columns:
        students["payment_date_norm"] = pd.to_datetime(students["payment_date_parsed"], errors="coerce").dt.normalize()
    else:
        students["payment_date_norm"] = pd.NaT

    winner_df = data.get("winner_df", pd.DataFrame())
    out_cols = [
        "Student Name", "Email", "UG/PG", "Batch", "Payment Date",
        "Challenge Name", "Winner Announcement Date", "Winner/Spotlight", "Amount in USD"
    ]
    if winner_df is None or winner_df.empty:
        return students, pd.DataFrame(columns=out_cols)
    w = winner_df.copy()
    if "announcement_date" not in w.columns:
        w["announcement_date"] = pd.NaT
    w["announcement_date"] = pd.to_datetime(w["announcement_date"], errors="coerce").dt.normalize()

    if entry_kind == "spotlight":
        if "is_spotlight" in w.columns:
            w = w[w["is_spotlight"].fillna(False).astype(bool)].copy()
        elif "entry_type" in w.columns:
            w = w[w["entry_type"].astype(str).str.strip().str.lower().eq("spotlight")].copy()
    else:
        if "is_winner" in w.columns:
            w = w[w["is_winner"].fillna(False).astype(bool)].copy()
        elif "entry_type" in w.columns:
            w = w[w["entry_type"].astype(str).str.strip().str.lower().eq("winner")].copy()
    if w.empty:
        return students, pd.DataFrame(columns=out_cols)

    rows = []
    for _, stu in students.iterrows():
        pay_dt = pd.to_datetime(stu.get("payment_date_norm", pd.NaT), errors="coerce")
        if pd.isna(pay_dt):
            continue
        sid_email = clean_text(stu.get("email_key", ""))
        sid_name = clean_text(stu.get("student_key", ""))
        mask = pd.Series(False, index=w.index)
        if sid_email:
            mask = mask | w.get("email_key", pd.Series("", index=w.index)).astype(str).eq(sid_email)
        if sid_name:
            mask = mask | w.get("student_key", pd.Series("", index=w.index)).astype(str).eq(sid_name)
        cand = w.loc[mask].copy()
        if cand.empty:
            continue
        stu_batch_key = normalize_batch_token(stu.get("Batch", "")) if "Batch" in stu.index else ""
        if stu_batch_key and "batch_key" in cand.columns:
            narrowed = cand[cand["batch_key"].astype(str).eq(stu_batch_key)].copy()
            if not narrowed.empty:
                cand = narrowed
        start = pay_dt - pd.Timedelta(days=7)
        end = pay_dt
        cand = cand[cand["announcement_date"].notna() & cand["announcement_date"].between(start, end, inclusive="both")].copy()
        if cand.empty:
            continue
        cand["_dedupe"] = cand.apply(
            lambda r: f"{stu.get('student_id','')}|{_event_name_key(r.get('challenge_name',''))}|{pd.to_datetime(r.get('announcement_date'), errors='coerce').strftime('%Y-%m-%d') if pd.notna(pd.to_datetime(r.get('announcement_date'), errors='coerce')) else ''}|{entry_kind}",
            axis=1,
        )
        cand = cand.drop_duplicates("_dedupe")
        for _, r in cand.iterrows():
            # Email can come from Tetr-X parsed fields (email_key) or Winner sheet parsed fields (email_key).
            # Some parsed frames do not retain the original raw email column, so fall back to email_key
            # to avoid blank Email values in Winner Impact.
            stu_email = (
                clean_text(stu.get("email", ""))
                or clean_text(stu.get("email_key", ""))
                or clean_text(r.get("email", ""))
                or clean_text(r.get("email_id", ""))
                or clean_text(r.get("email_key", ""))
            )
            rows.append({
                "Student Name": clean_text(stu.get("student_name", "")) or clean_text(r.get("winner_name", "")),
                "Email": stu_email,
                "UG/PG": clean_text(stu.get("Program", "")),
                "Batch": clean_text(stu.get("Batch", "")) or clean_text(r.get("Batch", "")),
                "Payment Date": pay_dt,
                "Challenge Name": clean_text(r.get("challenge_name", "")),
                "Winner Announcement Date": pd.to_datetime(r.get("announcement_date", pd.NaT), errors="coerce"),
                "Winner/Spotlight": "Spotlight" if entry_kind == "spotlight" else "Winner",
                "Amount in USD": float(pd.to_numeric(pd.Series([r.get("amount_usd", 0)]), errors="coerce").fillna(0).iloc[0]) if entry_kind != "spotlight" else 0.0,
            })
    impact = pd.DataFrame(rows, columns=out_cols)
    if not impact.empty:
        impact = impact.sort_values(["Winner Announcement Date", "Student Name"], ascending=[False, True]).reset_index(drop=True)
    return students, impact


def _format_impact_display(impact):
    display = impact.copy()
    for col in ["Payment Date", "Winner Announcement Date"]:
        if col in display.columns:
            display[col] = pd.to_datetime(display[col], errors="coerce").dt.strftime("%d %b %Y").fillna("")
    if "Email" in display.columns:
        display["Email"] = display["Email"].astype(str).map(clean_text)
    return display


def _render_winner_impact_scope(data, scope, key_prefix):
    students, winner_impact = _winner_impact_students_for_scope(data, scope, entry_kind="winner")
    _, spotlight_impact = _winner_impact_students_for_scope(data, scope, entry_kind="spotlight")
    total_students = int(students["student_id"].nunique()) if students is not None and not students.empty and "student_id" in students.columns else 0
    winner_students = int(winner_impact[["Student Name", "Email"]].drop_duplicates().shape[0]) if winner_impact is not None and not winner_impact.empty else 0
    spotlight_students = int(spotlight_impact[["Student Name", "Email"]].drop_duplicates().shape[0]) if spotlight_impact is not None and not spotlight_impact.empty else 0
    winner_pct = (winner_students / total_students * 100) if total_students else 0.0
    spotlight_pct = (spotlight_students / total_students * 100) if total_students else 0.0
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tetr-X Students", f"{total_students:,}")
    c2.metric("Winner Announced in T-7", f"{winner_students:,}")
    c3.metric("% Winner Announced in T-7", f"{winner_pct:.1f}%")
    c4.metric("Spotlight Announced in T-7", f"{spotlight_students:,}")
    c5.metric("% Spotlight Announced in T-7", f"{spotlight_pct:.1f}%")

    st.markdown("#### Winner Announced in T-7")
    if winner_impact is None or winner_impact.empty:
        st.info(f"No {scope} Tetr-X students had a winner announcement in T-7 to T.")
    else:
        st.dataframe(_format_impact_display(winner_impact), use_container_width=True, hide_index=True, height=300, key=f"{key_prefix}_winner_impact_table")

    st.markdown("#### Spotlight Announced in T-7")
    if spotlight_impact is None or spotlight_impact.empty:
        st.info(f"No {scope} Tetr-X students had a spotlight announcement in T-7 to T.")
    else:
        st.dataframe(_format_impact_display(spotlight_impact), use_container_width=True, hide_index=True, height=300, key=f"{key_prefix}_spotlight_impact_table")


def render_winner_impact_activities(data):
    st.markdown("### Winner Impact")
    st.caption("Uses Tetr-X students only. Winner/Spotlight announcements are counted when the Winner sheet announcement date is within T-7 to T of the student's Tetr-X payment date.")
    total_tab, ug_tab, pg_tab = st.tabs(["Total", "UG", "PG"])
    with total_tab:
        _render_winner_impact_scope(data, "Total", "activities_winner_total")
    with ug_tab:
        _render_winner_impact_scope(data, "UG", "activities_winner_ug")
    with pg_tab:
        _render_winner_impact_scope(data, "PG", "activities_winner_pg")

def render_activities_page(data):
    st.subheader("Activities")
    all_tab, txug_tab, txpg_tab, unpaid_ug_tab, unpaid_pg_tab, winner_impact_tab = st.tabs(["All Events", "Tetr X UG", "Tetr X PG", "Unpaid UG", "Unpaid PG", "Winner Impact"])

    with all_tab:
        st.markdown("### All Events · Online Events + Masterclasses")
        all_events = build_activities_all_events(data)
        if all_events.empty:
            st.info("No online events or masterclasses found.")
        else:
            display = all_events.copy()
            display["Date"] = pd.to_datetime(display["Date"], errors="coerce").dt.strftime("%d %b %Y").fillna("")
            display["Attend to Eligible Ratio"] = display["Attend to Eligible Ratio"].map(lambda x: f"{x:.1f}%")
            display = display.drop(columns=["TetrX Attended"], errors="ignore")
            st.dataframe(display, use_container_width=True, hide_index=True, height=420, key="activities_all_events_table")
            _render_attendance_distribution_toggle(all_events, "activities_all_events")
            chart = all_events.head(30).copy()
            fig = px.bar(chart.sort_values("Date"), x="Date", y="Attend to Eligible Ratio", hover_data=["Event Name", "Audience", "Audience Size", "Total Attendance"], title="Attend to Eligible Ratio · Recent Events")
            fig.update_traces(marker_color=GREEN_2)
            st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-30), use_container_width=True, key="activities_all_events_ratio")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### AMA with Pratham")
            pratham = build_activities_all_events(data, speaker_filter="Pratham")
            if pratham.empty:
                st.info("No Pratham online events found.")
            else:
                pdisp = pratham.copy(); pdisp["Date"] = pd.to_datetime(pdisp["Date"], errors="coerce").dt.strftime("%d %b %Y").fillna(""); pdisp["Attend to Eligible Ratio"] = pdisp["Attend to Eligible Ratio"].map(lambda x: f"{x:.1f}%")
                pdisp = pdisp.drop(columns=["TetrX Attended"], errors="ignore")
                st.dataframe(pdisp, use_container_width=True, hide_index=True, height=260, key="activities_pratham_table")
                _render_attendance_distribution_toggle(pratham, "activities_pratham")
        with c2:
            st.markdown("### AMA with Tarun")
            tarun = build_activities_all_events(data, speaker_filter="Tarun")
            if tarun.empty:
                st.info("No Tarun online events found.")
            else:
                tdisp = tarun.copy(); tdisp["Date"] = pd.to_datetime(tdisp["Date"], errors="coerce").dt.strftime("%d %b %Y").fillna(""); tdisp["Attend to Eligible Ratio"] = tdisp["Attend to Eligible Ratio"].map(lambda x: f"{x:.1f}%")
                tdisp = tdisp.drop(columns=["TetrX Attended"], errors="ignore")
                st.dataframe(tdisp, use_container_width=True, hide_index=True, height=260, key="activities_tarun_table")
                _render_attendance_distribution_toggle(tarun, "activities_tarun")

    with winner_impact_tab:
        render_winner_impact_activities(data)

    with txug_tab:
        _render_tetrx_activity_matrix(data, "UG", "Tetr X UG", "activities_txug")

    with txpg_tab:
        _render_tetrx_activity_matrix(data, "PG", "Tetr X PG", "activities_txpg")

    with unpaid_ug_tab:
        _render_unpaid_activity_matrix(data, "UG", "Unpaid UG", "activities_unpaid_ug")

    with unpaid_pg_tab:
        _render_unpaid_activity_matrix(data, "PG", "Unpaid PG", "activities_unpaid_pg")

# ---------------- Hritabh Page ----------------
HRITABH_AMA_COLUMNS = [
    "Welcome webinar",
    "AMA with Pratham",
    "AMA with Tarun",
    "AMA with Amitoj",
    "AMA with Dr. Garima",
    "AMA with Capstone",
    "AMA with Jessica",
]


def _hritabh_ama_label(event_name, event_type):
    name = clean_text(event_name).lower()
    typ = _activity_event_type_norm(event_type) if "_activity_event_type_norm" in globals() else clean_text(event_type)
    if typ == "Online Event":
        if any(x in name for x in ["shahrose", "welcome webinar", "harshit"]):
            return "Welcome webinar"
        if "pratham" in name:
            return "AMA with Pratham"
        if "tarun" in name:
            return "AMA with Tarun"
        if "amitoj" in name:
            return "AMA with Amitoj"
        if "garima" in name:
            return "AMA with Dr. Garima"
        if any(x in name for x in ["kritee", "ayush", "saarthak", "sarthak"]):
            return "AMA with Capstone"
        if any(x in name for x in ["jessica", "yuliia", "yulia"]):
            return "AMA with Jessica"
    return None


def _hritabh_student_match_mask(df, email_key="", student_key="", mobile_key=""):
    if df is None or df.empty:
        return pd.Series(False, index=[])
    mask = pd.Series(False, index=df.index)
    if email_key and "email_key" in df.columns:
        mask = mask | df["email_key"].astype(str).eq(email_key)
    if student_key and "student_key" in df.columns:
        mask = mask | df["student_key"].astype(str).eq(student_key)
    if mobile_key and "mobile_key" in df.columns:
        mask = mask | df["mobile_key"].astype(str).eq(mobile_key)
    return mask


def _hritabh_relevant_sheets(program):
    program = clean_text(program).upper()
    if program == "UG":
        return UG_BATCH_SHEETS + ["Tetr-X-UG"]
    return PG_BATCH_SHEETS + ["Tetr-X-PG"]


def _hritabh_collect_attendance(data, program, email_key="", student_key="", mobile_key=""):
    """Return attended AMA labels and attended masterclass event names for one student.
    Checks respective batch sheets and Tetr-X sheet; any attendance in any matching row = Yes.
    """
    ama_yes = {c: False for c in HRITABH_AMA_COLUMNS}
    masterclass_yes = {}
    activities = data.get("activities", {}) if isinstance(data, dict) else {}
    contexts = data.get("activity_ctx", {}) if isinstance(data, dict) else {}
    for sheet in _hritabh_relevant_sheets(program):
        df = activities.get(sheet, pd.DataFrame())
        ctx = contexts.get(sheet, {}) if isinstance(contexts, dict) else {}
        if df is None or df.empty:
            continue
        event_info = ctx.get("event_info", pd.DataFrame()) if isinstance(ctx, dict) else pd.DataFrame()
        if event_info is None or event_info.empty:
            continue
        mask = _hritabh_student_match_mask(df, email_key, student_key, mobile_key)
        if not mask.any():
            continue
        sub = df.loc[mask].copy()
        for _, ev in event_info.iterrows():
            col = ev.get("column_name", "")
            if not col or col not in sub.columns:
                continue
            attended = pd.to_numeric(sub[col], errors="coerce").fillna(0).gt(0).any()
            if not attended:
                continue
            ev_name = clean_text(ev.get("event_name", "")) or clean_text(col)
            ev_type = _activity_event_type_norm(ev.get("event_type", "")) if "_activity_event_type_norm" in globals() else clean_text(ev.get("event_type", ""))
            ama_label = _hritabh_ama_label(ev_name, ev_type)
            if ama_label:
                ama_yes[ama_label] = True
            if ev_type == "Masterclass":
                # Keep a clean, stable column label; same name from multiple sheets remains one column.
                masterclass_yes[ev_name] = True
    return ama_yes, masterclass_yes


def _hritabh_master_rows(data, program):
    sheet = "Master UG" if clean_text(program).upper() == "UG" else "Master PG"
    df = data.get("masters", {}).get(sheet, pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
    ctx = data.get("master_ctx", {}).get(sheet, {}) if isinstance(data, dict) else {}
    if df is None or df.empty:
        return pd.DataFrame(), {}
    return df.copy(), ctx if isinstance(ctx, dict) else {}


def _hritabh_build_tables(data, program):
    master, ctx = _hritabh_master_rows(data, program)
    base_cols = ["Name", "Email", "Contact Number", "Batch"]
    if master is None or master.empty:
        return pd.DataFrame(columns=base_cols + HRITABH_AMA_COLUMNS), pd.DataFrame(columns=base_cols + ["Masterclass"])

    email_col = ctx.get("email_col")
    mobile_col = ctx.get("mobile_col")
    batch_col = ctx.get("batch_col")
    ama_rows = []
    mc_rows = []
    all_masterclasses = set()
    temp_mc_rows = []

    for _, r in master.iterrows():
        student_name = clean_text(r.get("student_name", ""))
        email = clean_text(r.get(email_col, "")) if email_col and email_col in r.index else clean_text(r.get("email_key", ""))
        contact = clean_text(r.get(mobile_col, "")) if mobile_col and mobile_col in r.index else clean_text(r.get("mobile_key", ""))
        batch = clean_text(r.get(batch_col, "")) if batch_col and batch_col in r.index else clean_text(r.get("Batch", ""))
        if batch:
            btoken = normalize_batch_token(batch)
            if btoken and not btoken.upper().startswith("UG") and not btoken.upper().startswith("PG"):
                batch = f"{clean_text(program).upper()} {btoken}"
        email_key = clean_text(r.get("email_key", "")) or normalize_email(email)
        student_key = clean_text(r.get("student_key", "")) or normalize_name(student_name)
        mobile_key = clean_text(r.get("mobile_key", "")) or normalize_phone(contact)
        ama_yes, mc_yes = _hritabh_collect_attendance(data, program, email_key, student_key, mobile_key)

        base = {"Name": student_name, "Email": email, "Contact Number": contact, "Batch": batch}
        ama_row = dict(base)
        for col in ["AMA with Amitoj", "AMA with Dr. Garima", "AMA with Pratham", "AMA with Tarun", "AMA with Capstone", "AMA with Jessica", "Welcome webinar"]:
            ama_row[col] = "Yes" if ama_yes.get(col, False) else "No"
        ama_rows.append(ama_row)
        temp_mc_rows.append((base, mc_yes))
        all_masterclasses.update(mc_yes.keys())

    masterclass_cols = sorted([c for c in all_masterclasses if clean_text(c)], key=lambda x: clean_text(x).lower())
    for base, mc_yes in temp_mc_rows:
        row = dict(base)
        any_mc = False
        for col in masterclass_cols:
            val = bool(mc_yes.get(col, False))
            row[col] = "Yes" if val else "No"
            any_mc = any_mc or val
        row["Masterclass"] = "Yes" if any_mc else "No"
        mc_rows.append(row)

    ama_order = ["Name", "Email", "Contact Number", "Batch", "AMA with Amitoj", "AMA with Dr. Garima", "AMA with Pratham", "AMA with Tarun", "AMA with Capstone", "AMA with Jessica", "Welcome webinar"]
    ama_df = pd.DataFrame(ama_rows)
    if not ama_df.empty:
        ama_df = ama_df[[c for c in ama_order if c in ama_df.columns]]
    mc_order = base_cols + masterclass_cols + ["Masterclass"]
    mc_df = pd.DataFrame(mc_rows)
    if not mc_df.empty:
        mc_df = mc_df[[c for c in mc_order if c in mc_df.columns]]
    return ama_df, mc_df


def render_hritabh_page(data):
    st.subheader("Hritabh")
    st.caption("Attendance matrix from Master UG/PG order, matched against respective batch sheets and Tetr-X sheets. A Yes means the student attended at least once in any matching sheet.")
    tabs = st.tabs(["UG", "PG"])
    for tab, program in zip(tabs, ["UG", "PG"]):
        with tab:
            ama_df, mc_df = _hritabh_build_tables(data, program)
            st.markdown(f"### {program} AMA Attendance")
            if ama_df.empty:
                st.info(f"No {program} master records found.")
            else:
                st.dataframe(ama_df, use_container_width=True, hide_index=True, height=520, key=f"hritabh_{program.lower()}_ama_table")
            st.markdown(f"### {program} Masterclass Attendance")
            if mc_df.empty:
                st.info(f"No {program} masterclass attendance records found.")
            else:
                st.dataframe(mc_df, use_container_width=True, hide_index=True, height=560, key=f"hritabh_{program.lower()}_masterclass_table")



# ---------------- Community Impact ----------------

def _community_impact_event_bucket(event_type: str) -> str:
    """Canonical event buckets for Community Impact scoring.

    Community Impact grouping:
    - Online Event + Masterclass are counted together.
    - Competition + Hackathon are counted together.
    - General/Fun includes General, Fun, Fun Task, Quiz, and Poll.
    """
    typ = _activity_event_type_norm(event_type)
    typ_l = clean_text(typ).lower()
    raw_l = clean_text(event_type).lower()
    if typ in {"Online Event", "Masterclass"}:
        return "Online Events & Masterclasses"
    if typ in {"Competition", "Hackathon"}:
        return "Competition"
    if typ in {"General", "Fun", "Fun Task", "Poll", "Quiz"} or any(x in raw_l or x in typ_l for x in ["general", "fun", "poll", "quiz", "fun task"]):
        return "General/Fun"
    return clean_text(typ) or "Other"


def _community_impact_event_breakdown_text(counts: dict) -> str:
    if not counts:
        return ""
    ordered = ["Online Events & Masterclasses", "Competition", "General/Fun", "Other"]
    parts = []
    for k in ordered:
        v = int(counts.get(k, 0) or 0)
        if v:
            parts.append(f"{k}: {v}")
    for k in sorted([x for x in counts.keys() if x not in ordered]):
        v = int(counts.get(k, 0) or 0)
        if v:
            parts.append(f"{k}: {v}")
    return ", ".join(parts)


def _community_challenge_matches_event(challenge_name: str, event_names) -> bool:
    chal = clean_text(challenge_name)
    if not chal:
        return False
    ck = normalize_name(chal)
    ck_light = _event_name_key(chal)
    if len(ck) < 3 and len(ck_light) < 3:
        return False
    for ev in event_names:
        evs = clean_text(ev)
        if not evs:
            continue
        ek = normalize_name(evs)
        ek_light = _event_name_key(evs)
        if ck and ek and (ck == ek or ck in ek or ek in ck):
            return True
        if ck_light and ek_light and (ck_light == ek_light or ck_light in ek_light or ek_light in ck_light):
            return True
    return False


def _community_pre_payment_winner_count(data: dict, row: pd.Series, pay_dt) -> int:
    """Winner/Spotlight count for Community Impact.

    Rule: mark/count Winner or Spotlight rows for the student from the Winner sheet when
    the announcement date is on/before payment date + 10 days. We do not require matching
    the winner challenge to a pre-payment participation event.
    """
    winner_df = data.get("winner_df", pd.DataFrame()) if isinstance(data, dict) else pd.DataFrame()
    if winner_df is None or winner_df.empty:
        return 0
    w = winner_df.copy()

    kind_mask = pd.Series(False, index=w.index)
    if "is_winner" in w.columns:
        kind_mask = kind_mask | w["is_winner"].fillna(False).astype(bool)
    if "is_spotlight" in w.columns:
        kind_mask = kind_mask | w["is_spotlight"].fillna(False).astype(bool)
    if kind_mask.any():
        w = w.loc[kind_mask].copy()
    if w.empty:
        return 0

    email = clean_text(row.get("Email", "")) or clean_text(row.get("email_key", ""))
    name_key = clean_text(row.get("student_key", "")) or normalize_name(row.get("Name", ""))
    mask = pd.Series(False, index=w.index)
    if email and "email_key" in w.columns:
        mask = mask | w["email_key"].astype(str).map(clean_text).eq(email)
    if name_key and "student_key" in w.columns:
        mask = mask | w["student_key"].astype(str).map(clean_text).eq(name_key)
    w = w.loc[mask].copy()
    if w.empty:
        return 0

    pay_dt = pd.to_datetime(pay_dt, errors="coerce")
    if pd.isna(pay_dt) or "announcement_date" not in w.columns:
        return 0
    ann = pd.to_datetime(w["announcement_date"], errors="coerce")
    w = w.loc[ann.notna() & ann.le(pay_dt.normalize() + pd.Timedelta(days=10))].copy()
    if w.empty:
        return 0
    # Count distinct winner/spotlight records; if challenge names/dates repeat, count once.
    dedupe_cols = [c for c in ["challenge_key", "announcement_date", "kind", "Challenge name", "Winner/Spotlight"] if c in w.columns]
    if dedupe_cols:
        return int(w.drop_duplicates(subset=dedupe_cols).shape[0])
    return int(len(w))

def _community_impact_paid_students(data: dict) -> pd.DataFrame:
    """Paid cohort for Community Impact.

    Source of truth: Tetr-X sheets. Include Admitted + Deferral students, exclude any refund rows.
    Payment date is the Tetr-X payment_date_parsed field. Touchpoints are pre-payment batch-sheet
    activities, deduped by student + event name + event type + date.
    """
    rows = []
    activities = data.get("activities", {}) or {}
    activity_ctx = data.get("activity_ctx", {}) or {}
    dates_df = data.get("dates_df", pd.DataFrame())

    for tx_sheet in TX_SHEETS:
        tx = activities.get(tx_sheet, pd.DataFrame())
        if tx is None or tx.empty:
            continue
        ctx = activity_ctx.get(tx_sheet, {}) or {}
        program = "UG" if tx_sheet.endswith("UG") else "PG"
        country_col = ctx.get("country_col")
        income_col = ctx.get("income_col")
        mobile_col = ctx.get("mobile_col")
        for _, r in tx.iterrows():
            name = clean_text(r.get("student_name", ""))
            email = clean_text(r.get("email_key", ""))
            student_key = clean_text(r.get("student_key", "")) or normalize_name(name)
            sid = clean_text(student_unique_id_from_row(r))
            if not sid:
                continue
            status_raw = clean_text(r.get("sheet_status_raw", ""))
            status_l = status_raw.lower()
            is_refund = "refund" in status_l
            is_paid_or_deferral = status_l.strip() == "admitted" or "deferral" in status_l
            if is_refund or not is_paid_or_deferral:
                continue
            pay_dt = pd.to_datetime(r.get("payment_date_parsed", pd.NaT), errors="coerce")
            batch = clean_text(r.get("Batch", ""))
            country = clean_text(r.get(country_col, "")) if country_col and country_col in r.index else clean_text(r.get("Country", ""))
            income = clean_text(r.get(income_col, "")) if income_col and income_col in r.index else clean_text(r.get("Income", ""))
            phone = clean_text(r.get(mobile_col, "")) if mobile_col and mobile_col in r.index else clean_text(r.get("mobile_key", ""))
            drow = find_student_dates_row(dates_df, name, email, student_key, program, batch) if dates_df is not None and not dates_df.empty else None
            offered_dt = pd.to_datetime(drow.get("offered_date_parsed", pd.NaT), errors="coerce") if drow is not None else pd.NaT
            deadline_dt = pd.to_datetime(drow.get("deadline_parsed", pd.NaT), errors="coerce") if drow is not None else pd.NaT
            rows.append({
                "student_id": sid,
                "Name": name,
                "Email": email,
                "Country": country,
                "Income": income,
                "Phone": phone,
                "UG/PG": program,
                "Batch": batch,
                "Status": status_raw,
                "Payment Date": pay_dt,
                "Offered Date": offered_dt,
                "Deadline": deadline_dt,
                "source_sheet": tx_sheet,
                "student_key": student_key,
                "email_key": email,
            })

    cohort = pd.DataFrame(rows)
    if cohort.empty:
        return pd.DataFrame(columns=["student_id", "Name", "Email", "Country", "UG/PG", "Batch", "Payment Date", "Offered Date", "Total Touchpoints (n)", "Impact score", "Impact"])

    # Deduplicate by student id; retain earliest usable payment date and first non-empty descriptors.
    out_rows = []
    for sid, g in cohort.groupby("student_id", dropna=False):
        g = g.copy()
        pay_dates = pd.to_datetime(g["Payment Date"], errors="coerce").dropna()
        offer_dates = pd.to_datetime(g["Offered Date"], errors="coerce").dropna()
        first = g.iloc[0].copy()
        first["Payment Date"] = pay_dates.min() if not pay_dates.empty else pd.NaT
        first["Offered Date"] = offer_dates.min() if not offer_dates.empty else pd.NaT
        for col in ["Name", "Email", "Country", "Income", "Phone", "Batch", "Status", "source_sheet", "student_key", "email_key", "UG/PG"]:
            vals = [clean_text(x) for x in g[col].tolist() if clean_text(x)] if col in g.columns else []
            if vals:
                first[col] = vals[0]
        out_rows.append(first.to_dict())
    cohort = pd.DataFrame(out_rows).reset_index(drop=True)

    touchpoints = []
    event_breakdowns = []
    pre_payment_winner_counts = []
    online_masterclass_counts = []
    competition_counts = []
    general_fun_counts = []

    for _, r in cohort.iterrows():
        pay_dt = pd.to_datetime(r.get("Payment Date", pd.NaT), errors="coerce")
        offered_dt = pd.to_datetime(r.get("Offered Date", pd.NaT), errors="coerce")
        deadline_dt = pd.to_datetime(r.get("Deadline", pd.NaT), errors="coerce")
        ev = collect_student_profile_events(
            data,
            clean_text(r.get("email_key", "")),
            clean_text(r.get("student_key", "")),
            clean_text(r.get("Name", "")),
            pay_dt=pay_dt,
            offered_dt=offered_dt,
            deadline_dt=deadline_dt,
        )
        counts = {}
        prepay = pd.DataFrame()
        if ev is not None and not ev.empty and pd.notna(pay_dt):
            ev2 = ev.copy()
            ev2["event_date"] = pd.to_datetime(ev2.get("event_date", pd.NaT), errors="coerce")
            source = ev2.get("source_group", pd.Series("", index=ev2.index)).astype(str).str.lower()
            # Pre-payment touchpoints come from batch sheets and include payment-date activities because we do not have times.
            mask = source.str.contains("batch", na=False) & ev2["event_date"].notna() & ev2["event_date"].le(pay_dt.normalize())
            prepay = ev2.loc[mask].copy()
            if not prepay.empty and "dedupe_key" in prepay.columns:
                prepay = prepay.drop_duplicates(subset=["dedupe_key"]).copy()
            for typ in prepay.get("event_type", pd.Series(dtype=str)).tolist():
                bucket = _community_impact_event_bucket(typ)
                counts[bucket] = int(counts.get(bucket, 0)) + 1
            n = int(prepay["dedupe_key"].nunique()) if (not prepay.empty and "dedupe_key" in prepay.columns) else int(len(prepay))
        else:
            n = 0

        touchpoints.append(n)
        event_breakdowns.append(_community_impact_event_breakdown_text(counts))
        pre_payment_winner_counts.append(_community_pre_payment_winner_count(data, r, pay_dt))
        online_masterclass_counts.append(int(counts.get("Online Events & Masterclasses", 0)))
        competition_counts.append(int(counts.get("Competition", 0)))
        general_fun_counts.append(int(counts.get("General/Fun", 0)))

    cohort["Total Touchpoints (n)"] = touchpoints
    cohort["Event Breakdown"] = event_breakdowns
    cohort["Pre-Payment Winner"] = pre_payment_winner_counts
    cohort["_OnlineMasterclass Count"] = online_masterclass_counts
    cohort["_Competition Count"] = competition_counts
    cohort["_General/Fun Count"] = general_fun_counts

    def _baseline_impact(n):
        try:
            n = int(n)
        except Exception:
            n = 0
        if n <= 0:
            return 0.0, "No Impact"
        if n <= 3:
            return 0.33, "Low Impact"
        if n <= 7:
            return 0.66, "Medium Impact"
        return 1.0, "High Impact"

    def _final_impact(row):
        def _int0(v):
            v = pd.to_numeric(v, errors="coerce")
            return 0 if pd.isna(v) else int(v)
        n = _int0(row.get("Total Touchpoints (n)", 0))
        score, impact = _baseline_impact(n)
        om = _int0(row.get("_OnlineMasterclass Count", 0))
        comp = _int0(row.get("_Competition Count", 0))
        gen = _int0(row.get("_General/Fun Count", 0))
        winner_count = _int0(row.get("Pre-Payment Winner", 0))
        non_general_count = max(0, n - gen)
        three_all_non_general = n == 3 and gen == 0
        has_non_general = non_general_count >= 1

        # Upgrade to High Impact.
        if n > 0 and (om >= 5 or comp >= 5):
            return 1.0, "High Impact"
        if winner_count >= 1 and non_general_count >= 3:
            return 1.0, "High Impact"
        if winner_count >= 2 and non_general_count >= 1:
            return 1.0, "High Impact"
        if impact == "Medium Impact" and winner_count > 2:
            return 1.0, "High Impact"
        if impact == "Medium Impact" and n in {6, 7} and winner_count >= 1:
            return 1.0, "High Impact"

        # Upgrade Low to Medium when not already upgraded to High.
        # Keep only: exactly 3 attended events and all are non-General/Fun, OR winner count >= 1 with at least one non-General/Fun event.
        if impact == "Low Impact" and three_all_non_general:
            return 0.66, "Medium Impact"
        if impact == "Low Impact" and winner_count >= 1 and has_non_general:
            return 0.66, "Medium Impact"
        return score, impact

    vals = cohort.apply(_final_impact, axis=1)
    cohort["Impact score"] = vals.apply(lambda x: x[0])
    cohort["Impact"] = vals.apply(lambda x: x[1])
    cohort["Days from Offer to Payment"] = (pd.to_datetime(cohort["Payment Date"], errors="coerce") - pd.to_datetime(cohort["Offered Date"], errors="coerce")).dt.days

    def _pay_bucket(days):
        if pd.isna(days):
            return "Unknown"
        try:
            d = int(days)
        except Exception:
            return "Unknown"
        if d <= 5:
            return "1-5"
        if d <= 10:
            return "6-10"
        if d <= 15:
            return "11-15"
        if d <= 20:
            return "16-20"
        if d <= 25:
            return "21-25"
        if d <= 30:
            return "26-30"
        return "30+"
    cohort["Payment After Offer Range"] = cohort["Days from Offer to Payment"].apply(_pay_bucket)
    return cohort


def _format_community_impact_table(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in ["Payment Date", "Offered Date", "Deadline"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime("%d %b %Y").fillna("")
    cols = ["Name", "Email", "Country", "UG/PG", "Batch", "Payment Date", "Offered Date", "Total Touchpoints (n)", "Event Breakdown", "Pre-Payment Winner", "Impact score", "Impact"]
    return out[[c for c in cols if c in out.columns]].sort_values(["Impact score", "Total Touchpoints (n)", "Name"], ascending=[False, False, True])


def _render_community_impact_scope(df: pd.DataFrame, scope_label: str, key_prefix: str):
    st.markdown(f"### {scope_label}")
    if df is None or df.empty:
        st.info(f"No paid/admitted/deferral non-refund students found for {scope_label}.")
        return

    total_students = int(df["student_id"].nunique()) if "student_id" in df.columns else int(len(df))
    total_score = float(pd.to_numeric(df.get("Impact score", 0), errors="coerce").fillna(0).sum())
    impact_counts = df.get("Impact", pd.Series(dtype=str)).value_counts().to_dict()
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Paid Students", f"{total_students:,}")
    c2.metric("Total Impact Score", f"{total_score:.2f}")
    c3.metric("No Impact", f"{int(impact_counts.get('No Impact', 0)):,}")
    c4.metric("Low Impact", f"{int(impact_counts.get('Low Impact', 0)):,}")
    c5.metric("Medium Impact", f"{int(impact_counts.get('Medium Impact', 0)):,}")
    c6.metric("High Impact", f"{int(impact_counts.get('High Impact', 0)):,}")

    st.markdown("#### Student-Level Community Impact")
    display = _format_community_impact_table(df)
    st.dataframe(display, use_container_width=True, hide_index=True, height=430, key=f"{key_prefix}_impact_students")

    imp_order = ["No Impact", "Low Impact", "Medium Impact", "High Impact"]
    imp_df = pd.DataFrame({"Impact": imp_order, "Admissions": [int(impact_counts.get(x, 0)) for x in imp_order]})
    a, b = st.columns([1, 1])
    with a:
        fig = px.bar(imp_df, x="Impact", y="Admissions", title="Admissions by Community Impact", text="Admissions")
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{key_prefix}_impact_bar")
    with b:
        score_df = df.groupby("Impact", as_index=False)["Impact score"].sum().rename(columns={"Impact score": "Impact Score Sum"})
        score_df["Impact"] = pd.Categorical(score_df["Impact"], categories=imp_order, ordered=True)
        score_df = score_df.sort_values("Impact")
        fig = px.bar(score_df, x="Impact", y="Impact Score Sum", title="Impact Score Contribution", text="Impact Score Sum")
        fig.update_traces(marker_color=GREEN_3, texttemplate="%{text:.2f}")
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{key_prefix}_score_bar")

    st.markdown("#### Payment Timing After Offer")
    pay_cols = ["Name", "Email", "UG/PG", "Batch", "Offered Date", "Payment Date", "Days from Offer to Payment", "Payment After Offer Range"]
    timing = df[[c for c in pay_cols if c in df.columns]].copy()
    for col in ["Payment Date", "Offered Date"]:
        if col in timing.columns:
            timing[col] = pd.to_datetime(timing[col], errors="coerce").dt.strftime("%d %b %Y").fillna("")
    st.dataframe(timing.sort_values(["Payment After Offer Range", "Name"]), use_container_width=True, hide_index=True, height=360, key=f"{key_prefix}_payment_timing_table")

    bucket_order = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "30+", "Unknown"]
    bdf = df.get("Payment After Offer Range", pd.Series(dtype=str)).value_counts().reindex(bucket_order, fill_value=0).reset_index()
    bdf.columns = ["Range", "Students"]
    bdf = bdf[bdf["Students"] > 0].copy()
    if not bdf.empty:
        c, d = st.columns([1, 1])
        with c:
            fig = px.bar(bdf, x="Range", y="Students", title="Students by Payment Timing Range", text="Students")
            fig.update_traces(marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=350), use_container_width=True, key=f"{key_prefix}_payment_timing_bar")
        with d:
            st.dataframe(bdf, use_container_width=True, hide_index=True, height=350, key=f"{key_prefix}_payment_timing_summary")


def render_community_impact_page(data):
    st.subheader("Community Impact")
    st.caption("Paid cohort = Tetr-X students with Status = Admitted or Status containing Deferral, excluding rows whose status contains Refund. Touchpoints count deduped batch-sheet activities attended on/before payment date; winner and event-type overrides are applied before final impact tier.")
    cohort = _community_impact_paid_students(data)
    tabs = st.tabs(["Total", "UG", "PG"])
    with tabs[0]:
        _render_community_impact_scope(cohort, "Total", "community_impact_total")
    with tabs[1]:
        _render_community_impact_scope(cohort[cohort.get("UG/PG", pd.Series(dtype=str)).astype(str).str.upper().eq("UG")].copy() if not cohort.empty else cohort, "UG", "community_impact_ug")
    with tabs[2]:
        _render_community_impact_scope(cohort[cohort.get("UG/PG", pd.Series(dtype=str)).astype(str).str.upper().eq("PG")].copy() if not cohort.empty else cohort, "PG", "community_impact_pg")

def main():
    cfg = resolve_source()
    render_header(cfg)

    with st.sidebar:
        st.markdown("## 🧭 Navigation")
        default_pages = ["Overview", "Recent Activity", "Success Metrics", "Student Profile", "UG", "PG", "Courses", "UG vs PG", "Tetr-X", "T-7 & T+7 Analysis", "Conversion", "Retention", "Refund Analytics", "Community Impact", "Activities", "Hritabh"]
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
    elif page == "Success Metrics":
        render_success_metrics_page(data)
    elif page == "Activities":
        render_activities_page(data)
    elif page == "Student Profile":
        render_student_profile(data)
    elif page == "UG":
        render_program_page("UG Batch Sheets", UG_BATCH_SHEETS, data, "ug")
    elif page == "PG":
        render_program_page("PG Batch Sheets", PG_BATCH_SHEETS, data, "pg")
    elif page == "Courses":
        render_courses_page(data)
    elif page == "UG vs PG":
        render_ug_vs_pg_page(data)
    elif page == "Tetr-X":
        render_tetrx_page(data)
    elif page == "T-7 & T+7 Analysis":
        render_t7_analysis_page(data)
    elif page == "Conversion":
        render_retention_page(data)
    elif page == "Retention":
        render_true_retention_page(data)
    elif page == "Recent Activity":
        render_recent_activity_page(data)
    elif page == "Refund Analytics":
        render_refund_analytics_page(data)
    elif page == "Community Impact":
        render_community_impact_page(data)
    elif page == "Hritabh":
        render_hritabh_page(data)


if __name__ == "__main__":
    main()
