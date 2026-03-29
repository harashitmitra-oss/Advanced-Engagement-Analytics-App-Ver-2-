
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

st.set_page_config(page_title="Tetr Business School Analytics Dashboard", layout="wide")

MASTER_SHEETS = ["Master UG", "Master PG"]
DETAIL_SHEETS = ["UG B9", "UG B8", "UG B7", "UG B6", "PG B5"]
NAV_ITEMS = ["Overview", "Student Profile"] + DETAIL_SHEETS
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

HARDCODED_SHEET_ID = "1By2Zb8vKQnTIQn72JRgyEuuRgO6ZZARCZ1JNklmf25U"
HARDCODED_SERVICE_ACCOUNT_JSON = r"""{
  "type": "service_account",
  "project_id": "strong-summer-488709-b9",
  "private_key_id": "7f0ed5e69e39a9dcea12af767271707fa76232a3",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQDp0aGl5LMQDvmH\nrnrRnVsoJx1Mq/52/L+1dGgpL/RAn42HundcN/D7TBH68YmQkDXyk+ZRf/MtEXfc\naNVA1zL20dtMTQdzrqTTidHL4Q36df3OCmthWIMy5f2q9LJNMPt/aDhlmrbQlb6Q\noBDFwXhCcnwuFzy8tEqBGEVmMIzJBTWsZ9c7iQKeJpB+u6HAd4YJjnHJMdJ/7vM9\nRIjmdcYKXpVLNsCE4md5GlJU0iVxDYXlwkKknPOLdLh+Y/5EjHhtZXSSqdSjntqS\nVJPTsn/fsMqko5CBIEdsWWT6leVvq3HBaVBETs5OrB6vVUkQ0PdwqtKcOMmKTBvE\nWr6f+cK/AgMBAAECggEAA5uSCVbXn6fOsdYMKK/Rq3NVc5lm+iI5BjofLh0cGuMo\nJXzH2rE1ed5ZAkY3zFzTq3dP8AuWKCipo/z6GixrEBcTPlolHOPU9Ag5J6wq5KWg\nnoBxDhxCtsZVsHP8f4E1LQIVD+Ks1dfgDA5sM9Qh+hewhgWI9y03X3nAZSzamNkA\n+Ti8iFosVvQ+bbLt5S+M2II46UwCW9lZBR/NHMF2AVp70TxzjyLfxmr8LsZVvr/o\ndclRLW0LAz951cukYEfby5nwLCg37NnFNnXtEQq9mxpQHtZtoKH2W+tp1yTFfiHt\n7qsCm6/u9NE/HjuB3WVCPmcULcA9xXsQLv8WVRHiAQKBgQD/ZsFMbcq1S6VKrETc\nfjEEGVC2B3FBhC2Lyid/z2sr2fHYJTss/FBEgEF2CFst4DDyBz5/odcpmV4zNNXY\nOR+l4kSd7loNMv7yxmhf7ToCgsNu3epVabqZf9w2BI5e2v17D7T+fiIWCuGreHM+\n/+Xisbn8IE6azAx/p4RNzLq/PQKBgQDqXe0vyzxN2NaX9pNuJCMoZ5DRJPZTC2y6\neCvcmDZ11MG1UhELvTh8+V0QGzhAz24yBcLPE2z4DJjhHgrDbmdVt9qeApeoLpM7\nLl5ZBbWXbY4pqYV3rv6kvQ2MSdLorlhGvtU2J457HNp3ZZxYA8XoEnQ8JbuEyci1\nHDZOjrxpqwKBgElWtV0ADfxfW3iE3UU/i021A0MyAeihTv7cLtl5szmlXNgHYOW7\nEkWJWsLNBXm37fYh9GVsEL/mRXGI03tCc/8LaU68eeleYm1OYfxhv42nBP2aBcc/\nFBEt8Qsl5cgBNFaZHQ0TJTCVMVYuwVEu5FFjXZezoz66J0Ck1s4MYve1AoGAagN0\nv/LR43DbmT/bbq4ADU3TrxdmKSh41Vx4kr9zmxdTTD7EIShFvhpaY2e8qWxrL0t1\n1I+38fhYyzP4sHBnY9nXlTQc/+GZjeKqoOA5RTc0YFojWoEZBNHTqArY0ZHTsqSt\n82IvTDdAB1Q6RYHnatO2KmLzENzp4irR0fU0+yMCgYBKpHfgS3PgOX1StB+8lE3B\nLOz7Oj795yrmyR2K+1a+86dHz1EPhxkEP7QExfCwJLSKc3o0OiBDv52KHYj6nrvi\nyQIyxw25Ya2C14qhDPTD6VkN2ia5TSibzO7l0ts8Jwj8wRUj6aT8LPmQ8+VaaPne\nB7AeElVWsTkHo+sHQ4TNgA==\n-----END PRIVATE KEY-----\n",
  "client_email": "tetr-101@strong-summer-488709-b9.iam.gserviceaccount.com",
  "client_id": "110965885023187393080",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/tetr-101%40strong-summer-488709-b9.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}

"""


def safe_get_secrets_dict():
    try:
        return st.secrets
    except Exception:
        return {}


def resolve_streamlit_service_account():
    secrets = safe_get_secrets_dict()
    for key in ("GOOGLE_SERVICE_ACCOUNT", "gcp_service_account"):
        try:
            if key in secrets:
                val = secrets[key]
                if hasattr(val, "to_dict"):
                    val = val.to_dict()
                else:
                    val = dict(val)
                if "private_key" in val:
                    return json.dumps(val)
        except Exception:
            pass
    return None


def resolve_spreadsheet_id_default():
    secrets = safe_get_secrets_dict()
    try:
        return secrets.get("GSHEET_SPREADSHEET_ID", HARDCODED_SHEET_ID)
    except Exception:
        return HARDCODED_SHEET_ID


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
        .profile-card {{
            background: #ffffff;
            border: 1px solid #dbeee0;
            border-radius: 18px;
            padding: 18px 20px;
            box-shadow: 0 4px 14px rgba(11, 61, 46, 0.04);
            margin-bottom: 14px;
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


def normalize_name(s: str) -> str:
    s = clean_text(s).lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def parse_numeric(x):
    if x is None:
        return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = clean_text(x)
    if not s or s.lower() in {"nan", "none", "#div/0!", "inf", "-inf"}:
        return np.nan
    s = s.replace("%", "").replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan


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


def best_matching_index(values, candidates):
    lowered = [clean_text(v).lower() for v in values]
    for cand in candidates:
        for idx, low in enumerate(lowered):
            if cand in low:
                return idx
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


def find_header_row(raw: pd.DataFrame, max_scan=30, is_master=False):
    best_row = None
    best_score = -1

    for i in range(min(max_scan, len(raw))):
        vals = [clean_text(v).lower() for v in raw.iloc[i].tolist()]
        non_empty = sum(1 for v in vals if v)
        if non_empty < 3:
            continue

        score = 0
        joined = " | ".join(vals)

        if any("student name" in v for v in vals):
            score += 10
        if any(v == "name" or v.startswith("name ") for v in vals):
            score += 8
        if any("email" in v for v in vals):
            score += 6
        if any("country" in v for v in vals):
            score += 4
        if any("batch" in v for v in vals):
            score += 4
        if any("status" in v for v in vals):
            score += 4
        if any("payment" in v for v in vals):
            score += 5
        if any("contact" in v or "phone" in v or "mobile" in v for v in vals):
            score += 3
        if is_master and any("admitted group" in v for v in vals):
            score += 3
        if is_master and any("term zero" in v for v in vals):
            score += 2

        # Penalize rows that are mostly numbers or percentages.
        numeric_like = sum(bool(re.fullmatch(r"[\d\.,%]+", v)) for v in vals if v)
        score -= numeric_like * 0.5

        if "name" in joined and "email" in joined and score >= 10:
            return i

        if score > best_score:
            best_score = score
            best_row = i

    return best_row if best_score >= 8 else None


def row_nonempty_count(raw: pd.DataFrame, row_idx: int, start_col: int = 0) -> int:
    if row_idx is None or row_idx < 0 or row_idx >= len(raw):
        return 0
    vals = raw.iloc[row_idx, start_col:].tolist()
    return sum(1 for v in vals if clean_text(v) != "")


def detect_event_rows(raw: pd.DataFrame, header_row: int, event_start: int):
    search_rows = list(range(max(0, header_row - 6), header_row))
    best_date_row = None
    best_date_hits = -1
    for r in search_rows:
        hits = sum(pd.notna(parse_event_date(v)) for v in raw.iloc[r, event_start:].tolist())
        if hits > best_date_hits:
            best_date_hits = hits
            best_date_row = r
    if best_date_hits <= 0:
        best_date_row = None
    best_name_row = None
    if best_date_row is not None:
        candidate_rows = [r for r in search_rows if r < best_date_row]
        best_name_row = max(candidate_rows, key=lambda r: row_nonempty_count(raw, r, event_start), default=None)
        if best_name_row is not None and row_nonempty_count(raw, best_name_row, event_start) == 0:
            best_name_row = None
    return best_name_row, best_date_row


def standardize_numeric_columns(df: pd.DataFrame, columns):
    for col in columns:
        if col and col in df.columns:
            df[col] = df[col].apply(parse_numeric)


def is_paid_status(val) -> bool:
    s = clean_text(val).lower()
    if not s:
        return False
    if "admitted" in s:
        return True
    return bool(re.search(r"\bpaid\b", s))


def clean_student_frame(df: pd.DataFrame, name_col: str, email_col: str | None):
    df = df.dropna(how="all").copy()
    if name_col:
        df = df[df[name_col].astype(str).str.strip().ne("")]
    if email_col and email_col in df.columns:
        temp_email = df[email_col].astype(str).str.strip().str.lower()
        df = df.loc[~((temp_email == "") & (df[name_col].astype(str).str.strip() == ""))]
        df = df.assign(_dedupe_key=np.where(temp_email.ne(""), temp_email, df[name_col].astype(str).map(normalize_name)))
        df = df.drop_duplicates("_dedupe_key", keep="first").drop(columns=["_dedupe_key"])
    else:
        df = df.drop_duplicates(subset=[name_col], keep="first")
    return df.reset_index(drop=True)


def load_master_sheet(raw: pd.DataFrame, program: str):
    header_row = find_header_row(raw, max_scan=40, is_master=True)
    if header_row is None:
        # Fallback for live Google Sheets when header metadata is slightly inconsistent.
        first_rows = min(6, len(raw))
        for i in range(first_rows):
            vals = [clean_text(v).lower() for v in raw.iloc[i].tolist()]
            if any(v in {"name", "student name"} for v in vals) and any("email" in v for v in vals):
                header_row = i
                break
    if header_row is None:
        raise ValueError(f"Could not detect header row in master sheet for {program}.")

    header_vals = raw.iloc[header_row].tolist()
    df = raw.iloc[header_row + 1:].copy().reset_index(drop=True)
    df.columns = make_unique(header_vals)
    df = df.dropna(how="all")

    # Master sheets contain summary rows immediately below the header.
    # Remove leading rows where the name cell is blank or purely numeric.
    preview_name_col = best_matching_col(df, ["student name", "name"])
    if preview_name_col:
        start_idx = 0
        while start_idx < len(df):
            candidate = clean_text(df.iloc[start_idx][preview_name_col])
            if candidate and not re.fullmatch(r"[\d\.]+", candidate):
                break
            start_idx += 1
        if start_idx > 0:
            df = df.iloc[start_idx:].reset_index(drop=True)

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

    df = clean_student_frame(df, name_col, email_col)
    df["Program"] = program
    df["Batch"] = df[batch_col].astype(str).str.strip() if batch_col and batch_col in df.columns else ""

    standardize_numeric_columns(df, [engagement_pct_col, engagement_score_col])
    df["engagement_pct"] = df[engagement_pct_col].fillna(0) if engagement_pct_col else 0
    if pd.to_numeric(df["engagement_pct"], errors="coerce").max() <= 1.05:
        df["engagement_pct"] = pd.to_numeric(df["engagement_pct"], errors="coerce").fillna(0) * 100
    df["engagement_score"] = df[engagement_score_col].fillna(0) if engagement_score_col else 0

    payment_series = df[payment_status_col] if payment_status_col and payment_status_col in df.columns else pd.Series("", index=df.index)
    df["payment_status_clean"] = payment_series.astype(str).str.strip()
    df["is_paid"] = payment_series.apply(is_paid_status)
    df["paid_label"] = np.where(df["is_paid"], "Paid / Admitted", "Not Paid")
    df["is_active"] = pd.to_numeric(df["engagement_pct"], errors="coerce").fillna(0) > 0

    if payment_date_col and payment_date_col in df.columns:
        df[payment_date_col] = df[payment_date_col].apply(parse_date_safe)

    if email_col and email_col in df.columns:
        df["email_clean"] = df[email_col].astype(str).str.strip().str.lower()
    else:
        df["email_clean"] = ""
    df["student_name_clean"] = df[name_col].astype(str).map(normalize_name)

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
    header_row = find_header_row(raw, max_scan=40, is_master=False)
    if header_row is None:
        raise ValueError(f"Could not detect header row in {sheet_name}.")

    header_vals = raw.iloc[header_row].tolist()
    comments_idx = best_matching_index(header_vals, ["comments"])
    if comments_idx is None:
        comments_idx = 19
    event_start = comments_idx + 1

    event_name_row, event_date_row = detect_event_rows(raw, header_row, event_start)

    built_headers = []
    date_map = {}
    for idx, h in enumerate(header_vals):
        header_name = clean_text(h)
        if idx >= event_start and not header_name:
            if event_name_row is not None:
                header_name = clean_text(raw.iloc[event_name_row, idx])
            if not header_name:
                header_name = f"Event {idx - event_start + 1}"
        built_headers.append(header_name)
        if idx >= event_start and event_date_row is not None:
            dt = parse_event_date(raw.iloc[event_date_row, idx])
            if pd.notna(dt):
                date_map[idx] = dt

    df = raw.iloc[header_row + 1:].copy().reset_index(drop=True)
    df.columns = make_unique(built_headers)
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

    df = clean_student_frame(df, name_col, email_col)
    df["Program"] = infer_program_from_sheet(sheet_name)
    df["Batch"] = infer_batch_from_sheet_name(sheet_name)

    standardize_numeric_columns(df, [engagement_pct_col, engagement_score_col])
    event_cols = []
    event_dates = {}
    for idx, col in enumerate(df.columns):
        if idx < event_start:
            continue
        if col.startswith("Unnamed"):
            continue
        ser = df[col] if col in df.columns else pd.Series(dtype=object)
        if is_probably_event_series(ser) or pd.to_numeric(ser.apply(parse_numeric), errors="coerce").fillna(0).isin([0, 1]).mean() > 0.8:
            df[col] = ser.apply(normalize_yes_no).astype(int)
            event_cols.append(col)
            if idx in date_map:
                event_dates[col] = date_map[idx]

    df["participation_count"] = df[event_cols].sum(axis=1) if event_cols else 0
    if engagement_score_col and engagement_score_col in df.columns:
        df["engagement_score"] = pd.to_numeric(df[engagement_score_col], errors="coerce").fillna(df["participation_count"])
    else:
        df["engagement_score"] = df["participation_count"]

    if engagement_pct_col and engagement_pct_col in df.columns:
        df["engagement_pct"] = pd.to_numeric(df[engagement_pct_col], errors="coerce").fillna(0)
        if df["engagement_pct"].max() <= 1.05:
            df["engagement_pct"] = df["engagement_pct"] * 100
    else:
        total_events = max(len(event_cols), 1)
        df["engagement_pct"] = (df["participation_count"] / total_events) * 100

    payment_series = df[payment_status_col] if payment_status_col and payment_status_col in df.columns else pd.Series("", index=df.index)
    df["payment_status_clean"] = payment_series.astype(str).str.strip()
    df["is_paid"] = payment_series.apply(is_paid_status)
    df["paid_label"] = np.where(df["is_paid"], "Paid / Admitted", "Not Paid")
    df["is_active"] = pd.to_numeric(df["engagement_pct"], errors="coerce").fillna(0) > 0

    if payment_date_col and payment_date_col in df.columns:
        df[payment_date_col] = df[payment_date_col].apply(parse_date_safe)

    if email_col and email_col in df.columns:
        df["email_clean"] = df[email_col].astype(str).str.strip().str.lower()
    else:
        df["email_clean"] = ""
    df["student_name_clean"] = df[name_col].astype(str).map(normalize_name)

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


def is_probably_event_series(series: pd.Series) -> bool:
    s = series.fillna("").astype(str).str.strip().str.lower()
    allowed = {"yes", "no", "y", "n", "true", "false", "1", "0", "attended", "present", "absent", "done", "", "nan"}
    return ((s.isin(allowed)).mean() >= 0.55) if len(s) else False


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
    width = max(len(r) for r in values)
    padded = [r + [""] * (width - len(r)) for r in values]
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
    fig = go.Figure(
        go.Indicator(
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
        )
    )
    return nice_layout(fig, height=300)


def donut_chart(labels, values, title):
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.62,
            marker=dict(colors=[GREEN, GREEN_2, GREEN_3, GREEN_4, GREEN_5][: len(labels)]),
            textinfo="label+percent",
        )
    )
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
    m4.metric("UG vs PG", f"{ug_students:,} / {pg_students:,}", delta="UG / PG")

    g1, g2, g3 = st.columns([1.2, 1, 1])
    with g1:
        st.plotly_chart(gauge_chart(total_students, "Total Students", maximum=max(total_students, 1)), use_container_width=True, key=f"{key_prefix}_gauge")
    with g2:
        st.plotly_chart(donut_chart(["UG", "PG"], [ug_students, pg_students], "UG / PG Distribution"), use_container_width=True, key=f"{key_prefix}_ugpg")
    with g3:
        st.plotly_chart(donut_chart(["UG Paid", "PG Paid"], [ug_paid, pg_paid], "Paid Distribution"), use_container_width=True, key=f"{key_prefix}_paid_dist")

    a1, a2 = st.columns(2)
    with a1:
        if batch_col and batch_col in overview_df.columns:
            batch_plot = (
                overview_df.groupby(batch_col, dropna=False)[name_col]
                .count()
                .reset_index(name="Students")
                .sort_values("Students", ascending=False)
            )
            batch_plot[batch_col] = batch_plot[batch_col].replace("", "Unknown")
            fig = px.bar(batch_plot, x=batch_col, y="Students", title="Students by Batch")
            fig.update_traces(marker_color=GREEN_2)
            st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-25), use_container_width=True, key=f"{key_prefix}_batch")
    with a2:
        status_plot = overview_df.groupby(["Program", "paid_label"])[name_col].count().reset_index(name="Students")
        fig = px.bar(
            status_plot,
            x="Program",
            y="Students",
            color="paid_label",
            barmode="group",
            title="Paid vs Not Paid by Program",
            color_discrete_map={"Paid / Admitted": GREEN, "Not Paid": GREEN_4},
        )
        st.plotly_chart(nice_layout(fig, height=380), use_container_width=True, key=f"{key_prefix}_status")

    b1, b2 = st.columns(2)
    with b1:
        if country_col and country_col in overview_df.columns:
            country_plot = (
                overview_df.groupby(country_col)[name_col]
                .count()
                .reset_index(name="Students")
                .sort_values("Students", ascending=False)
                .head(12)
            )
            if not country_plot.empty:
                fig = px.bar(country_plot, x=country_col, y="Students", title="Top Countries")
                fig.update_traces(marker_color=GREEN_3)
                st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-30), use_container_width=True, key=f"{key_prefix}_countries")
    with b2:
        if income_col and income_col in overview_df.columns:
            income_plot = (
                overview_df.groupby(income_col)[name_col]
                .count()
                .reset_index(name="Students")
                .sort_values("Students", ascending=False)
            )
            if not income_plot.empty:
                fig = px.bar(income_plot, x=income_col, y="Students", title="Income Distribution")
                fig.update_traces(marker_color=GREEN)
                st.plotly_chart(nice_layout(fig, height=400, x_tickangle=-25), use_container_width=True, key=f"{key_prefix}_income")

    st.markdown("#### Overview Table")
    preview_cols = [c for c in [name_col, "Program", batch_col, country_col, income_col, "engagement_pct", "engagement_score", "payment_status_clean", "paid_label"] if c and c in overview_df.columns]
    st.dataframe(
        overview_df[preview_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False),
        use_container_width=True,
        height=420,
        key=f"{key_prefix}_table",
    )


def render_detail_tab(sheet_name, df, ctx, key_prefix):
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
    avg_engagement = round(float(pd.to_numeric(df["engagement_pct"], errors="coerce").fillna(0).mean()), 1) if len(df) else 0

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students", f"{total_students:,}")
    k2.metric("Active", f"{active_students:,}", delta=f"{(active_students / total_students * 100 if total_students else 0):.1f}%")
    k3.metric("Paid / Admitted", f"{paid_students:,}", delta=f"{(paid_students / total_students * 100 if total_students else 0):.1f}%")
    k4.metric("Avg Engagement", f"{avg_engagement:.1f}%")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="engagement_pct", nbins=10, title="Engagement Distribution")
        fig.update_traces(marker_color=GREEN_2)
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{key_prefix}_eng_hist")
    with c2:
        status = df["paid_label"].value_counts().reset_index()
        status.columns = ["Status", "Students"]
        fig = px.pie(status, names="Status", values="Students", hole=0.58, color="Status", color_discrete_map={"Paid / Admitted": GREEN, "Not Paid": GREEN_4})
        fig.update_layout(title="Paid Status")
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True, key=f"{key_prefix}_paid_pie")

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
            top_country = (
                df.groupby(country_col)[name_col]
                .count().reset_index(name="Students")
                .sort_values("Students", ascending=False)
                .head(10)
            )
            fig = px.bar(top_country, x=country_col, y="Students", title="Country Split")
            fig.update_traces(marker_color=GREEN_3)
            st.plotly_chart(nice_layout(fig, height=430, x_tickangle=-30), use_container_width=True, key=f"{key_prefix}_country")

    t1, t2 = st.columns(2)
    with t1:
        students = df[[name_col, "engagement_pct", "engagement_score", "payment_status_clean", "paid_label"]].sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
        st.markdown("#### Top Students")
        st.dataframe(students, use_container_width=True, height=390, key=f"{key_prefix}_top_students")
    with t2:
        target = df[(~df["is_paid"]) & (df["engagement_pct"] > 0)][[name_col, "engagement_pct", "engagement_score", "payment_status_clean", "paid_label"]]
        target = target.sort_values(["engagement_pct", "engagement_score"], ascending=False).head(20)
        st.markdown("#### Best Upgrade Targets")
        st.dataframe(target, use_container_width=True, height=390, key=f"{key_prefix}_targets")

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
    event_preview = event_cols[:8]
    display_cols += [c for c in event_preview if c in df.columns]
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
            if key not in profiles:
                profiles[key] = {"master": None, "details": [], "name": clean_text(row[name_col]), "email": clean_text(row.get("email_clean", ""))}
            profiles[key]["master"] = row.to_dict()
            profiles[key]["name"] = clean_text(row[name_col])
            profiles[key]["email"] = clean_text(row.get("email_clean", ""))
    for sheet, df in data["details"].items():
        ctx = data["detail_contexts"][sheet]
        name_col = ctx["name_col"]
        for _, row in df.iterrows():
            key = get_key(row, name_col)
            if key not in profiles:
                profiles[key] = {"master": None, "details": [], "name": clean_text(row[name_col]), "email": clean_text(row.get("email_clean", ""))}
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
    query = st.text_input("Search student name", placeholder="Type a student name")
    matches = [n for n in names if query.lower() in n.lower()] if query else names[:50]
    selected = st.multiselect("Select one or more students", options=matches, default=matches[:1] if query and matches else [])

    bulk = st.text_area("Or paste multiple student names", placeholder="One name per line or comma separated")
    extra_names = []
    if bulk.strip():
        extra_names = [n.strip() for n in re.split(r"[\n,]+", bulk) if n.strip()]
    wanted = list(dict.fromkeys(selected + extra_names))

    if not wanted:
        st.info("Search and select one or more students to view their profiles.")
        return

    lookup = {}
    for key, p in profiles.items():
        lookup.setdefault(normalize_name(p["name"]), []).append(p)

    for wanted_name in wanted:
        found = lookup.get(normalize_name(wanted_name), [])
        if not found:
            st.warning(f"No profile found for {wanted_name}")
            continue

        for idx, prof in enumerate(found, start=1):
            master = prof["master"] or {}
            details = prof["details"]
            title = prof["name"] or wanted_name
            program = clean_text(master.get("Program", "")) or (clean_text(details[0]["row"].get("Program", "")) if details else "")
            batch = clean_text(master.get("Batch", "")) or (clean_text(details[0]["row"].get("Batch", "")) if details else "")
            country = clean_text(master.get("Country", master.get("country", "")))
            email = clean_text(master.get("email_clean", prof.get("email", "")))
            paid = "Paid / Admitted" if bool(master.get("is_paid", False)) else clean_text(master.get("payment_status_clean", ""))
            engagement = master.get("engagement_pct", np.nan)

            st.markdown(f"""
            <div class="profile-card">
                <div style="font-size:28px;font-weight:800;color:{DARK};">{title}</div>
                <div style="margin-top:6px;color:#2a6a52;font-size:15px;">
                    {program or "Program N/A"} • {batch or "Batch N/A"} • {country or "Country N/A"}<br>
                    {email or "Email N/A"}
                </div>
            </div>
            """, unsafe_allow_html=True)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Program", program or "N/A")
            c2.metric("Batch", batch or "N/A")
            c3.metric("Payment", paid or "Not Paid")
            c4.metric("Engagement", f"{float(engagement):.1f}%" if pd.notna(engagement) else "N/A")

            if master:
                master_view = {
                    "Student Name": title,
                    "Program": master.get("Program", ""),
                    "Batch": master.get("Batch", ""),
                    "Country": master.get("Country", master.get("country", "")),
                    "Income": master.get("Income", master.get("income", "")),
                    "Payment Status": master.get("payment_status_clean", ""),
                    "Engagement %": round(float(master.get("engagement_pct", 0) or 0), 2),
                    "Engagement Score": round(float(master.get("engagement_score", 0) or 0), 2),
                    "Active": "Yes" if master.get("is_active", False) else "No",
                }
                st.markdown("#### Master Profile")
                st.dataframe(pd.DataFrame([master_view]), use_container_width=True, key=f"profile_master_{normalize_name(title)}_{idx}")

            if details:
                st.markdown("#### Batch Sheet Details")
                rows = []
                attendance_rows = []
                for entry in details:
                    row = entry["row"]
                    ctx = entry["ctx"]
                    event_cols = ctx["event_cols"]
                    attended = [c for c in event_cols if row.get(c, 0) == 1]
                    rows.append({
                        "Sheet": entry["sheet"],
                        "Program": row.get("Program", ""),
                        "Batch": row.get("Batch", ""),
                        "Payment Status": row.get("payment_status_clean", ""),
                        "Engagement %": round(float(row.get("engagement_pct", 0) or 0), 2),
                        "Engagement Score": round(float(row.get("engagement_score", 0) or 0), 2),
                        "Participation Count": int(row.get("participation_count", 0) or 0),
                        "Events Attended": ", ".join(attended[:8]),
                    })
                    for ev in attended:
                        attendance_rows.append({
                            "Sheet": entry["sheet"],
                            "Event": ev,
                            "Date": ctx["event_dates"].get(ev, pd.NaT),
                        })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, key=f"profile_detail_{normalize_name(title)}_{idx}")
                if attendance_rows:
                    att_df = pd.DataFrame(attendance_rows).sort_values(["Date", "Event"], na_position="last")
                    st.markdown("#### Attendance History")
                    st.dataframe(att_df, use_container_width=True, key=f"profile_att_{normalize_name(title)}_{idx}")


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
            source_choice = st.radio("Source", ["Google Sheets (auto)", "Upload Excel (manual)"], index=0)
    else:
        source_choice = "Upload Excel (manual)"
        with st.sidebar:
            st.warning("Install `gspread` and `google-auth` to enable Google Sheets auto-fetch.")

    if source_choice == "Google Sheets (auto)":
        default_sheet_id = resolve_spreadsheet_id_default()
        with st.sidebar:
            spreadsheet_id = st.text_input("Spreadsheet ID", value=default_sheet_id, help="Google Sheets ID from the URL (pre-filled)")

        secret_creds = resolve_streamlit_service_account()
        credentials_payload = secret_creds if secret_creds else HARDCODED_SERVICE_ACCOUNT_JSON

        try:
            sheet_names = gsheets_get_sheet_names(spreadsheet_id, credentials_payload)
            connected_ok = True
            connection_note = f"Connected to Google Sheets · {len(sheet_names)} tabs found"
        except Exception as e:
            connected_ok = False
            connection_note = f"Connection failed: {e}"
            debug_note = "Check Streamlit logs for the exact Google Sheets auth or access error."
    else:
        uploaded_file = st.file_uploader("Upload Master Engagement Tracker Excel File", type=["xlsx"])
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
    cfg = resolve_credentials_and_source()
    live_mode = cfg["source_mode"] == "gsheets"

    with st.sidebar:
        st.markdown("---")
        st.markdown("### Navigation")
        page = st.radio("Navigation", NAV_ITEMS, index=0, label_visibility="collapsed")

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
        if cfg["debug_note"]:
            st.caption(cfg["debug_note"])
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
