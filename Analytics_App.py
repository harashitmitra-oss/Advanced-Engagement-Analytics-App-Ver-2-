
import re
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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

PAID_LABELS = {"paid", "admitted"}
NON_ACTIVE_STATUS = {"", "nan", "none"}

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
        h1, h2, h3 {{
            color: {DARK} !important;
        }}
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

def best_col(df, keywords):
    scores = []
    for c in df.columns:
        cl = clean_text(c).lower()
        hit = sum(1 for k in keywords if k in cl)
        if hit:
            scores.append((c, hit, df[c].notna().sum()))
    if not scores:
        return None
    scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return scores[0][0]

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
    if isinstance(x, (int, float)) and not pd.isna(x):
        try:
            return pd.to_datetime("1899-12-30") + pd.to_timedelta(float(x), unit="D")
        except Exception:
            pass
    return pd.to_datetime(x, errors="coerce", dayfirst=True)

@st.cache_data(show_spinner=False)
def get_sheet_names(file_bytes):
    return pd.ExcelFile(BytesIO(file_bytes)).sheet_names

@st.cache_data(show_spinner=False)
def read_raw_sheet(file_bytes, sheet_name):
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, header=None).dropna(how="all")

def load_master_sheet(raw_df, program):
    header = [clean_text(x) for x in raw_df.iloc[0].tolist()]
    df = raw_df.iloc[3:].copy().reset_index(drop=True)
    df.columns = make_unique(header)
    df = df.dropna(how="all").copy()

    name_col = best_col(df, ["name"])
    email_col = best_col(df, ["email"])
    batch_col = best_col(df, ["batch"])
    country_col = best_col(df, ["country"])
    status_col = best_col(df, ["status"])
    payment_col = best_col(df, ["payment"])
    income_col = best_col(df, ["income"])
    admitted_group_col = best_col(df, ["admitted group"])

    base_cols = {c for c in [name_col, email_col, batch_col, country_col, status_col, payment_col, income_col, admitted_group_col] if c}
    event_cols = [c for c in df.columns if c not in base_cols]

    if name_col:
        df = df[df[name_col].astype(str).map(clean_text).ne("")].copy()

    for c in event_cols:
        df[c] = df[c].apply(normalize_yes_no).astype(int)

    df["Program"] = program
    if batch_col:
        df["Batch"] = df[batch_col].astype(str).str.replace(".0", "", regex=False).map(clean_text).radd("B")
        df["Batch"] = df["Batch"].str.replace("BB", "B", regex=False)
    else:
        df["Batch"] = ""

    if admitted_group_col:
        admitted_series = df[admitted_group_col].astype(str).map(clean_text).str.lower()
    else:
        admitted_series = pd.Series([""] * len(df), index=df.index)

    status_series = df[status_col].astype(str).map(clean_text).str.lower() if status_col else pd.Series([""] * len(df), index=df.index)
    payment_series = df[payment_col].astype(str).map(clean_text).str.lower() if payment_col else pd.Series([""] * len(df), index=df.index)

    df["engagement_count"] = df[event_cols].sum(axis=1) if event_cols else 0
    df["engagement_pct"] = (df[event_cols].mean(axis=1) * 100).round(1) if event_cols else 0.0
    df["is_active"] = df["engagement_pct"] > 0
    df["is_paid"] = (
        payment_series.str.contains("paid", na=False)
        | status_series.str.contains("admitted", na=False)
        | admitted_series.str.contains("tetr x", na=False)
    )
    df["paid_label"] = np.where(df["is_paid"], "Paid / Admitted", "Not Paid")
    df["status_clean"] = np.where(status_series.eq(""), "Unknown", status_series.str.title())

    ctx = {
        "name_col": name_col,
        "email_col": email_col,
        "country_col": country_col,
        "batch_col": "Batch",
        "status_col": status_col,
        "payment_col": payment_col,
        "event_cols": event_cols,
        "income_col": income_col,
    }
    return df, ctx

def load_detail_sheet(raw_df, sheet_name):
    def row_text(i):
        return " | ".join([clean_text(v).lower() for v in raw_df.iloc[i, :].tolist()])

    header_row = None
    for i in range(min(15, len(raw_df))):
        txt = row_text(i)
        if "student name" in txt or ("name" in txt and "email" in txt and "country" in txt):
            header_row = i
            break
    if header_row is None:
        raise ValueError(f"Could not detect header row in {sheet_name}")

    header = [clean_text(x) for x in raw_df.iloc[header_row].tolist()]
    df = raw_df.iloc[header_row + 1:].copy().reset_index(drop=True)
    df.columns = make_unique(header)
    df = df.dropna(how="all").copy()

    name_col = best_col(df, ["student name", "name"])
    email_col = best_col(df, ["email"])
    country_col = best_col(df, ["country"])
    payment_status_col = best_col(df, ["payment status", "payment"])
    payment_date_col = best_col(df, ["payment date"])
    overall_pct_col = best_col(df, ["overall engagement %"])
    overall_score_col = best_col(df, ["overall engagement score"])

    meta_cols = {c for c in [name_col, email_col, country_col, payment_status_col, payment_date_col, overall_pct_col, overall_score_col] if c}
    event_cols = [c for c in df.columns if c not in meta_cols]

    if name_col:
        df = df[df[name_col].astype(str).map(clean_text).ne("")].copy()

    for c in event_cols:
        if df[c].dtype == object:
            df[c] = df[c].apply(normalize_yes_no)

    if overall_pct_col:
        df["engagement_pct"] = pd.to_numeric(df[overall_pct_col], errors="coerce").fillna(0)
        # if 0-1 scale convert to %
        if df["engagement_pct"].max() <= 1.0:
            df["engagement_pct"] = (df["engagement_pct"] * 100).round(1)
    else:
        df["engagement_pct"] = (pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").fillna(0) for c in event_cols}).mean(axis=1) * 100).round(1) if event_cols else 0

    if overall_score_col:
        df["engagement_score"] = pd.to_numeric(df[overall_score_col], errors="coerce").fillna(0)
    else:
        df["engagement_score"] = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce").fillna(0) for c in event_cols}).sum(axis=1) if event_cols else 0

    payment_series = df[payment_status_col].astype(str).map(clean_text).str.lower() if payment_status_col else pd.Series([""] * len(df), index=df.index)
    df["is_paid"] = payment_series.str.contains("paid|admitted", na=False)
    df["paid_label"] = np.where(df["is_paid"], "Paid / Admitted", "Not Paid")
    df["is_active"] = df["engagement_pct"] > 0
    df["Sheet"] = sheet_name
    df["Program"] = "UG" if sheet_name.startswith("UG") else "PG"

    date_row_idx = header_row - 3 if header_row >= 3 else None
    title_row_idx = header_row - 4 if header_row >= 4 else None
    event_dates = {}
    if date_row_idx is not None:
        for col_name, raw_val in zip(df.columns, raw_df.iloc[date_row_idx].tolist()):
            dt = parse_date_safe(raw_val)
            if pd.notna(dt):
                event_dates[col_name] = dt

    event_type_map = {}
    if title_row_idx is not None:
        for col_name, raw_val in zip(df.columns, raw_df.iloc[title_row_idx].tolist()):
            if col_name in event_cols:
                event_type_map[col_name] = clean_text(raw_val) or "Event"

    ctx = {
        "name_col": name_col,
        "email_col": email_col,
        "country_col": country_col,
        "payment_col": payment_status_col,
        "payment_date_col": payment_date_col,
        "event_cols": event_cols,
        "event_dates": event_dates,
        "event_type_map": event_type_map,
    }
    return df, ctx

def gauge_chart(value, title, suffix=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(value),
        number={"suffix": suffix, "font": {"size": 28, "color": GREEN}},
        title={"text": title, "font": {"size": 16, "color": DARK}},
        gauge={
            "axis": {"range": [0, max(100, value * 1.2 if value else 100)]},
            "bar": {"color": GREEN},
            "bgcolor": "#edf7f1",
            "steps": [
                {"range": [0, max(25, value * 0.35 if value else 25)], "color": "#dff3e7"},
                {"range": [max(25, value * 0.35 if value else 25), max(60, value * 0.7 if value else 60)], "color": "#bfe4cd"},
                {"range": [max(60, value * 0.7 if value else 60), max(100, value * 1.2 if value else 100)], "color": "#8dcca8"},
            ],
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="white")
    return fig

def donut_chart(df, names, values, title, color_map=None):
    fig = px.pie(df, names=names, values=values, hole=0.65, title=title, color=names, color_discrete_map=color_map)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=60, b=10), paper_bgcolor="white", font_color=DARK, showlegend=True)
    return fig

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

@st.cache_data(show_spinner=False)
def load_dashboard_data(file_bytes):
    sheet_names = get_sheet_names(file_bytes)
    missing = [s for s in ALL_REQUIRED if s not in sheet_names]

    masters = {}
    details = {}
    master_contexts = {}
    detail_contexts = {}

    for sheet in MASTER_SHEETS:
        if sheet in sheet_names:
            raw = read_raw_sheet(file_bytes, sheet)
            program = "UG" if sheet.endswith("UG") else "PG"
            masters[sheet], master_contexts[sheet] = load_master_sheet(raw, program)

    for sheet in DETAIL_SHEETS:
        if sheet in sheet_names:
            raw = read_raw_sheet(file_bytes, sheet)
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
        st.plotly_chart(gauge_chart(total_students, "Total Students"), use_container_width=True)
    with g2:
        program_dist = pd.DataFrame({"Program": ["UG", "PG"], "Students": [ug_students, pg_students]})
        st.plotly_chart(
            donut_chart(
                program_dist,
                "Program",
                "Students",
                "UG vs PG Distribution",
                {"UG": GREEN, "PG": GREEN_3},
            ),
            use_container_width=True,
        )
    with g3:
        paid_dist = pd.DataFrame({"Program": ["UG Paid", "PG Paid"], "Students": [ug_paid, pg_paid]})
        st.plotly_chart(
            donut_chart(
                paid_dist,
                "Program",
                "Students",
                "Paid Students Distribution",
                {"UG Paid": GREEN, "PG Paid": GREEN_3},
            ),
            use_container_width=True,
        )

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        batch = (
            overview_df.groupby(["Program", batch_col], dropna=False)
            .agg(Students=(name_col, "count"), Paid=("is_paid", "sum"), Active=("is_active", "sum"))
            .reset_index()
        )
        batch["Active Rate %"] = np.where(batch["Students"] > 0, batch["Active"] / batch["Students"] * 100, 0).round(1)
        batch["Paid Rate %"] = np.where(batch["Students"] > 0, batch["Paid"] / batch["Students"] * 100, 0).round(1)
        batch = batch.sort_values(["Program", batch_col])
        fig = px.bar(
            batch,
            x=batch_col,
            y="Students",
            color="Program",
            barmode="group",
            text_auto=True,
            title="Students by Batch",
            color_discrete_map={"UG": GREEN, "PG": GREEN_3},
        )
        st.plotly_chart(nice_layout(fig, height=380), use_container_width=True)
    with c2:
        status = (
            overview_df.groupby(["Program", "status_clean"], dropna=False)
            .size()
            .reset_index(name="Students")
            .sort_values("Students", ascending=False)
        )
        fig = px.bar(
            status,
            x="status_clean",
            y="Students",
            color="Program",
            barmode="group",
            title="Status Distribution",
            color_discrete_map={"UG": GREEN, "PG": GREEN_3},
        )
        st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-30), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        if country_col and country_col in overview_df.columns:
            top_countries = (
                overview_df.groupby(country_col)
                .agg(Students=(name_col, "count"))
                .reset_index()
                .sort_values("Students", ascending=False)
                .head(12)
            )
            fig = px.bar(top_countries, x=country_col, y="Students", title="Top Countries")
            fig.update_traces(marker_color=GREEN)
            st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-30), use_container_width=True)
    with c4:
        if income_col and income_col in overview_df.columns:
            income = (
                overview_df.groupby(income_col)
                .agg(Students=(name_col, "count"))
                .reset_index()
                .sort_values("Students", ascending=False)
            )
            fig = px.bar(income, x=income_col, y="Students", title="Income Distribution")
            fig.update_traces(marker_color=GREEN_2)
            st.plotly_chart(nice_layout(fig, height=380, x_tickangle=-30), use_container_width=True)

    st.markdown("#### Master-level student table")
    show_cols = [c for c in [name_col, "Program", batch_col, country_col, "status_clean", "paid_label", "engagement_pct", "engagement_count"] if c and c in overview_df.columns]
    table = overview_df[show_cols].sort_values(["Program", "engagement_pct"], ascending=[True, False])
    st.dataframe(table, use_container_width=True, height=420)

def render_detail_tab(sheet_name, df, ctx):
    name_col = ctx["name_col"]
    country_col = ctx["country_col"]
    payment_col = ctx["payment_col"]
    event_cols = ctx["event_cols"]
    event_dates = ctx["event_dates"]

    st.subheader(f"{sheet_name} Analytics")

    total = int(df[name_col].count())
    active = int(df["is_active"].sum())
    paid = int(df["is_paid"].sum())
    avg_engagement = round(df["engagement_pct"].mean(), 1) if len(df) else 0
    paid_active = int((df["is_paid"] & df["is_active"]).sum())

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Students", f"{total:,}")
    k2.metric("Active Students", f"{active:,}", delta=f"{(active/total*100 if total else 0):.1f}% active")
    k3.metric("Paid / Admitted", f"{paid:,}", delta=f"{(paid/total*100 if total else 0):.1f}% paid")
    k4.metric("Avg Engagement %", f"{avg_engagement:.1f}%")

    left, right = st.columns(2)
    with left:
        tmp = df.copy()
        tmp["Engagement Band"] = pd.cut(
            tmp["engagement_pct"],
            bins=[-0.1, 0, 25, 50, 75, 100],
            labels=["0%", "1–25%", "26–50%", "51–75%", "76–100%"],
        )
        band = tmp.groupby("Engagement Band", observed=False).size().reset_index(name="Students")
        fig = px.bar(band, x="Engagement Band", y="Students", title="Engagement Distribution")
        fig.update_traces(marker_color=GREEN)
        st.plotly_chart(nice_layout(fig, height=360), use_container_width=True)
    with right:
        pay = df.groupby("paid_label").size().reset_index(name="Students")
        st.plotly_chart(
            donut_chart(
                pay,
                "paid_label",
                "Students",
                "Paid vs Not Paid",
                {"Paid / Admitted": GREEN, "Not Paid": GREEN_5},
            ),
            use_container_width=True,
        )

    c1, c2 = st.columns(2)
    with c1:
        if event_cols:
            event_summary = pd.DataFrame({
                "Event": event_cols,
                "Participants": [int(pd.to_numeric(df[c], errors="coerce").fillna(0).sum()) for c in event_cols],
                "Date": [event_dates.get(c, pd.NaT) for c in event_cols],
            }).sort_values(["Participants", "Event"], ascending=[False, True]).head(12)
            fig = px.bar(event_summary.sort_values("Participants"), x="Participants", y="Event", orientation="h", title="Top Events")
            fig.update_traces(marker_color=GREEN_2)
            st.plotly_chart(nice_layout(fig, height=430), use_container_width=True)
    with c2:
        if country_col and country_col in df.columns:
            top_country = (
                df.groupby(country_col)
                .agg(Students=(name_col, "count"))
                .reset_index()
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
    display_cols = [c for c in [name_col, country_col, payment_col, "engagement_pct", "engagement_score", "paid_label"] if c and c in df.columns]
    event_preview = event_cols[:8]
    display_cols += [c for c in event_preview if c in df.columns]
    st.dataframe(df[display_cols].sort_values(["engagement_pct", "engagement_score"], ascending=False), use_container_width=True, height=440)

def main():
    st.markdown(
        """
        <div class="hero-card">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:16px;flex-wrap:wrap;">
                <div>
                    <div style="font-size:34px;font-weight:900;color:#12372a;">📊 Tetr Business School Analytics Dashboard</div>
                    <div style="margin-top:6px;color:#2a6a52;font-size:16px;font-weight:600;">
                        Overview from <b>Master UG</b> + <b>Master PG</b> • Detailed analytics for <b>UG B9, UG B8, UG B7, UG B6, PG B5</b>
                    </div>
                </div>
                <div style="padding:8px 14px;border-radius:999px;background:#e8f6ed;border:1px solid #cfe8d9;color:#0b3d2e;font-weight:700;">
                    Cleaner UI · Better filters · Stronger insights
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Upload workbook")
        uploaded_file = st.file_uploader("Excel workbook", type=["xlsx"])
        st.caption("Use the workbook that contains the Master UG / Master PG and batch tabs.")
        if uploaded_file is None:
            sample_path = Path("New Master Engagement  (13).xlsx")
            if sample_path.exists():
                st.info("No file uploaded. Using local workbook if running beside the Excel file.")
        st.markdown("## Global filters")
        only_active = st.checkbox("Show only active students", value=False)
        only_paid = st.checkbox("Show only paid/admitted students", value=False)
        selected_programs = st.multiselect("Program filter", ["UG", "PG"], default=["UG", "PG"])

    file_bytes = None
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
    else:
        local_path = Path("New Master Engagement  (13).xlsx")
        if local_path.exists():
            file_bytes = local_path.read_bytes()

    if file_bytes is None:
        st.warning("Upload the Excel file to start the dashboard.")
        st.stop()

    data = load_dashboard_data(file_bytes)

    if data["missing"]:
        st.warning("Missing expected sheets: " + ", ".join(data["missing"]))

    overview_df = data["overview_df"].copy()
    if not overview_df.empty:
        if selected_programs:
            overview_df = overview_df[overview_df["Program"].isin(selected_programs)]
        if only_active:
            overview_df = overview_df[overview_df["is_active"]]
        if only_paid:
            overview_df = overview_df[overview_df["is_paid"]]

    overview_ctx = next(iter(data["master_contexts"].values())) if data["master_contexts"] else {
        "name_col": "Name", "country_col": None, "batch_col": "Batch", "income_col": None
    }

    tabs = st.tabs(["Overview"] + [s for s in DETAIL_SHEETS if s in data["details"]])

    with tabs[0]:
        render_overview(overview_df, overview_ctx)

    for i, sheet_name in enumerate([s for s in DETAIL_SHEETS if s in data["details"]], start=1):
        with tabs[i]:
            ddf = data["details"][sheet_name].copy()
            if selected_programs:
                ddf = ddf[ddf["Program"].isin(selected_programs)]
            if only_active:
                ddf = ddf[ddf["is_active"]]
            if only_paid:
                ddf = ddf[ddf["is_paid"]]
            render_detail_tab(sheet_name, ddf, data["detail_contexts"][sheet_name])

if __name__ == "__main__":
    main()
