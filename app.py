import base64
import io
import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
from openai import OpenAI
from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="NudgeEd | Early Signal → Early Support",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# SAFE, LIGHT THEME CSS
# ============================================================
st.markdown(
    """
    <style>
    :root {
        --navy:#0B1739;
        --navy2:#14254F;
        --coral:#FF5F45;
        --coral-soft:#FFF0EC;
        --ink:#172033;
        --muted:#65718A;
        --bg:#F6F8FC;
        --card:#FFFFFF;
        --line:#E6EAF1;
        --green:#18A36B;
        --amber:#E9A23B;
        --red:#E44E55;
        --blue:#3B72E9;
    }

    html, body, [data-testid="stAppViewContainer"], .stApp {
        background: var(--bg) !important;
        color: var(--ink) !important;
    }

    [data-testid="stHeader"] {
        background: rgba(246,248,252,.92) !important;
        border-bottom: 1px solid rgba(11,23,57,.05);
        backdrop-filter: blur(10px);
    }

    .block-container {
        max-width: 1380px;
        padding-top: 1.15rem;
        padding-bottom: 3rem;
    }

    /* Keep native Streamlit text readable. */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span,
    label,
    .stCaption {
        color: var(--ink);
    }

    h1,h2,h3,h4,h5,h6 {
        color: var(--ink) !important;
        letter-spacing: -.02em;
    }

    /* ===== Brand row ===== */
    .brand-row {
        display:flex;
        align-items:center;
        justify-content:space-between;
        gap:1rem;
        margin-bottom:.9rem;
    }
    .brand-left {display:flex;align-items:center;gap:.7rem;}
    .brand-logo {
        width:42px;height:42px;border-radius:13px;
        display:grid;place-items:center;
        background:linear-gradient(135deg,#FF765D,#FF4F36);
        color:white;font-weight:900;font-size:1.05rem;
        box-shadow:0 8px 20px rgba(255,95,69,.24);
    }
    .brand-name {font-size:1.06rem;font-weight:800;color:var(--navy);line-height:1.1;}
    .brand-sub {font-size:.72rem;color:var(--muted);margin-top:.16rem;}
    .prototype-chip {
        display:inline-flex;align-items:center;gap:.45rem;
        border:1px solid #CBEEDF;background:#F1FBF6;color:#167D54;
        padding:.43rem .72rem;border-radius:999px;font-size:.76rem;font-weight:700;
    }
    .prototype-dot {width:7px;height:7px;background:#18A36B;border-radius:50%;}

    /* ===== Hero ===== */
    .hero {
        position:relative;
        overflow:hidden;
        border-radius:28px;
        padding:2.4rem 2.55rem;
        background:
            radial-gradient(circle at 84% 10%, rgba(255,95,69,.30), transparent 30%),
            radial-gradient(circle at 70% 85%, rgba(77,119,255,.19), transparent 34%),
            linear-gradient(135deg,#0B1739 0%,#10224A 58%,#172C59 100%);
        box-shadow:0 24px 70px rgba(18,35,73,.16);
        margin-bottom:1rem;
    }
    .hero::after {
        content:"";position:absolute;right:-70px;bottom:-115px;
        width:300px;height:300px;border-radius:50%;
        border:1px solid rgba(255,255,255,.09);
        box-shadow:0 0 0 35px rgba(255,255,255,.025),0 0 0 70px rgba(255,255,255,.018);
    }
    .hero-kicker {
        color:#FFB4A3;font-size:.72rem;font-weight:800;letter-spacing:.15em;
        text-transform:uppercase;margin-bottom:.75rem;
    }
    .hero-title {
        color:#FFFFFF !important;
        font-size:clamp(2.25rem,4.2vw,4.15rem);
        line-height:1.02;font-weight:850;letter-spacing:-.055em;
        max-width:850px;margin:0 0 .85rem 0;
    }
    .hero-title .accent {color:#FF8068;}
    .hero-copy {
        color:#D5DCEE !important;max-width:900px;font-size:1.02rem;line-height:1.62;
        margin:0 0 1.25rem 0;
    }
    .hero-chips {display:flex;gap:.55rem;flex-wrap:wrap;position:relative;z-index:2;}
    .hero-chip {
        border:1px solid rgba(255,255,255,.14);background:rgba(255,255,255,.07);
        color:#F5F7FF !important;border-radius:999px;padding:.47rem .72rem;
        font-size:.78rem;font-weight:700;
    }
    .hero-chip b {color:#FF9E8B;}

    /* ===== Navigation ===== */
    div[role="radiogroup"] {
        display:flex;gap:.45rem;flex-wrap:wrap;
        background:white;border:1px solid var(--line);border-radius:17px;
        padding:.38rem;box-shadow:0 8px 24px rgba(31,45,74,.045);
        margin-bottom:1.15rem;
    }
    div[role="radiogroup"] label {
        border-radius:12px !important;padding:.56rem .82rem !important;margin:0 !important;
        border:1px solid transparent !important;
    }
    div[role="radiogroup"] label:has(input:checked) {
        background:var(--navy) !important;
        border-color:var(--navy) !important;
    }
    div[role="radiogroup"] label:has(input:checked) p,
    div[role="radiogroup"] label:has(input:checked) span {
        color:white !important;
    }
    div[role="radiogroup"] label:not(:has(input:checked)) p,
    div[role="radiogroup"] label:not(:has(input:checked)) span {
        color:#42506A !important;
    }
    div[role="radiogroup"] input {display:none !important;}
    div[role="radiogroup"] [data-baseweb="radio"] > div:first-child {display:none !important;}

    /* ===== Section heading ===== */
    .section-wrap {margin:.55rem 0 .82rem 0;}
    .section-kicker {font-size:.68rem;color:var(--coral);font-weight:850;letter-spacing:.13em;text-transform:uppercase;}
    .section-title {font-size:1.42rem;color:var(--navy);font-weight:850;letter-spacing:-.035em;margin-top:.18rem;}
    .section-copy {font-size:.84rem;color:var(--muted);margin-top:.18rem;line-height:1.45;}

    /* ===== KPI cards ===== */
    .kpi {
        position:relative;min-height:126px;background:white;border:1px solid var(--line);
        border-radius:20px;padding:1.08rem 1.08rem 1rem;
        box-shadow:0 10px 30px rgba(28,43,75,.055);overflow:hidden;
    }
    .kpi::before {content:"";position:absolute;left:0;top:0;bottom:0;width:4px;background:var(--kpi-color);}
    .kpi-icon {position:absolute;right:1rem;top:1rem;width:34px;height:34px;border-radius:11px;display:grid;place-items:center;background:var(--kpi-soft);color:var(--kpi-color);font-weight:900;}
    .kpi-label {font-size:.76rem;color:#7A869D;font-weight:700;}
    .kpi-value {font-size:2.1rem;line-height:1;color:var(--navy);font-weight:850;letter-spacing:-.05em;margin:.42rem 0 .18rem;}
    .kpi-note {font-size:.72rem;color:#8D98AC;}

    /* ===== Panels / cards ===== */
    .card {
        background:white;border:1px solid var(--line);border-radius:20px;
        padding:1.05rem 1.1rem;box-shadow:0 10px 30px rgba(28,43,75,.045);
    }
    .risk-card {text-align:center;padding:1.15rem 1rem 1rem;}
    .risk-ring {
        width:118px;height:118px;border-radius:50%;margin:.15rem auto .75rem;
        display:grid;place-items:center;
        background:conic-gradient(var(--risk-color) calc(var(--risk)*1%), #EEF1F6 0);
        position:relative;
    }
    .risk-ring::after {content:"";position:absolute;width:88px;height:88px;border-radius:50%;background:#fff;}
    .risk-number {position:relative;z-index:1;color:var(--navy);font-size:1.7rem;font-weight:850;line-height:.95;}
    .risk-den {position:relative;z-index:1;color:#8995A9;font-size:.62rem;margin-top:-.26rem;}
    .risk-name {font-size:1.05rem;font-weight:850;color:var(--navy);}
    .risk-meta {font-size:.76rem;color:var(--muted);margin-top:.15rem;}

    .signal-card {
        display:flex;gap:.72rem;align-items:flex-start;background:white;border:1px solid var(--line);
        border-radius:14px;padding:.78rem .85rem;margin:.5rem 0;
        box-shadow:0 5px 18px rgba(31,45,74,.035);
    }
    .signal-num {width:28px;height:28px;flex:0 0 28px;border-radius:9px;display:grid;place-items:center;background:var(--coral-soft);color:var(--coral);font-size:.72rem;font-weight:900;}
    .signal-text {font-size:.82rem;color:#344158;line-height:1.45;}
    .action-card {border:1px solid #CBEEDF;background:#F4FBF7;border-radius:14px;padding:.85rem .9rem;color:#285C47;font-size:.82rem;line-height:1.5;margin-top:.6rem;}
    .action-card b {color:#167D54;}

    .hotspot {
        min-height:270px;border-radius:22px;padding:1.25rem;
        background:linear-gradient(145deg,#FFF7EE,#FFFFFF 72%);border:1px solid #F1DFCA;
        box-shadow:0 10px 30px rgba(99,70,36,.05);display:flex;flex-direction:column;justify-content:space-between;
    }
    .hotspot-tag {font-size:.67rem;color:#C67D27;font-weight:850;letter-spacing:.12em;text-transform:uppercase;}
    .hotspot-num {font-size:3.1rem;font-weight:900;color:#B96B18;letter-spacing:-.06em;line-height:1;margin:.45rem 0 .15rem;}
    .hotspot-title {font-size:1.22rem;font-weight:850;color:var(--navy);line-height:1.2;}
    .hotspot-copy {font-size:.82rem;color:#6E6E75;line-height:1.55;margin-top:.55rem;}
    .privacy {font-size:.7rem;color:#8E8B86;border-top:1px solid #EFE1D2;padding-top:.75rem;margin-top:1rem;}

    /* ===== Student ===== */
    .student-hero {
        background:linear-gradient(135deg,#FFF0EC,#FFFFFF 62%);border:1px solid #F6D7CF;
        border-radius:22px;padding:1.35rem 1.45rem;margin:.2rem 0 1rem;
    }
    .student-label {font-size:.67rem;color:var(--coral);font-weight:850;letter-spacing:.13em;text-transform:uppercase;}
    .student-title {font-size:1.65rem;font-weight:850;color:var(--navy);letter-spacing:-.035em;margin:.35rem 0;}
    .student-copy {font-size:.87rem;color:#667187;line-height:1.55;max-width:800px;}
    .step-card {display:flex;gap:.75rem;align-items:flex-start;background:white;border:1px solid var(--line);border-radius:15px;padding:.8rem .88rem;margin:.5rem 0;}
    .step-num {width:28px;height:28px;flex:0 0 28px;border-radius:9px;display:grid;place-items:center;background:#EEF3FF;color:#3B72E9;font-size:.7rem;font-weight:900;}
    .step-text {font-size:.83rem;color:#354159;line-height:1.47;}
    .mini-stat {background:white;border:1px solid var(--line);border-radius:15px;padding:.88rem .95rem;margin-bottom:.55rem;}
    .mini-label {font-size:.69rem;color:#8490A5;font-weight:800;letter-spacing:.05em;}
    .mini-value {font-size:1.34rem;color:var(--navy);font-weight:850;margin-top:.12rem;}

    /* ===== Lens / voice flow ===== */
    .flow {display:grid;grid-template-columns:repeat(4,1fr);gap:.65rem;margin:.3rem 0 1rem;}
    .flow-step {background:white;border:1px solid var(--line);border-radius:15px;padding:.82rem;box-shadow:0 6px 18px rgba(31,45,74,.035);}
    .flow-num {font-size:.65rem;color:var(--coral);font-weight:900;letter-spacing:.11em;}
    .flow-title {font-size:.82rem;color:var(--navy);font-weight:800;margin-top:.22rem;}
    .flow-copy {font-size:.72rem;color:#7A869B;margin-top:.18rem;line-height:1.38;}

    /* ===== Native widgets ===== */
    .stButton > button, .stDownloadButton > button {
        border-radius:12px !important;font-weight:800 !important;min-height:2.65rem;
        border:1px solid #DDE3EC !important;background:white !important;color:var(--navy) !important;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {border-color:#FF9B87 !important;color:#CF4932 !important;}
    .stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
        background:linear-gradient(135deg,#FF7359,#FF5339) !important;color:white !important;border-color:#FF5F45 !important;
        box-shadow:0 8px 20px rgba(255,95,69,.17);
    }

    div[data-testid="stMetric"] {
        background:white;border:1px solid var(--line);border-radius:15px;padding:.82rem .9rem;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {color:var(--navy) !important;}
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {color:#748097 !important;}

    [data-testid="stDataFrame"] {border:1px solid var(--line);border-radius:16px;overflow:hidden;background:white;}
    [data-testid="stAlert"] {border-radius:14px !important;}

    /* ===== Native form controls: force TRUE light mode ===== */
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        color-scheme: light !important;
    }

    /* Streamlit Selectbox (BaseWeb) */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSelectbox"] label p {
        color:#172033 !important;
        font-weight:700 !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
        background-color:#FFFFFF !important;
        border:1px solid #DDE3EC !important;
        color:#172033 !important;
        min-height:46px !important;
        border-radius:12px !important;
        box-shadow:0 2px 8px rgba(31,45,74,.035) !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
        border-color:#FF9B87 !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] span,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] input,
    div[data-testid="stSelectbox"] div[data-baseweb="select"] [role="combobox"] {
        color:#172033 !important;
        -webkit-text-fill-color:#172033 !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] svg {
        fill:#667187 !important;
        color:#667187 !important;
    }

    /* Extra fallback for Streamlit/BaseWeb versions that ignore the selector above */
    [data-testid="stSelectbox"] [data-baseweb="select"],
    [data-testid="stSelectbox"] [data-baseweb="select"] > div,
    [data-testid="stSelectbox"] [role="combobox"] {
        background:#FFFFFF !important;
        background-color:#FFFFFF !important;
        color:#172033 !important;
        -webkit-text-fill-color:#172033 !important;
        border-color:#DDE3EC !important;
    }
    [data-testid="stSelectbox"] [role="combobox"] *,
    [data-testid="stSelectbox"] [data-baseweb="select"] * {
        color:#172033 !important;
        -webkit-text-fill-color:#172033 !important;
    }

    /* Segmented controls used for device context; intentionally theme-independent */
    [data-testid="stSegmentedControl"] {
        background:transparent !important;
    }
    [data-testid="stSegmentedControl"] button {
        background:#FFFFFF !important;
        color:#172033 !important;
        -webkit-text-fill-color:#172033 !important;
        border:1px solid #DDE3EC !important;
        border-radius:12px !important;
        min-height:44px !important;
        font-weight:750 !important;
    }
    [data-testid="stSegmentedControl"] button:hover {
        border-color:#FF9B87 !important;
        background:#FFF7F4 !important;
    }
    [data-testid="stSegmentedControl"] button[aria-pressed="true"] {
        background:#172033 !important;
        color:#FFFFFF !important;
        -webkit-text-fill-color:#FFFFFF !important;
        border-color:#172033 !important;
    }
    [data-testid="stSegmentedControl"] button[aria-pressed="true"] * {
        color:#FFFFFF !important;
        -webkit-text-fill-color:#FFFFFF !important;
    }

    /* Open dropdown menu */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] > div,
    ul[role="listbox"] {
        color-scheme:light !important;
        background:#FFFFFF !important;
        color:#172033 !important;
    }

    ul[role="listbox"] li,
    [role="option"] {
        background:#FFFFFF !important;
        color:#172033 !important;
        -webkit-text-fill-color:#172033 !important;
    }

    ul[role="listbox"] li:hover,
    [role="option"]:hover {
        background:#FFF0EC !important;
        color:#B43F2E !important;
    }

    [role="option"][aria-selected="true"] {
        background:#FFF0EC !important;
        color:#B43F2E !important;
        font-weight:700 !important;
    }

    /* Other text inputs */
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        background:#FFFFFF !important;
        border-color:#DDE3EC !important;
        color:var(--ink) !important;
        color-scheme:light !important;
    }
    input, textarea {
        color:var(--ink) !important;
        -webkit-text-fill-color:var(--ink) !important;
        background:#FFFFFF !important;
    }
    input::placeholder, textarea::placeholder {color:#9AA5B6 !important;}

    /* ===== Bulletproof light widgets (UI v5) ===== */
    [data-testid="stTextInput"] label,
    [data-testid="stTextInput"] label p,
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] label p,
    [data-testid="stCameraInput"] label,
    [data-testid="stCameraInput"] label p,
    [data-testid="stAudioInput"] label,
    [data-testid="stAudioInput"] label p {
        color:#172033 !important; -webkit-text-fill-color:#172033 !important;
    }
    [data-testid="stTextInput"] [data-baseweb="input"],
    [data-testid="stTextInput"] [data-baseweb="base-input"],
    [data-testid="stTextInput"] input {
        background:#FFFFFF !important; background-color:#FFFFFF !important;
        color:#172033 !important; -webkit-text-fill-color:#172033 !important;
        border-color:#DDE3EC !important; color-scheme:light !important;
    }
    [data-testid="stTextInput"] input::placeholder {
        color:#8A96AA !important; -webkit-text-fill-color:#8A96AA !important; opacity:1 !important;
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stCameraInput"] > div,
    [data-testid="stAudioInput"] > div {
        background:#FFFFFF !important; color:#172033 !important; color-scheme:light !important;
        border-color:#DDE3EC !important;
    }
    [data-testid="stFileUploader"] *,
    [data-testid="stCameraInput"] *,
    [data-testid="stAudioInput"] * {
        color:#172033 !important; -webkit-text-fill-color:#172033 !important;
    }
    [data-testid="stFileUploader"] button,
    [data-testid="stCameraInput"] button,
    [data-testid="stAudioInput"] button {
        background:#FFFFFF !important; color:#172033 !important;
        -webkit-text-fill-color:#172033 !important; border:1px solid #DDE3EC !important;
    }
    [data-testid="stFileUploader"] svg,
    [data-testid="stCameraInput"] svg,
    [data-testid="stAudioInput"] svg { color:#42506A !important; fill:#42506A !important; }
    [data-testid="stChatInput"], [data-testid="stChatInput"] > div {
        background:#FFFFFF !important; color:#172033 !important; color-scheme:light !important;
    }

    .footer {text-align:center;color:#8C96A8;font-size:.68rem;padding-top:.3rem;letter-spacing:.05em;}

    @media (max-width: 850px) {
        .block-container {padding-left:1rem;padding-right:1rem;padding-top:.85rem;}
        .hero {padding:1.55rem 1.25rem;border-radius:22px;}
        .hero-title {font-size:2.35rem;}
        .hero-copy {font-size:.92rem;}
        .flow {grid-template-columns:repeat(2,1fr);}
        .brand-sub {display:none;}
        div[role="radiogroup"] label {padding:.5rem .62rem !important;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA + RISK ENGINE
# ============================================================
@st.cache_data
def build_demo_data():
    students = [
        ("STU-021", "Aarav Mehta", "CSE-AIML", 2),
        ("STU-034", "Diya Sharma", "CSE-AIML", 2),
        ("STU-041", "Kabir Singh", "CSE-AIML", 2),
        ("STU-052", "Meera Nair", "CSE-AIML", 2),
        ("STU-067", "Rohan Verma", "CSE-AIML", 2),
        ("STU-078", "Sara Khan", "CSE-AIML", 2),
        ("STU-085", "Vivaan Joshi", "CSE-AIML", 2),
        ("STU-093", "Ananya Rao", "CSE-AIML", 2),
    ]
    patterns = {
        "STU-021": ([91,90,88,84,80,74], [82,80,78,73,68,61], [0,0,1,1,2,3], [0,0,0,1,1,2]),
        "STU-034": ([82,83,82,81,82,81], [74,75,73,74,75,74], [0,0,0,1,0,0], [0,0,0,0,0,0]),
        "STU-041": ([78,77,75,72,68,63], [71,69,67,66,61,55], [0,1,1,1,2,3], [0,0,1,1,2,2]),
        "STU-052": ([94,93,95,94,93,94], [88,89,87,90,91,90], [0,0,0,0,0,0], [0,0,0,0,0,0]),
        "STU-067": ([88,85,83,80,79,77], [80,79,77,75,72,71], [0,0,1,1,1,1], [0,0,0,1,1,1]),
        "STU-078": ([86,87,86,85,84,84], [77,78,78,76,77,78], [0,0,0,0,1,0], [0,0,0,0,0,0]),
        "STU-085": ([80,79,80,78,77,76], [66,67,65,65,64,65], [0,1,0,1,1,1], [0,0,0,0,1,0]),
        "STU-093": ([92,91,90,90,89,90], [84,83,85,84,86,85], [0,0,0,0,0,0], [0,0,0,0,0,0]),
    }
    rows = []
    start = date.today() - timedelta(weeks=5)
    for sid, name, course, year in students:
        att, grade, missed, grievance = patterns[sid]
        for idx in range(6):
            rows.append({
                "student_id": sid,
                "name": name,
                "course": course,
                "year": year,
                "week": idx + 1,
                "week_start": start + timedelta(weeks=idx),
                "attendance": att[idx],
                "grade": grade[idx],
                "missed_assignments": missed[idx],
                "grievances": grievance[idx],
            })
    weekly = pd.DataFrame(rows)
    pain = pd.DataFrame({
        "week": [2,2,3,3,4,4,5,5,6,6],
        "topic": ["Probability", "DBMS Joins", "Probability", "OS Scheduling", "DBMS Joins", "Linear Algebra", "OS Scheduling", "Probability", "Linear Algebra", "DBMS Joins"],
        "struggle_index": [42,35,53,41,59,37,66,63,72,68],
        "students_affected": [16,12,21,14,24,13,28,26,31,29],
    })
    return weekly, pain


def slope(values):
    y = np.array(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    if len(y) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


def calculate_student_risk(student_df):
    student_df = student_df.sort_values("week")
    last = student_df.iloc[-1]
    att_slope = slope(student_df["attendance"].tail(4))
    grade_slope = slope(student_df["grade"].tail(4))
    missed_delta = float(student_df["missed_assignments"].iloc[-1] - student_df["missed_assignments"].iloc[-3])
    grievance_delta = float(student_df["grievances"].iloc[-1] - student_df["grievances"].iloc[-3])
    recent_att = slope(student_df["attendance"].tail(3))
    prev_att = slope(student_df["attendance"].iloc[-5:-2]) if len(student_df) >= 5 else att_slope
    acceleration = recent_att - prev_att

    attendance_pressure = max(0, (80 - float(last.attendance)) * 1.2) + max(0, -att_slope * 5.2)
    grade_pressure = max(0, (70 - float(last.grade))) + max(0, -grade_slope * 4.4)
    assignment_pressure = max(0, float(last.missed_assignments) * 7 + missed_delta * 5)
    grievance_pressure = max(0, float(last.grievances) * 8 + grievance_delta * 5)
    acceleration_pressure = max(0, -acceleration * 4.2)

    raw = (
        attendance_pressure * 0.30
        + grade_pressure * 0.28
        + assignment_pressure * 0.17
        + grievance_pressure * 0.15
        + acceleration_pressure * 0.10
    )
    # Demo calibration: preserves relative ranking but makes the signal visible.
    score = int(round(max(0, min(100, raw * 2.15))))
    if score >= 55:
        level = "High"
    elif score >= 18:
        level = "Watch"
    else:
        level = "Stable"

    reasons = []
    if att_slope <= -2:
        reasons.append(f"Attendance is falling about {abs(att_slope):.1f} points per week across the recent trajectory.")
    if grade_slope <= -2:
        reasons.append(f"Grades are trending down about {abs(grade_slope):.1f} points per week.")
    if acceleration <= -0.8:
        reasons.append("The attendance decline is accelerating, so the pattern is worsening faster than before.")
    if last.missed_assignments >= 2:
        reasons.append(f"{int(last.missed_assignments)} assignments are currently missed.")
    if last.grievances >= 2:
        reasons.append(f"{int(last.grievances)} recent grievance or support signals were recorded.")
    if not reasons:
        reasons.append("No strong negative trajectory detected; current indicators are comparatively stable.")

    return {
        "risk_score": score,
        "risk_level": level,
        "attendance_slope": att_slope,
        "grade_slope": grade_slope,
        "acceleration": acceleration,
        "reasons": reasons,
        "latest_attendance": float(last.attendance),
        "latest_grade": float(last.grade),
        "missed": int(last.missed_assignments),
        "grievances": int(last.grievances),
    }


def risk_table(weekly):
    out = []
    for sid, sdf in weekly.groupby("student_id"):
        r = calculate_student_risk(sdf)
        latest = sdf.sort_values("week").iloc[-1]
        out.append({
            "student_id": sid,
            "name": latest["name"],
            "course": latest["course"],
            "year": int(latest["year"]),
            "risk_score": r["risk_score"],
            "risk_level": r["risk_level"],
            "attendance": r["latest_attendance"],
            "grade": r["latest_grade"],
            "att_trend": r["attendance_slope"],
            "grade_trend": r["grade_slope"],
        })
    return pd.DataFrame(out).sort_values("risk_score", ascending=False)


weekly, pain = build_demo_data()
risks = risk_table(weekly)


def get_xai_key():
    """Read the xAI/Grok API key from environment variables or Streamlit Secrets."""
    key = os.getenv("XAI_API_KEY", "").strip()
    if not key:
        try:
            key = str(st.secrets.get("XAI_API_KEY", "")).strip()
        except Exception:
            key = ""
    return key


def xai_model():
    """Allow model override without editing app.py."""
    model = os.getenv("XAI_MODEL", "").strip()
    if not model:
        try:
            model = str(st.secrets.get("XAI_MODEL", "")).strip()
        except Exception:
            model = ""
    return model or "grok-4.6"


@st.cache_resource
def get_xai_client():
    """xAI exposes an OpenAI-compatible Responses API."""
    key = get_xai_key()
    if not key:
        return None
    return OpenAI(
        api_key=key,
        base_url="https://api.x.ai/v1",
        timeout=120.0,
    )


def grok_text(prompt, system_prompt=None):
    """Send a text-only request to Grok and return plain text."""
    client = get_xai_client()
    if client is None:
        return None

    items = []
    if system_prompt:
        items.append({"role": "system", "content": system_prompt})
    items.append({"role": "user", "content": prompt})

    response = client.responses.create(
        model=xai_model(),
        input=items,
        store=False,
    )
    return response.output_text


def image_to_data_url(image):
    """Convert a PIL image into a base64 JPEG data URL for xAI image understanding."""
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=92)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def grok_image(image, prompt):
    """Read an image with Grok vision through the Responses API."""
    client = get_xai_client()
    if client is None:
        return None

    response = client.responses.create(
        model=xai_model(),
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": image_to_data_url(image),
                        "detail": "high",
                    },
                    {
                        "type": "input_text",
                        "text": prompt,
                    },
                ],
            }
        ],
        store=False,
    )
    return response.output_text


def xai_transcribe(audio_bytes, filename="nudgeed-question.wav", mime_type="audio/wav"):
    """Transcribe recorded audio with xAI Speech-to-Text."""
    key = get_xai_key()
    if not key:
        return None

    response = requests.post(
        "https://api.x.ai/v1/stt",
        headers={"Authorization": f"Bearer {key}"},
        files={"file": (filename, audio_bytes, mime_type)},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("text", "").strip()


def section(kicker, title, copy=""):
    st.markdown(
        f"""
        <div class="section-wrap">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            {f'<div class="section-copy">{copy}</div>' if copy else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi(label, value, note, icon, color, soft):
    return f"""
    <div class="kpi" style="--kpi-color:{color};--kpi-soft:{soft}">
        <div class="kpi-icon">{icon}</div>
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-note">{note}</div>
    </div>
    """


def risk_color(level):
    return {"High": "#E44E55", "Watch": "#E9A23B", "Stable": "#18A36B"}.get(level, "#3B72E9")


# ============================================================
# HEADER / HERO
# ============================================================
st.markdown(
    """
    <div class="brand-row">
        <div class="brand-left">
            <div class="brand-logo">N</div>
            <div><div class="brand-name">NudgeEd</div><div class="brand-sub">Campus early-support intelligence</div></div>
        </div>
        <div class="prototype-chip"><span class="prototype-dot"></span>Prototype online · UI v5 LIGHT</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <div class="hero-kicker">⚡ EARLY SIGNAL → EARLY SUPPORT</div>
        <div class="hero-title">See the signal <span class="accent">before</span><br>the semester slips.</div>
        <div class="hero-copy">NudgeEd tracks how attendance, grades, missed work and support signals are changing — then explains why a student needs attention and turns that signal into a useful next action.</div>
        <div class="hero-chips">
            <span class="hero-chip"><b>●</b> Trajectory risk</span>
            <span class="hero-chip"><b>●</b> Explainable flags</span>
            <span class="hero-chip"><b>●</b> NudgeEd Lens</span>
            <span class="hero-chip"><b>●</b> Voice interaction</span>
            <span class="hero-chip"><b>●</b> Faculty handoff</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if "mode" not in st.session_state:
    st.session_state.mode = "Faculty Command"

nav_items = ["Faculty Command", "Student Nudge", "NudgeEd Lens", "Voice Assistant", "Office Handoff"]
nav_cols = st.columns(len(nav_items), gap="small")
for nav_col, nav_item in zip(nav_cols, nav_items):
    with nav_col:
        nav_type = "primary" if st.session_state.mode == nav_item else "secondary"
        if st.button(nav_item, key=f"nav_{nav_item}", use_container_width=True, type=nav_type):
            st.session_state.mode = nav_item
            st.rerun()
mode = st.session_state.mode


# ============================================================
# FACULTY COMMAND
# ============================================================
if mode == "Faculty Command":
    high_count = int((risks.risk_level == "High").sum())
    watch_count = int((risks.risk_level == "Watch").sum())
    stable_count = int((risks.risk_level == "Stable").sum())

    section("FACULTY COMMAND CENTER", "What needs attention right now", "Priority first, then transparent reasoning before any intervention.")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(kpi("Students monitored", len(risks), "6-week rolling view", "◎", "#3B72E9", "#EEF3FF"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Needs action", high_count, "High trajectory risk", "!", "#E44E55", "#FFF0F1"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Watch closely", watch_count, "Early drift detected", "↘", "#E9A23B", "#FFF7E8"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("Stable", stable_count, "No escalation needed", "✓", "#18A36B", "#EEF9F4"), unsafe_allow_html=True)

    st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
    section("PRIORITY QUEUE", "Ranked by trajectory, not one static cutoff", "Students with fast deterioration surface before a final threshold is crossed.")

    display = risks[["name","student_id","risk_level","risk_score","attendance","grade","att_trend","grade_trend"]].copy()
    display.columns = ["Student","ID","Status","Risk","Attendance %","Grade %","Attendance Δ/wk","Grade Δ/wk"]
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        height=318,
        column_config={
            "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d"),
            "Attendance Δ/wk": st.column_config.NumberColumn(format="%.1f"),
            "Grade Δ/wk": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    st.markdown("<div style='height:.45rem'></div>", unsafe_allow_html=True)
    left, right = st.columns([1.2, .8], gap="large")

    with left:
        section("TRAJECTORY EXPLORER", "How the signal changed", "Select a student to inspect the recent direction of change.")
        if "faculty_student" not in st.session_state:
            st.session_state.faculty_student = risks["name"].iloc[0]
        student_names = risks["name"].tolist()
        chooser_cols = st.columns(4, gap="small")
        for i, name in enumerate(student_names):
            with chooser_cols[i % 4]:
                btn_type = "primary" if st.session_state.faculty_student == name else "secondary"
                if st.button(name, key=f"faculty_student_{i}", type=btn_type, use_container_width=True):
                    st.session_state.faculty_student = name
                    st.rerun()
        selected_name = st.session_state.faculty_student
        sid = risks.loc[risks.name == selected_name, "student_id"].iloc[0]
        sdf = weekly[weekly.student_id == sid].sort_values("week")
        r = calculate_student_risk(sdf)
        chart_df = sdf.set_index("week_start")[["attendance", "grade"]]
        st.line_chart(chart_df, height=290)
        m1, m2, m3 = st.columns(3)
        m1.metric("Risk", f"{r['risk_score']}/100", r["risk_level"])
        m2.metric("Attendance drift", f"{r['attendance_slope']:.1f} pts/wk")
        m3.metric("Grade drift", f"{r['grade_slope']:.1f} pts/wk")

    with right:
        section("EXPLAINABILITY", "Why this student surfaced", "Every contributing signal remains visible to faculty.")
        rc = risk_color(r["risk_level"])
        st.markdown(
            f"""
            <div class="card risk-card">
                <div class="risk-ring" style="--risk:{r['risk_score']};--risk-color:{rc}">
                    <div><div class="risk-number">{r['risk_score']}</div><div class="risk-den">RISK / 100</div></div>
                </div>
                <div class="risk-name">{selected_name}</div>
                <div class="risk-meta">{r['risk_level']} priority · {r['latest_attendance']:.0f}% attendance · {r['latest_grade']:.0f}% grade</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for idx, reason in enumerate(r["reasons"], 1):
            st.markdown(f'<div class="signal-card"><div class="signal-num">{idx}</div><div class="signal-text">{reason}</div></div>', unsafe_allow_html=True)

        if r["risk_level"] == "High":
            action = "Reach out within 48 hours, check the attendance barrier, and send a 7-day recovery plan."
        elif r["risk_level"] == "Watch":
            action = "Send a low-friction nudge now and review again after the next academic update."
        else:
            action = "No escalation. Continue passive monitoring and preserve the current routine."
        st.markdown(f'<div class="action-card"><b>Recommended next move</b><br>{action}</div>', unsafe_allow_html=True)

    st.markdown("<div style='height:.55rem'></div>", unsafe_allow_html=True)
    section("COHORT INTELLIGENCE", "Fix the class, not only the student", "Aggregated patterns reveal where many students begin struggling together.")
    cohort = pain.groupby("topic", as_index=False).agg(avg_struggle=("struggle_index", "mean"), peak_students=("students_affected", "max")).sort_values("avg_struggle", ascending=False)
    c1, c2 = st.columns([1.15, .85], gap="large")
    with c1:
        st.bar_chart(cohort.set_index("topic")["avg_struggle"], height=285)
    with c2:
        top = cohort.iloc[0]
        st.markdown(
            f"""
            <div class="hotspot">
                <div>
                    <div class="hotspot-tag">Current hotspot</div>
                    <div class="hotspot-num">{int(top['peak_students'])}</div>
                    <div class="hotspot-title">students hit friction in {top['topic']}</div>
                    <div class="hotspot-copy">Faculty can schedule a recap, focused worksheet or office hour while the pattern is still recoverable.</div>
                </div>
                <div class="privacy">Cohort insights are aggregated. Student-level reasons appear only when intervention is needed.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# STUDENT NUDGE
# ============================================================
elif mode == "Student Nudge":
    section("STUDENT EXPERIENCE", "Support that feels useful, not alarming", "The student gets a next step instead of a scary prediction label.")
    st.caption("Choose a demo student")
    if "demo_student" not in st.session_state:
        st.session_state.demo_student = risks["name"].iloc[0]
    demo_names = risks["name"].tolist()
    demo_cols = st.columns(4, gap="small")
    for i, name in enumerate(demo_names):
        with demo_cols[i % 4]:
            btn_type = "primary" if st.session_state.demo_student == name else "secondary"
            if st.button(name, key=f"demo_student_{i}", type=btn_type, use_container_width=True):
                st.session_state.demo_student = name
                st.rerun()
    selected_name = st.session_state.demo_student
    sid = risks.loc[risks.name == selected_name, "student_id"].iloc[0]
    sdf = weekly[weekly.student_id == sid].sort_values("week")
    r = calculate_student_risk(sdf)
    first = selected_name.split()[0]

    if r["risk_level"] == "High":
        headline = "Your pattern changed quickly — here’s the smallest useful next step."
        copy = "This is not a final prediction. NudgeEd noticed a recent change and is helping you recover before it becomes harder to fix."
    elif r["risk_level"] == "Watch":
        headline = "A small drift showed up. This week is a good time to correct it."
        copy = "Nothing is being labelled as failure. The goal is to make the next action obvious while the gap is still small."
    else:
        headline = "You’re on a stable path. Keep the momentum simple."
        copy = "NudgeEd keeps watching quietly and only surfaces support when the direction meaningfully changes."

    st.markdown(
        f"""
        <div class="student-hero">
            <div class="student-label">Hi {first}</div>
            <div class="student-title">{headline}</div>
            <div class="student-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.25,.75], gap="large")
    with left:
        section("YOUR 7-DAY NUDGE", "A short recovery plan")
        steps = []
        if r["attendance_slope"] < -1:
            steps.append("Protect the next three classes first — the attendance trend matters more than one isolated percentage.")
        if r["grade_slope"] < -1:
            steps.append("Spend two focused 25-minute sessions on the weakest recent topic before adding new material.")
        if r["missed"]:
            steps.append(f"Clear {r['missed']} missed assignment(s), starting with the smallest task today.")
        if r["grievances"]:
            steps.append("If a timetable, faculty, finance or personal issue is blocking progress, open a support request instead of waiting.")
        if not steps:
            steps = ["Keep your current attendance and revision routine consistent this week."]
        for i, step in enumerate(steps[:4], 1):
            st.markdown(f'<div class="step-card"><div class="step-num">0{i}</div><div class="step-text">{step}</div></div>', unsafe_allow_html=True)

    with right:
        section("THIS WEEK", "Signal snapshot")
        st.markdown(f'<div class="mini-stat"><div class="mini-label">CURRENT ATTENDANCE</div><div class="mini-value">{r["latest_attendance"]:.0f}%</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mini-stat"><div class="mini-label">RECENT GRADE</div><div class="mini-value">{r["latest_grade"]:.0f}%</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="mini-stat"><div class="mini-label">MISSED ASSIGNMENTS</div><div class="mini-value">{r["missed"]}</div></div>', unsafe_allow_html=True)

    section("NUDGEED COPILOT", "Ask for a practical next step", "Recovery, deadlines, study planning or campus support.")
    question = st.text_input("Ask NudgeEd", placeholder="How do I recover my attendance and DBMS score this week?", label_visibility="collapsed")
    if st.button("Get my 7-day plan", type="primary", disabled=not bool(question.strip())):
        if get_xai_client():
            context = {
                "attendance": r["latest_attendance"],
                "attendance_trend_per_week": round(r["attendance_slope"], 2),
                "grade": r["latest_grade"],
                "grade_trend_per_week": round(r["grade_slope"], 2),
                "missed_assignments": r["missed"],
                "risk_level": r["risk_level"],
            }
            prompt = (
                f"Student context: {json.dumps(context)}\n"
                f"Student question: {question}\n\n"
                "Create a short, prioritized 7-day recovery plan. Keep it practical, "
                "specific and supportive. Do not shame, diagnose, or present the risk score "
                "as a final prediction."
            )
            try:
                answer = grok_text(
                    prompt,
                    system_prompt=(
                        "You are NudgeEd, a concise campus student-support assistant. "
                        "Focus on attendance recovery, study planning, deadlines and appropriate "
                        "campus support. Never invent college-specific rules or facts."
                    ),
                )
                st.markdown(answer or "No response returned by Grok.")
            except Exception as e:
                st.error(f"Grok service error: {e}")
        else:
            st.info("Prototype guidance: protect the next three classes, clear one missed task today, and schedule two 25-minute revision blocks. Add XAI_API_KEY for live Grok guidance.")


# ============================================================
# LENS
# ============================================================
elif mode == "NudgeEd Lens":
    section("NUDGEED LENS", "Turn a physical noticeboard into your deadlines", "Use the phone camera — NudgeEd filters the notice for the student’s course and year.")
    st.markdown(
        """
        <div class="flow">
            <div class="flow-step"><div class="flow-num">01 · CAPTURE</div><div class="flow-title">Open camera</div><div class="flow-copy">Use the phone’s real camera.</div></div>
            <div class="flow-step"><div class="flow-num">02 · READ</div><div class="flow-title">Understand notice</div><div class="flow-copy">Extract dates and actions.</div></div>
            <div class="flow-step"><div class="flow-num">03 · FILTER</div><div class="flow-title">Match context</div><div class="flow-copy">Course + year relevance.</div></div>
            <div class="flow-step"><div class="flow-num">04 · NUDGE</div><div class="flow-title">Surface deadline</div><div class="flow-copy">Show only what matters.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if "lens_course_value" not in st.session_state:
        st.session_state.lens_course_value = "CSE-AIML"
    if "lens_year_value" not in st.session_state:
        st.session_state.lens_year_value = 2
    if "lens_source" not in st.session_state:
        st.session_state.lens_source = "Take photo"

    f1, f2 = st.columns(2, gap="large")
    with f1:
        st.markdown("**Course**")
        cc = st.columns(4, gap="small")
        for i, value in enumerate(["CSE-AIML", "CSE", "ECE", "MBA"]):
            with cc[i]:
                t = "primary" if st.session_state.lens_course_value == value else "secondary"
                if st.button(value, key=f"course_btn_{value}", type=t, use_container_width=True):
                    st.session_state.lens_course_value = value
                    st.rerun()
    with f2:
        st.markdown("**Year**")
        yc = st.columns(4, gap="small")
        for i, value in enumerate([1, 2, 3, 4]):
            with yc[i]:
                t = "primary" if st.session_state.lens_year_value == value else "secondary"
                if st.button(str(value), key=f"year_btn_{value}", type=t, use_container_width=True):
                    st.session_state.lens_year_value = value
                    st.rerun()

    course = st.session_state.lens_course_value
    year = st.session_state.lens_year_value

    st.markdown("**Input source**")
    src_cols = st.columns(2, gap="small")
    for i, value in enumerate(["Take photo", "Upload image"]):
        with src_cols[i]:
            t = "primary" if st.session_state.lens_source == value else "secondary"
            if st.button(value, key=f"source_btn_{i}", type=t, use_container_width=True):
                st.session_state.lens_source = value
                st.rerun()
    source = st.session_state.lens_source
    image_file = st.camera_input("Point the phone at a noticeboard") if source == "Take photo" else st.file_uploader("Upload noticeboard photo", type=["png","jpg","jpeg"])
    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Captured notice", use_container_width=True)
        if st.button("Extract relevant deadlines", type="primary"):
            if get_xai_client():
                prompt = (
                    f"Read this campus noticeboard image. The student is in course {course}, year {year}. "
                    "Extract ONLY notices that are clearly relevant to that course/year. "
                    "Return concise markdown. For each relevant item include: title, date/deadline, "
                    "required action, and location/link if visibly present. If text is unclear, say "
                    "'unclear in image' instead of inventing information."
                )
                try:
                    answer = grok_image(image, prompt)
                    st.markdown(answer or "No relevant notice text was returned.")
                except Exception as e:
                    st.error(f"Lens error: {e}")
            else:
                demo = pd.DataFrame([
                    {"Deadline": (date.today()+timedelta(days=3)).isoformat(), "Notice": "DBMS Lab Record Submission", "Action": "Submit signed record to Lab-2", "Relevant to": f"{course} Year {year}"},
                    {"Deadline": (date.today()+timedelta(days=7)).isoformat(), "Notice": "Mid-Sem Remedial Registration", "Action": "Register via department office", "Relevant to": f"{course} Year {year}"},
                ])
                st.dataframe(demo, hide_index=True, use_container_width=True)
                st.caption("Camera capture is live. Add XAI_API_KEY for real Grok notice understanding.")
    else:
        st.info("Capture or upload a noticeboard image to test the Lens flow.")


# ============================================================
# VOICE
# ============================================================
elif mode == "Voice Assistant":
    section("VOICE SUPPORT", "Ask without typing", "Use the phone microphone for quick, hands-free campus support.")
    st.markdown(
        """
        <div class="flow">
            <div class="flow-step"><div class="flow-num">01 · SPEAK</div><div class="flow-title">Use phone mic</div><div class="flow-copy">Record a natural question.</div></div>
            <div class="flow-step"><div class="flow-num">02 · UNDERSTAND</div><div class="flow-title">Interpret intent</div><div class="flow-copy">Transcribe and understand.</div></div>
            <div class="flow-step"><div class="flow-num">03 · CONTEXT</div><div class="flow-title">Use support signal</div><div class="flow-copy">Connect with student context.</div></div>
            <div class="flow-step"><div class="flow-num">04 · RESPOND</div><div class="flow-title">Return next step</div><div class="flow-copy">Short, useful guidance.</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    audio = st.audio_input("Ask NudgeEd a question", sample_rate=16000)
    if audio:
        st.audio(audio)
        if st.button("Understand & answer", type="primary"):
            if get_xai_key():
                try:
                    transcript = xai_transcribe(
                        audio.getvalue(),
                        filename="nudgeed-question.wav",
                        mime_type="audio/wav",
                    )
                    if not transcript:
                        st.warning("I couldn't detect speech in that recording. Try again a little closer to the microphone.")
                    else:
                        st.markdown(f"**You asked:** {transcript}")
                        answer = grok_text(
                            transcript,
                            system_prompt=(
                                "You are NudgeEd, a concise campus support assistant. "
                                "Answer the student's spoken question with practical next steps. "
                                "Do not invent college-specific dates, policies, people or locations. "
                                "If such information is required, say the student should verify it with the college."
                            ),
                        )
                        st.markdown(answer or "No response returned by Grok.")
                except requests.HTTPError as e:
                    detail = ""
                    try:
                        detail = e.response.text
                    except Exception:
                        pass
                    st.error(f"xAI speech-to-text error: {detail or e}")
                except Exception as e:
                    st.error(f"Voice assistant error: {e}")
            else:
                st.info("Microphone capture works. Add XAI_API_KEY to enable xAI speech-to-text and Grok responses.")


# ============================================================
# OFFICE HANDOFF
# ============================================================
elif mode == "Office Handoff":
    section("FACULTY HANDOFF", "From phone signal to spreadsheet-ready review", "Generate a compact report while keeping an anonymized cohort-review mode.")
    if "anonymize_report" not in st.session_state:
        st.session_state.anonymize_report = True
    st.markdown("**Report privacy**")
    pc1, pc2 = st.columns(2, gap="small")
    with pc1:
        if st.button("Anonymized", key="privacy_anon", type="primary" if st.session_state.anonymize_report else "secondary", use_container_width=True):
            st.session_state.anonymize_report = True
            st.rerun()
    with pc2:
        if st.button("Named demo data", key="privacy_named", type="secondary" if st.session_state.anonymize_report else "primary", use_container_width=True):
            st.session_state.anonymize_report = False
            st.rerun()
    anonymize = st.session_state.anonymize_report
    report = risks.copy()
    if anonymize:
        report["name"] = report["student_id"].apply(lambda x: f"Student {x[-3:]}")
        report["student_id"] = report["student_id"].apply(lambda x: f"ANON-{x[-3:]}")
    report["recommended_action"] = report["risk_level"].map({
        "High": "Faculty outreach within 48h",
        "Watch": "Send nudge + review next update",
        "Stable": "Continue passive monitoring",
    })
    st.dataframe(report, hide_index=True, use_container_width=True)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="Risk Queue", index=False)
        pain.to_excel(writer, sheet_name="Cohort Insights", index=False)
        pd.DataFrame({
            "Field": ["Generated", "Purpose", "Privacy"],
            "Value": [datetime.now().isoformat(timespec="seconds"), "Early-support faculty review", "Anonymized" if anonymize else "Named demo data"],
        }).to_excel(writer, sheet_name="About", index=False)
    buffer.seek(0)
    st.download_button(
        "Download faculty report (.xlsx)",
        data=buffer,
        file_name="NudgeEd_Faculty_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        use_container_width=True,
    )
    st.caption("Prototype boundary: the XLSX demonstrates the handoff workflow. Native iQOO Office Kit APIs can replace this adapter in an Android/device build.")


st.divider()
st.markdown('<div class="footer">NUDGEED · DEMO DATA ONLY · EARLY-SUPPORT SIGNALS ARE NOT DISCIPLINARY OR FINAL PREDICTIONS</div>', unsafe_allow_html=True)
