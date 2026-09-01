import io
import json
import os
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None


st.set_page_config(
    page_title="NudgeEd | Early Student Support",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg,#0b0b0d 0%,#121217 38%,#f7f7f8 38%,#f7f7f8 100%); }
      .hero {padding: 1.1rem 0 1.4rem 0; color: white;}
      .eyebrow {font-size:.78rem; letter-spacing:.16em; text-transform:uppercase; color:#ff795f; font-weight:800;}
      .hero h1 {font-size:3rem; margin:.25rem 0 .4rem 0; line-height:1.02;}
      .hero p {font-size:1.03rem; color:#d4d4d8; max-width:900px;}
      .pill {display:inline-block; border:1px solid #3f3f46; color:#e4e4e7; border-radius:999px; padding:.35rem .7rem; margin:.2rem .28rem .2rem 0; font-size:.82rem;}
      .card {background:white; border:1px solid #e7e7ea; border-radius:18px; padding:1rem 1.1rem; box-shadow:0 7px 20px rgba(0,0,0,.04);}
      .risk-high {color:#b42318; font-weight:800;}
      .risk-med {color:#b54708; font-weight:800;}
      .risk-low {color:#027a48; font-weight:800;}
      .explain {background:#fff7ed;border-left:4px solid #fb5d42;padding:.8rem 1rem;border-radius:8px;margin:.35rem 0;}
      .good {background:#ecfdf3;border-left:4px solid #12b76a;padding:.8rem 1rem;border-radius:8px;margin:.35rem 0;}
      div[data-testid="stMetric"] {background:white;border:1px solid #e7e7ea;padding:12px 14px;border-radius:16px;}
      .small-muted {color:#71717a;font-size:.84rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------
# Demo data + risk engine
# ---------------------------
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


def clamp(x, lo=0, hi=100):
    return max(lo, min(hi, x))


def calculate_student_risk(student_df):
    student_df = student_df.sort_values("week")
    last = student_df.iloc[-1]
    att_slope = slope(student_df["attendance"].tail(4))
    grade_slope = slope(student_df["grade"].tail(4))
    missed_delta = float(student_df["missed_assignments"].iloc[-1] - student_df["missed_assignments"].iloc[-3])
    grievance_delta = float(student_df["grievances"].iloc[-1] - student_df["grievances"].iloc[-3])

    # Momentum compares the newest 3-point slope with the previous 3-point slope.
    recent_att = slope(student_df["attendance"].tail(3))
    prev_att = slope(student_df["attendance"].iloc[-5:-2]) if len(student_df) >= 5 else att_slope
    acceleration = recent_att - prev_att

    attendance_pressure = max(0, (80 - float(last.attendance)) * 1.2) + max(0, -att_slope * 5.2)
    grade_pressure = max(0, (70 - float(last.grade)) * 1.0) + max(0, -grade_slope * 4.4)
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
    score = int(round(clamp(raw)))

    if score >= 45:
        level = "High"
    elif score >= 24:
        level = "Watch"
    else:
        level = "Stable"

    reasons = []
    if att_slope <= -2:
        reasons.append(f"Attendance is falling ~{abs(att_slope):.1f} points/week over the recent trajectory.")
    if grade_slope <= -2:
        reasons.append(f"Grades are trending down ~{abs(grade_slope):.1f} points/week.")
    if acceleration <= -0.8:
        reasons.append("Attendance decline is accelerating, so the trajectory is worsening faster than before.")
    if last.missed_assignments >= 2:
        reasons.append(f"{int(last.missed_assignments)} assignments are currently missed.")
    if last.grievances >= 2:
        reasons.append(f"{int(last.grievances)} recent grievance/support signals were recorded.")
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
    result = []
    for sid, sdf in weekly.groupby("student_id"):
        r = calculate_student_risk(sdf)
        latest = sdf.sort_values("week").iloc[-1]
        result.append({
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
    return pd.DataFrame(result).sort_values("risk_score", ascending=False)


weekly, pain = build_demo_data()
risks = risk_table(weekly)


def get_gemini_client():
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key or genai is None:
        return None
    try:
        return genai.Client(api_key=key)
    except Exception:
        return None


def gemini_model():
    return os.getenv("GEMINI_MODEL", "gemini-3.7-flash")


# ---------------------------
# Hero / navigation
# ---------------------------
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">NUDGEED · EARLY SIGNAL → EARLY SUPPORT</div>
      <h1>See the signal before the semester slips.</h1>
      <p>Trajectory-aware student support that explains <b>why</b> a student is at risk, nudges them on-phone, and turns cohort patterns into faculty action.</p>
      <span class="pill">Trajectory risk</span><span class="pill">Explainable flags</span><span class="pill">NudgeEd Lens</span><span class="pill">Voice interaction</span><span class="pill">Office Kit-ready reports</span>
    </div>
    """,
    unsafe_allow_html=True,
)

mode = st.radio(
    "Prototype view",
    ["Faculty Command Center", "Student Nudge", "NudgeEd Lens", "Voice Assistant", "Office Kit Handoff"],
    horizontal=True,
    label_visibility="collapsed",
)

# ---------------------------
# Faculty dashboard
# ---------------------------
if mode == "Faculty Command Center":
    high_count = int((risks.risk_level == "High").sum())
    watch_count = int((risks.risk_level == "Watch").sum())
    improving = int((risks.att_trend >= 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students monitored", len(risks))
    c2.metric("Needs action", high_count, help="High trajectory risk")
    c3.metric("Watch closely", watch_count)
    c4.metric("Stable / improving", len(risks) - high_count - watch_count)

    st.subheader("Risk queue — ranked by trajectory, not one static cutoff")
    display = risks[["name","student_id","risk_level","risk_score","attendance","grade","att_trend","grade_trend"]].copy()
    display.columns = ["Student","ID","Status","Risk","Attendance %","Grade %","Attendance Δ/wk","Grade Δ/wk"]
    st.dataframe(
        display,
        hide_index=True,
        width="stretch",
        column_config={
            "Risk": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100, format="%d"),
            "Attendance Δ/wk": st.column_config.NumberColumn(format="%.1f"),
            "Grade Δ/wk": st.column_config.NumberColumn(format="%.1f"),
        },
    )

    left, right = st.columns([1.05, .95])
    with left:
        st.subheader("Student trajectory explorer")
        selected_name = st.selectbox("Select a student", risks["name"].tolist())
        sid = risks.loc[risks.name == selected_name, "student_id"].iloc[0]
        sdf = weekly[weekly.student_id == sid].sort_values("week")
        r = calculate_student_risk(sdf)

        chart_df = sdf.set_index("week_start")[["attendance", "grade"]]
        st.line_chart(chart_df, height=280)
        m1, m2, m3 = st.columns(3)
        m1.metric("Current risk", f"{r['risk_score']}/100", r["risk_level"])
        m2.metric("Attendance trend", f"{r['attendance_slope']:.1f} pts/wk")
        m3.metric("Grade trend", f"{r['grade_slope']:.1f} pts/wk")

    with right:
        st.subheader("Why was this student flagged?")
        st.caption("Every signal is shown to the faculty member before intervention.")
        for reason in r["reasons"]:
            st.markdown(f'<div class="explain">{reason}</div>', unsafe_allow_html=True)
        if r["risk_level"] == "High":
            st.markdown('<div class="good"><b>Suggested action:</b> Reach out within 48 hours; check workload + attendance barriers; send a 7-day recovery plan.</div>', unsafe_allow_html=True)
        elif r["risk_level"] == "Watch":
            st.markdown('<div class="good"><b>Suggested action:</b> Send a low-friction nudge now and re-check the next trajectory update.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="good"><b>Suggested action:</b> No escalation. Continue passive monitoring.</div>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Cohort intelligence — fix the class, not only the student")
    cohort = pain.groupby("topic", as_index=False).agg(
        avg_struggle=("struggle_index", "mean"),
        peak_students=("students_affected", "max"),
    ).sort_values("avg_struggle", ascending=False)
    c1, c2 = st.columns([1.1, .9])
    with c1:
        st.bar_chart(cohort.set_index("topic")["avg_struggle"], height=270)
    with c2:
        top = cohort.iloc[0]
        st.markdown(
            f"""
            <div class="card">
              <div class="eyebrow">COHORT SIGNAL</div>
              <h3 style="margin:.4rem 0">{top['topic']} is the current hotspot</h3>
              <p><b>{int(top['peak_students'])} students</b> were affected at peak. Instead of waiting for individual failures, faculty can schedule a recap, targeted worksheet, or office hour now.</p>
              <div class="small-muted">Insights are aggregated for faculty view; individual explanations remain available only for legitimate intervention.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------
# Student view / Copilot
# ---------------------------
elif mode == "Student Nudge":
    st.subheader("Student-facing support — private, actionable, phone-first")
    selected_name = st.selectbox("Demo as", risks["name"].tolist(), index=0)
    sid = risks.loc[risks.name == selected_name, "student_id"].iloc[0]
    sdf = weekly[weekly.student_id == sid].sort_values("week")
    r = calculate_student_risk(sdf)

    a, b = st.columns([.65, .35])
    with a:
        st.markdown(f"### Hey {selected_name.split()[0]} — here’s your next best move")
        if r["risk_level"] == "High":
            st.warning("Your recent pattern changed quickly. This is an early support nudge, not a label or final prediction.")
        elif r["risk_level"] == "Watch":
            st.info("A few signals are slipping. A small correction this week can stop them from compounding.")
        else:
            st.success("Your recent trajectory looks steady. Keep the routine going.")

        steps = []
        if r["attendance_slope"] < -1:
            steps.append("Protect the next 3 classes — your attendance trajectory is the fastest-moving signal.")
        if r["grade_slope"] < -1:
            steps.append("Review the last two low-scoring topics for 25 minutes each, then attempt one practice set.")
        if r["missed"]:
            steps.append(f"Clear {r['missed']} missed assignment(s) by choosing the smallest one first.")
        if r["grievances"]:
            steps.append("If a timetable, faculty, finance, or personal issue is blocking progress, open a support request instead of waiting.")
        if not steps:
            steps = ["Keep your current attendance and revision routine consistent this week."]
        for i, step in enumerate(steps[:4], 1):
            st.write(f"**{i}.** {step}")

    with b:
        st.metric("Current attendance", f"{r['latest_attendance']:.0f}%")
        st.metric("Recent grade", f"{r['latest_grade']:.0f}%")
        st.metric("Missed assignments", r["missed"])

    st.divider()
    st.subheader("Ask NudgeEd Copilot")
    question = st.text_input("Ask about recovery, deadlines, study planning, or campus support", placeholder="How do I recover my attendance and DBMS score this week?")
    if st.button("Get guidance", type="primary", disabled=not bool(question.strip())):
        client = get_gemini_client()
        if client:
            context = {
                "attendance": r["latest_attendance"],
                "attendance_trend_per_week": round(r["attendance_slope"], 2),
                "grade": r["latest_grade"],
                "grade_trend_per_week": round(r["grade_slope"], 2),
                "missed_assignments": r["missed"],
                "risk_level": r["risk_level"],
            }
            prompt = f"""You are NudgeEd, a concise campus student-support assistant. Never shame or diagnose the student. Use the data only to recommend practical academic/support steps. Student context: {json.dumps(context)}. Question: {question}. Reply with a short prioritized plan for the next 7 days."""
            try:
                response = client.models.generate_content(model=gemini_model(), contents=prompt)
                st.write(response.text)
            except Exception as e:
                st.error(f"AI service error: {e}")
        else:
            st.markdown("**Prototype guidance:** Focus on the fastest-changing signal first. Protect the next three classes, clear one missed task today, and schedule two 25-minute revision blocks for the weakest topic. Re-check your trajectory after the next academic update.")
            st.caption("Add GEMINI_API_KEY to enable live personalized AI responses.")

# ---------------------------
# Lens
# ---------------------------
elif mode == "NudgeEd Lens":
    st.subheader("NudgeEd Lens — turn a physical noticeboard into relevant deadlines")
    st.write("Use the phone camera, then filter the notice for the student’s course and year.")
    f1, f2 = st.columns(2)
    with f1:
        course = st.selectbox("Course", ["CSE-AIML", "CSE", "ECE", "MBA"])
    with f2:
        year = st.selectbox("Year", [1,2,3,4], index=1)

    source = st.radio("Input", ["Take photo", "Upload image"], horizontal=True)
    image_file = None
    if source == "Take photo":
        image_file = st.camera_input("Point the phone at a noticeboard")
    else:
        image_file = st.file_uploader("Upload noticeboard photo", type=["png","jpg","jpeg"])

    if image_file:
        image = Image.open(image_file).convert("RGB")
        st.image(image, caption="Captured notice", width="stretch")
        if st.button("Extract my deadlines", type="primary"):
            client = get_gemini_client()
            if client:
                prompt = f"""Read this campus noticeboard image. Extract only items relevant to course {course}, year {year}. Return concise markdown with: title, date/deadline, required action, location/link if visible, and confidence. Ignore unrelated notices. If a deadline is unclear, say 'date unclear' rather than inventing it."""
                try:
                    response = client.models.generate_content(model=gemini_model(), contents=[prompt, image])
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Lens error: {e}")
            else:
                demo = pd.DataFrame([
                    {"Deadline": (date.today()+timedelta(days=3)).isoformat(), "Notice": "DBMS Lab Record Submission", "Action": "Submit signed record to Lab-2", "Relevant to": f"{course} Year {year}"},
                    {"Deadline": (date.today()+timedelta(days=7)).isoformat(), "Notice": "Mid-Sem Remedial Registration", "Action": "Register via department office", "Relevant to": f"{course} Year {year}"},
                ])
                st.dataframe(demo, hide_index=True, width="stretch")
                st.caption("Camera capture is real. This fallback shows the extraction UX; add GEMINI_API_KEY for live image understanding.")
    else:
        st.info("Capture or upload a noticeboard image to test the Lens flow.")

# ---------------------------
# Voice interaction
# ---------------------------
elif mode == "Voice Assistant":
    st.subheader("Voice interaction — hands-free campus support")
    st.write("Record a question using the phone microphone. With Gemini enabled, NudgeEd can interpret the audio and respond from student-support context.")
    audio = st.audio_input("Ask NudgeEd a question", sample_rate=16000)
    if audio:
        st.audio(audio)
        if st.button("Understand & answer", type="primary"):
            client = get_gemini_client()
            if client and types is not None:
                try:
                    audio_bytes = audio.getvalue()
                    audio_part = types.Part.from_bytes(data=audio_bytes, mime_type="audio/wav")
                    prompt = "Transcribe this student's question, then answer it concisely as a campus support assistant. First line: 'You asked: ...'. Then give the answer. Do not invent campus-specific facts that are not in the audio."
                    response = client.models.generate_content(model=gemini_model(), contents=[audio_part, prompt])
                    st.write(response.text)
                except Exception as e:
                    st.error(f"Voice assistant error: {e}")
            else:
                st.info("Microphone capture works in this prototype. Add GEMINI_API_KEY to enable speech understanding and AI responses.")

# ---------------------------
# Office Kit / export
# ---------------------------
elif mode == "Office Kit Handoff":
    st.subheader("Faculty report handoff — phone → spreadsheet-ready review")
    st.write("For the prototype, this generates an anonymized XLSX report that can be opened on a laptop/office suite. A native iQOO Office Kit handoff can replace this export adapter in the device build.")

    anonymize = st.toggle("Anonymize student identities", value=True)
    report = risks.copy()
    if anonymize:
        report["name"] = report["student_id"].apply(lambda x: f"Student {x[-3:]}")
        report["student_id"] = report["student_id"].apply(lambda x: f"ANON-{x[-3:]}")

    report["recommended_action"] = report["risk_level"].map({
        "High": "Faculty outreach within 48h",
        "Watch": "Send nudge + review next update",
        "Stable": "Continue passive monitoring",
    })
    st.dataframe(report, hide_index=True, width="stretch")

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        report.to_excel(writer, sheet_name="Risk Queue", index=False)
        pain.to_excel(writer, sheet_name="Cohort Insights", index=False)
        meta = pd.DataFrame({
            "Field": ["Generated", "Purpose", "Privacy"],
            "Value": [datetime.now().isoformat(timespec="seconds"), "Early-support faculty review", "Anonymized" if anonymize else "Named demo data"],
        })
        meta.to_excel(writer, sheet_name="About", index=False)
    buffer.seek(0)
    st.download_button(
        "Download faculty report (.xlsx)",
        data=buffer,
        file_name="NudgeEd_Faculty_Report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
        width="stretch",
    )

    st.caption("Prototype boundary: this demonstrates the report handoff and file format. Native Office Kit APIs/device-sharing hooks should be wired in the Android/iQOO build when the official integration surface is available.")

st.divider()
st.caption("NudgeEd prototype · Demo data only · Risk scores are early-support signals, not disciplinary decisions or final predictions.")
