# NudgeEd Prototype

**NudgeEd sees the signal before a student silently slips.**

This is a hackathon-ready Streamlit prototype based on a trajectory-first campus support concept. It does not simply check whether attendance or marks are below one threshold: it calculates recent slopes, worsening momentum, missed assignments, and grievance signals, then explains why a student was flagged.

## Prototype flows

1. **Faculty Command Center**
   - Ranked risk queue
   - Attendance + grade trajectories
   - Explainable risk reasons
   - Suggested faculty intervention
   - Aggregated cohort/topic struggle insights

2. **Student Nudge**
   - Private 7-day recovery actions
   - Gemini-powered student copilot when an API key is configured

3. **NudgeEd Lens**
   - Uses Streamlit camera input for a real phone/browser camera capture
   - With Gemini enabled, reads the captured noticeboard image and extracts only deadlines relevant to course/year
   - Has a demo extraction fallback when no API key is supplied

4. **Voice Assistant**
   - Uses Streamlit microphone input
   - With Gemini enabled, understands recorded audio and answers the student

5. **Office Kit Handoff**
   - Generates an XLSX faculty report with optional anonymization
   - This is the prototype adapter for phone-to-laptop review; a native iQOO/Office Kit integration can replace the export layer in a device build

## Run locally

```bash
python -m venv .venv
```

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, either use Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

or run Streamlit directly without activation:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Enable live AI

Set `GEMINI_API_KEY` in your environment before launching the app.

PowerShell example:

```powershell
$env:GEMINI_API_KEY="YOUR_KEY"
streamlit run app.py
```

The prototype defaults to `gemini-3.7-flash` but supports `GEMINI_MODEL` as an override.

## Demo story for judges

- Start on **Faculty Command Center** and select *Aarav Mehta*.
- Point out that the app explains both the declining attendance/grade slope and the *acceleration* of the decline.
- Switch to **Student Nudge** to show how the flag becomes a concrete 7-day action rather than a punishment.
- Open **NudgeEd Lens** on a phone and take a real picture of a printed/physical notice.
- Open **Voice Assistant** and record a question.
- Finish with **Office Kit Handoff** and export the anonymized faculty report.

## Important prototype boundary

The sample student data is synthetic. Risk scoring is intentionally transparent and deterministic for demo purposes. In a production campus deployment, thresholds/weights should be validated with academic-support teams, protected attributes should not be used to penalize students, access should be role-based, and interventions should remain human-reviewed.
