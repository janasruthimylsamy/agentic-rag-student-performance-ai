import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import re

st.set_page_config(page_title="Student Performance AI", page_icon="🎓", layout="wide")
ROOT = Path(__file__).parent
POLICY_FILES = ["grading_policy.md", "attendance_policy.md", "alert_rules.md", "progress_milestones.md", "faq.md"]

# Demo records are synthetic. If the five private CSVs are supplied in ./data, the app loads them instead.
def load_students():
    folder = ROOT / "data"
    needed = ["students.csv", "courses.csv", "assessments.csv", "attendance.csv", "mentors.csv"]
    if all((folder / f).exists() for f in needed):
        students = pd.read_csv(folder / "students.csv")
        assessments = pd.read_csv(folder / "assessments.csv")
        attendance = pd.read_csv(folder / "attendance.csv")
        # Use available student identifiers and derive safe summary metrics where possible.
        sid = "student_id" if "student_id" in students.columns else students.columns[0]
        rows = []
        for _, s in students.iterrows():
            key = s[sid]
            a = assessments[assessments["student_id"] == key] if "student_id" in assessments else pd.DataFrame()
            at = attendance[attendance["student_id"] == key] if "student_id" in attendance else pd.DataFrame()
            grade_col = next((c for c in ["score", "marks", "grade", "percentage"] if c in a.columns), None)
            grade = float(pd.to_numeric(a[grade_col], errors="coerce").mean()) if grade_col and not a.empty else 0
            present_col = next((c for c in ["status", "attendance_status"] if c in at.columns), None)
            if present_col:
                vals = at[present_col].astype(str).str.lower()
                att = float(vals.isin(["present", "p", "1", "true"]).mean() * 100) if len(vals) else 0
            else:
                att = 0
            rows.append({"Student ID": str(key), "Student": str(s.get("student_name", key)), "Course": str(s.get("course_enrolled", "General")), "Batch": str(s.get("batch", "N/A")), "Grade": round(grade, 1), "Attendance": round(att, 1)})
        return pd.DataFrame(rows), False
    rng = np.random.default_rng(17)
    names = ["Aarav Sharma", "Diya Patel", "Ishaan Kumar", "Meera Nair", "Arjun Reddy", "Ananya Das", "Kabir Singh", "Sana Khan", "Rohan Iyer", "Aadhya Rao", "Vivaan Shah", "Ira Menon", "Reyansh Gupta", "Kiara Joshi", "Advik Bose", "Myra Thomas", "Neil Verma", "Tara Pillai", "Dhruv Jain", "Navya Roy"]
    grades = rng.integers(48, 97, len(names)).astype(float)
    attendance = rng.integers(55, 100, len(names)).astype(float)
    courses = ["Python Programming", "Data Structures", "Database Systems", "Web Development"]
    df = pd.DataFrame({"Student ID": [f"STU{1001+i}" for i in range(len(names))], "Student": names, "Course": [courses[i % 4] for i in range(len(names))], "Batch": ["2026-A" if i % 2 == 0 else "2026-B" for i in range(len(names))], "Grade": grades, "Attendance": attendance})
    return df, True

students, demo_mode = load_students()
students["Alert"] = np.select([students["Grade"] < 50, students["Attendance"] < 75, students["Grade"] < 65], ["Critical", "Attendance Risk", "Needs Support"], default="On Track")

# Small local retrieval over the supplied policy documents; no external API key is needed for the demo.
def retrieve_policy(question):
    terms = set(re.findall(r"[a-z0-9]+", question.lower()))
    chunks = []
    for filename in POLICY_FILES:
        path = ROOT / filename
        if path.exists():
            text = path.read_text(encoding="utf-8", errors="ignore")
            for paragraph in re.split(r"\n\s*\n", text):
                words = set(re.findall(r"[a-z0-9]+", paragraph.lower()))
                score = len(terms & words)
                if score:
                    chunks.append((score, filename, paragraph.strip()))
    return sorted(chunks, reverse=True)[:3]

st.markdown("""
<style>
.stApp {background:#faf8fc;color:#241b2b}
.block-container {padding-top:2rem;max-width:1200px}
[data-testid="stSidebar"] {background:#24152f}
[data-testid="stSidebar"] * {color:#fff !important}
.hero {padding:1.35rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#382044,#704b7e);color:white;margin-bottom:1.2rem}
.hero h1 {color:white;margin:0;font-size:2rem}.hero p {color:#f0e7f4;margin:.4rem 0 0}
div[data-testid="stMetric"] {background:white;border:1px solid #eee5f1;padding:14px;border-radius:14px}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("## 🎓 Student Performance AI")
page = st.sidebar.radio("Navigation", ["Dashboard", "Student Analytics", "Risk Alerts", "AI Assistant", "Policy Library"])
st.sidebar.caption("Agentic RAG • Student insights • Early intervention")

st.markdown('<div class="hero"><h1>Student Performance AI</h1><p>Turn academic data into clear insights and timely student support.</p></div>', unsafe_allow_html=True)
if demo_mode:
    st.info("Demo mode: showing synthetic sample students because the private Kaggle CSV files are not connected to this deployment.")

if page == "Dashboard":
    st.subheader("Academic overview")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Students", f"{len(students):,}")
    c2.metric("Average grade", f"{students['Grade'].mean():.1f}%")
    c3.metric("Average attendance", f"{students['Attendance'].mean():.1f}%")
    c4.metric("Students at risk", int(students["Alert"].ne("On Track").sum()))
    left,right = st.columns(2)
    with left:
        st.markdown("#### Grade distribution")
        bins = pd.cut(students["Grade"], bins=[0,50,65,80,100], labels=["Below 50","50–64","65–79","80+"])
        st.bar_chart(bins.value_counts().sort_index())
    with right:
        st.markdown("#### Attendance vs grade")
        st.scatter_chart(students, x="Attendance", y="Grade", color="Alert")
    st.markdown("#### Student snapshot")
    st.dataframe(students[["Student ID","Student","Course","Grade","Attendance","Alert"]], use_container_width=True, hide_index=True)

elif page == "Student Analytics":
    st.subheader("Individual student analytics")
    selected = st.selectbox("Choose a student", students["Student"].tolist())
    row = students[students["Student"] == selected].iloc[0]
    a,b,c = st.columns(3)
    a.metric("Overall grade", f"{row['Grade']:.1f}%")
    b.metric("Attendance", f"{row['Attendance']:.1f}%")
    c.metric("Status", row["Alert"])
    st.write(f"**Student ID:** {row['Student ID']}  ·  **Course:** {row['Course']}  ·  **Batch:** {row['Batch']}")
    if row["Alert"] == "On Track": st.success("The student is currently meeting the demo performance indicators.")
    else: st.warning("Suggested action: review recent assessments, check attendance barriers, and arrange a mentor check-in.")

elif page == "Risk Alerts":
    st.subheader("Early-warning student alerts")
    risky = students[students["Alert"] != "On Track"].sort_values(["Grade","Attendance"])
    st.metric("Students requiring attention", len(risky))
    st.dataframe(risky[["Student ID","Student","Course","Grade","Attendance","Alert"]], use_container_width=True, hide_index=True)
    st.caption("Demo alert logic: grade below 50 = Critical; attendance below 75% = Attendance Risk; grade below 65 = Needs Support.")

elif page == "AI Assistant":
    st.subheader("Ask the student support assistant")
    st.caption("Answers are grounded in the policy documents included in this repository. This demo uses local retrieval and does not call an external LLM.")
    prompt = st.text_area("Your question", placeholder="Example: What should I do if a student's attendance is low?")
    if st.button("Generate answer", type="primary"):
        if not prompt.strip(): st.warning("Enter a question first.")
        else:
            matches = retrieve_policy(prompt)
            st.markdown("#### Assistant response")
            if matches:
                st.write("I found these relevant policy sections. Use them to guide the next student-support action:")
                for score, source, excerpt in matches:
                    with st.expander(f"{source} · relevance {score}", expanded=True): st.write(excerpt[:1800])
            else:
                st.write("I couldn't find a matching section in the available policy documents. Please check the policy library or ask a mentor for clarification.")
            st.caption("Retrieval-based demo response — not a substitute for an authorized academic decision.")

elif page == "Policy Library":
    st.subheader("Policy knowledge base")
    found = False
    for filename in POLICY_FILES:
        path = ROOT / filename
        if path.exists():
            found = True
            with st.expander(filename): st.markdown(path.read_text(encoding="utf-8", errors="ignore"))
    if not found: st.warning("No policy Markdown files found in the repository root.")

st.divider()
st.caption("Student Performance AI · Academic insights and early intervention · Demo build")
