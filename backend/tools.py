
import pandas as pd
import numpy as np

from backend import calculations as calc
from pathlib import Path
import os


# --------------------------------------------------
# 1. LOAD STUDENT DATA
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = Path(
    os.getenv("STUDENT_DATA_DIR", BASE_DIR / "data")
)

REQUIRED_FILES = {
    "students": "students.csv",
    "courses": "courses.csv",
    "assessments": "assessments.csv",
    "attendance": "attendance.csv",
    "mentors": "mentors.csv",
}


def load_data(data_dir=None):
    """Load all five student datasets."""

    folder = Path(data_dir) if data_dir else DATA_DIR

    datasets = {}

    for name, filename in REQUIRED_FILES.items():
        file_path = folder / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {file_path}"
            )

        datasets[name] = pd.read_csv(file_path)

    calc.set_data(datasets)

    return datasets


# --------------------------------------------------
# 2. STUDENT PROFILE
# --------------------------------------------------

def get_complete_student_information(student_id):

    data = calc.get_data()

    students = data["students"]
    courses = data["courses"]
    mentors = data["mentors"]

    matches = students[
        students["student_id"] == student_id
    ]

    if matches.empty:
        return {"error": "Student not found"}

    student = matches.iloc[0]

    course = courses[
        courses["course_id"] == student["course_enrolled"]
    ]

    mentor = mentors[
        mentors["mentor_id"] == student["mentor_id"]
    ]

    grade = calc.calculate_grade(student_id)
    attendance = calc.calculate_attendance(student_id)
    progress = calc.calculate_progress(student_id)

    if grade is None or progress is None:
        return {"error": "Student information unavailable"}

    alert = calc.calculate_alert(
        grade["overall_grade"],
        attendance["weighted_attendance_pct"]
    )

    return {
        "student_id": student_id,
        "student_name": student["student_name"],
        "batch": student["batch"],
        "course": (
            course.iloc[0]["course_name"]
            if not course.empty else None
        ),
        "mentor": (
            mentor.iloc[0]["mentor_name"]
            if not mentor.empty else None
        ),
        "grade": grade,
        "attendance": attendance,
        "progress": progress,
        "alert": alert,
        "parent_priority": calc.calculate_parent_priority(alert),
        "consecutive_absences":
            calc.get_max_consecutive_absences(student_id),
    }


# --------------------------------------------------
# 3. SUBMISSION DELAY
# --------------------------------------------------

def get_average_submission_delay(student_id):

    assessments = calc.get_data()["assessments"]

    records = assessments[
        assessments["student_id"] == student_id
    ].copy()

    if records.empty:
        return 0.0

    records["due_date"] = pd.to_datetime(
        records["due_date"]
    )

    records["submitted_date"] = pd.to_datetime(
        records["submitted_date"],
        errors="coerce"
    )

    records = records.dropna(
        subset=["submitted_date"]
    )

    if records.empty:
        return 0.0

    delays = (
        records["submitted_date"]
        - records["due_date"]
    ).dt.days

    return float(delays.mean())


# --------------------------------------------------
# 4. NORMALIZED GRADE
# --------------------------------------------------

def get_normalized_grade(student_id):

    grade = calc.calculate_grade(student_id)

    if grade is None:
        return None

    components = {
        "quiz": grade["quiz_avg"],
        "assignment": grade["assignment_avg"],
        "project": grade["project_avg"],
        "exam": grade["exam_avg"],
    }

    weakest = min(
        components,
        key=components.get
    )

    remaining_weight = sum(
        weight
        for name, weight in calc.ASSESSMENT_WEIGHTS.items()
        if name != weakest
    )

    normalized = sum(
        components[name] * weight
        for name, weight in calc.ASSESSMENT_WEIGHTS.items()
        if name != weakest
    ) / remaining_weight

    return {
        "weakest_component": weakest,
        "normalized_grade": normalized,
    }


# --------------------------------------------------
# 5. GROUP STUDENT METRICS
# --------------------------------------------------

def get_group_student_metrics(student_ids):

    rows = []

    for student_id in student_ids:

        grade = calc.calculate_grade(student_id)
        attendance = calc.calculate_attendance(student_id)
        progress = calc.calculate_progress(student_id)

        if grade is None or progress is None:
            continue

        alert = calc.calculate_alert(
            grade["overall_grade"],
            attendance["weighted_attendance_pct"]
        )

        rows.append({
            "student_id": student_id,
            "overall_grade": grade["overall_grade"],
            "weighted_attendance":
                attendance["weighted_attendance_pct"],
            "completion_pct": progress["completion_pct"],
            "alert": alert,
        })

    return pd.DataFrame(rows)


# --------------------------------------------------
# 6. BATCH ANALYTICS
# --------------------------------------------------

def get_batch_analytics(batch_name):

    students = calc.get_data()["students"]

    batch_students = students[
        students["batch"] == batch_name
    ]

    metrics = get_group_student_metrics(
        batch_students["student_id"].tolist()
    )

    if metrics.empty:
        return {"error": "No students found in this batch"}

    counts = metrics["alert"].value_counts()

    return {
        "student_count": len(metrics),
        "average_grade": float(
            metrics["overall_grade"].mean()
        ),
        "average_weighted_attendance": float(
            metrics["weighted_attendance"].mean()
        ),
        "red_count": int(counts.get("Red", 0)),
        "yellow_count": int(counts.get("Yellow", 0)),
        "orange_count": int(counts.get("Orange", 0)),
        "green_count": int(counts.get("Green", 0)),
    }


# --------------------------------------------------
# 7. COURSE ANALYTICS
# --------------------------------------------------

def get_course_id(course_name):

    courses = calc.get_data()["courses"]

    matches = courses[
        courses["course_name"].str.strip().str.lower()
        == course_name.strip().lower()
    ]

    if matches.empty:
        raise ValueError("Course not found")

    return matches.iloc[0]["course_id"]


def get_course_analytics(course_name):

    data = calc.get_data()

    course_id = get_course_id(course_name)

    course_students = data["students"][
        data["students"]["course_enrolled"] == course_id
    ]

    metrics = get_group_student_metrics(
        course_students["student_id"].tolist()
    )

    if metrics.empty:
        return {"error": "No course students found"}

    return {
        "course_id": course_id,
        "student_count": len(metrics),
        "average_grade": float(
            metrics["overall_grade"].mean()
        ),
        "average_weighted_attendance": float(
            metrics["weighted_attendance"].mean()
        ),
        "average_completion": float(
            metrics["completion_pct"].mean()
        ),
        "red_count": int(
            (metrics["alert"] == "Red").sum()
        ),
    }


# --------------------------------------------------
# 8. COURSE PASSING STATISTICS
# --------------------------------------------------

def get_passing_statistics(course_name):

    data = calc.get_data()

    course_id = get_course_id(course_name)

    course = data["courses"][
        data["courses"]["course_id"] == course_id
    ]

    if course.empty:
        return {"error": "Course not found"}

    passing_grade = float(
        course.iloc[0]["passing_grade"]
    )

    course_students = data["students"][
        data["students"]["course_enrolled"] == course_id
    ]

    grades = []

    for student_id in course_students["student_id"]:

        grade = calc.calculate_grade(student_id)

        if grade is not None:
            grades.append(grade["overall_grade"])

    if not grades:
        return {"error": "No grade records found"}

    passed = [
        grade for grade in grades
        if grade >= passing_grade
    ]

    failed = [
        grade for grade in grades
        if grade < passing_grade
    ]

    return {
        "total_students": len(grades),
        "passing_count": len(passed),
        "failing_count": len(failed),
        "passing_percentage":
            len(passed) / len(grades) * 100,
        "average_failing_deficit": (
            passing_grade - np.mean(failed)
            if failed else 0.0
        ),
    }


# --------------------------------------------------
# 9. ASSESSMENT COUNT
# --------------------------------------------------

def get_assessment_count(student_id, assessment_type=None):

    assessments = calc.get_data()["assessments"]

    records = assessments[
        assessments["student_id"] == student_id
    ]

    if assessment_type:
        records = records[
            records["type"].str.lower()
            == assessment_type.lower()
        ]

    return len(records)


# --------------------------------------------------
# 10. ADDITIONAL LECTURE SESSIONS
# --------------------------------------------------

def get_additional_lecture_sessions(
    student_id,
    target=75
):

    attendance = calc.get_data()["attendance"]

    records = attendance[
        (attendance["student_id"] == student_id)
        & (attendance["session_type"].str.lower() == "lecture")
    ].copy()

    if records.empty or target <= 0 or target >= 100:
        return 0

    credits = records["status"].str.lower().map({
        "present": 1.0,
        "late": 0.5,
        "absent": 0.0,
    }).fillna(0.0)

    present_credit = credits.sum()
    total_sessions = len(records)

    if present_credit / total_sessions * 100 >= target:
        return 0

    required = (
        target * total_sessions - 100 * present_credit
    ) / (100 - target)

    return max(0, int(np.ceil(required)))


# --------------------------------------------------
# 11. ORIGINAL 16 AGENT TOOLS
# --------------------------------------------------

agent_tools = {
    "student_profile":
        get_complete_student_information,

    "student_grade":
        calc.calculate_grade,

    "student_attendance":
        calc.calculate_attendance,

    "student_progress":
        calc.calculate_progress,

    "student_alert":
        lambda student_id: calc.calculate_alert(
            calc.calculate_grade(student_id)["overall_grade"],
            calc.calculate_attendance(
                student_id
            )["weighted_attendance_pct"]
        ),

    "batch_analytics":
        get_batch_analytics,

    "course_analytics":
        get_course_analytics,

    "course_passing_statistics":
        get_passing_statistics,

    "assessment_count":
        get_assessment_count,

    "average_submission_delay":
        get_average_submission_delay,

    "normalized_grade":
        get_normalized_grade,

    "consecutive_absences":
        calc.get_max_consecutive_absences,

    "weeks_behind":
        calc.get_weeks_behind,

    "minimum_exam_score":
        calc.get_minimum_exam_score,

    "minimum_assignment_score":
        calc.get_minimum_assignment_score,

    "additional_lecture_sessions":
        get_additional_lecture_sessions,
}


# --------------------------------------------------
# 12. TOOL EXECUTION
# --------------------------------------------------

def execute_agent_tool(tool_name, arguments):

    if tool_name not in agent_tools:
        return {
            "success": False,
            "error": f"Unknown tool: {tool_name}"
        }

    try:
        result = agent_tools[tool_name](**arguments)

        return {
            "success": True,
            "tool": tool_name,
            "result": result,
        }

    except Exception as error:
        return {
            "success": False,
            "tool": tool_name,
            "error": str(error),
        }


# --------------------------------------------------
# 13. GROQ TOOL SCHEMAS
# --------------------------------------------------

def _tool(name, description, properties, required):

    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


student_id = {
    "type": "string",
    "description": "Student ID",
}

groq_tools = [

    _tool(
        "student_profile",
        "Get complete student information.",
        {"student_id": student_id},
        ["student_id"]
    ),

    _tool(
        "student_grade",
        "Get student grade information.",
        {"student_id": student_id},
        ["student_id"]
    ),

    _tool(
        "student_attendance",
        "Get student attendance information.",
        {"student_id": student_id},
        ["student_id"]
    ),

    _tool(
        "student_progress",
        "Get student progress information.",
        {"student_id": student_id},
        ["student_id"]
    ),

    _tool(
        "student_alert",
        "Get student alert status.",
        {"student_id": student_id},
        ["student_id"]
    ),

    _tool(
        "batch_analytics",
        "Get batch performance analytics.",
        {"batch_name": {"type": "string"}},
        ["batch_name"]
    ),

    _tool(
        "course_analytics",
        "Get course performance analytics.",
        {"course_name": {"type": "string"}},
        ["course_name"]
    ),

    _tool(
        "course_passing_statistics",
        "Get course passing statistics.",
        {"course_name": {"type": "string"}},
        ["course_name"]
    ),

    _tool(
        "assessment_count",
        "Count student assessments.",
        {
            "student_id": student_id,
            "assessment_type": {
                "type": "string",
                "enum": [
                    "quiz",
                    "assignment",
                    "project",
                    "exam"
                ]
            }
        },
        ["student_id"]
    ),

    _tool(
        "average_submission_delay",
        "Calculate average submission delay.",
        {"student_id": student_id},
        ["student_id"]
    ),

    _tool(
        "normalized_grade",
        "Calculate normalized student grade.",
        {"student_id": student_id},
        ["student_id"]
    ),

    _tool(
        "consecutive_absences",
        "Calculate consecutive absences.",
        {"student_id": student_id},
        ["student_id"]
    ),

    _tool(
        "weeks_behind",
        "Calculate weeks behind schedule.",
        {"student_id": student_id},
        ["student_id"]
    ),

    _tool(
        "minimum_exam_score",
        "Calculate minimum required exam score.",
        {
            "student_id": student_id,
            "target_grade": {"type": "number"}
        },
        ["student_id", "target_grade"]
    ),

    _tool(
        "minimum_assignment_score",
        "Calculate minimum required assignment score.",
        {
            "student_id": student_id,
            "passing_grade": {"type": "number"}
        },
        ["student_id", "passing_grade"]
    ),

    _tool(
        "additional_lecture_sessions",
        "Calculate additional lecture sessions.",
        {
            "student_id": student_id,
            "target": {"type": "number"}
        },
        ["student_id"]
    ),
]


assert len(agent_tools) == 16
assert len(groq_tools) == 16
