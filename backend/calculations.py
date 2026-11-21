
import pandas as pd
import numpy as np


# Competition reference date from the original notebook
REFERENCE_DATE = pd.Timestamp("2025-01-15")

ASSESSMENT_WEIGHTS = {
    "quiz": 0.20,
    "assignment": 0.30,
    "project": 0.25,
    "exam": 0.25,
}

SESSION_WEIGHTS = {
    "lecture": 1.0,
    "lab": 1.5,
    "mentoring": 1.0,
}

# Dataset storage
_data = {}


def set_data(datasets):
    """Make the loaded student datasets available to calculations."""
    global _data
    _data = datasets


def get_data():
    """Return the currently loaded datasets."""
    if not _data:
        raise RuntimeError(
            "Student datasets are not loaded. Call set_data() first."
        )
    return _data


def calculate_grade(student_id):
    data = get_data()
    students = data["students"]
    assessments = data["assessments"]

    student = students[
        students["student_id"] == student_id
    ]

    if student.empty:
        return None

    course_id = student.iloc[0]["course_enrolled"]

    records = assessments[
        (assessments["student_id"] == student_id)
        & (assessments["course_id"] == course_id)
    ]

    averages = {}

    for assessment_type in ASSESSMENT_WEIGHTS:
        component = records[
            records["type"].str.lower() == assessment_type
        ]

        if component.empty:
            averages[assessment_type] = 0.0
        else:
            total_scored = component["scored"].sum()
            total_max = component["max_score"].sum()

            averages[assessment_type] = (
                total_scored / total_max * 100
                if total_max > 0 else 0.0
            )

    overall_grade = sum(
        averages[name] * weight
        for name, weight in ASSESSMENT_WEIGHTS.items()
    )

    return {
        "quiz_avg": averages["quiz"],
        "assignment_avg": averages["assignment"],
        "project_avg": averages["project"],
        "exam_avg": averages["exam"],
        "overall_grade": overall_grade,
    }


def calculate_attendance(student_id):
    data = get_data()
    records = data["attendance"]

    records = records[
        records["student_id"] == student_id
    ].copy()

    if records.empty:
        return {
            "attendance_pct": 0.0,
            "weighted_attendance_pct": 0.0,
            "lecture_attendance": 0.0,
            "lab_attendance": 0.0,
            "mentoring_attendance": 0.0,
        }

    records["attendance_credit"] = (
        records["status"].str.lower().map({
            "present": 1.0,
            "late": 0.5,
            "absent": 0.0,
        }).fillna(0.0)
    )

    attendance_pct = (
        records["attendance_credit"].mean() * 100
    )

    records["session_weight"] = (
        records["session_type"].str.lower().map(
            SESSION_WEIGHTS
        )
    )

    records["session_weight"] = (
        records["session_weight"].fillna(0.0)
    )

    total_weight = records["session_weight"].sum()

    weighted_attendance_pct = (
        records["attendance_credit"].mul(
            records["session_weight"]
        ).sum() / total_weight * 100
        if total_weight > 0 else 0.0
    )

    type_percentages = {}

    for session_type in SESSION_WEIGHTS:
        subset = records[
            records["session_type"].str.lower() == session_type
        ]

        type_percentages[session_type] = (
            subset["attendance_credit"].mean() * 100
            if not subset.empty else 0.0
        )

    return {
        "attendance_pct": attendance_pct,
        "weighted_attendance_pct": weighted_attendance_pct,
        "lecture_attendance": type_percentages["lecture"],
        "lab_attendance": type_percentages["lab"],
        "mentoring_attendance": type_percentages["mentoring"],
    }


def calculate_progress(student_id):
    data = get_data()
    students = data["students"]
    courses = data["courses"]

    student_record = students[
        students["student_id"] == student_id
    ]

    if student_record.empty:
        return None

    student = student_record.iloc[0]

    course = courses[
        courses["course_id"] == student["course_enrolled"]
    ]

    if course.empty:
        return None

    total_modules = int(course.iloc[0]["total_modules"])
    current_module = int(student["current_module"])

    enrollment_date = pd.to_datetime(
        student["enrollment_date"]
    )

    weeks_since_enrollment = max(
        0,
        (REFERENCE_DATE - enrollment_date).days / 7
    )

    expected_module = min(
        weeks_since_enrollment / 3,
        total_modules
    )

    if current_module >= total_modules:
        progress_status = "Completed"
    elif current_module >= expected_module:
        progress_status = "On Track"
    elif current_module >= expected_module - 1:
        progress_status = "Slightly Behind"
    elif current_module >= expected_module - 3:
        progress_status = "Behind"
    else:
        progress_status = "At Risk"

    completion_pct = (
        current_module / total_modules * 100
        if total_modules > 0 else 0.0
    )

    if current_module >= total_modules:
        estimated_completion = REFERENCE_DATE
    elif current_module > 0:
        weeks_per_module = (
            weeks_since_enrollment / current_module
        )

        remaining_weeks = (
            total_modules - current_module
        ) * weeks_per_module

        estimated_completion = (
            REFERENCE_DATE
            + pd.Timedelta(days=remaining_weeks * 7)
        )
    else:
        estimated_completion = None

    return {
        "current_module": current_module,
        "total_modules": total_modules,
        "expected_module": expected_module,
        "weeks_since_enrollment": weeks_since_enrollment,
        "progress_status": progress_status,
        "completion_pct": completion_pct,
        "estimated_completion": estimated_completion,
    }


def calculate_alert(grade, weighted_attendance_pct):
    if grade < 50 or weighted_attendance_pct < 60:
        return "Red"
    elif grade < 65 or weighted_attendance_pct < 70:
        return "Yellow"
    elif grade >= 75 and weighted_attendance_pct >= 75:
        return "Green"
    else:
        return "Orange"


def calculate_parent_priority(alert_color):
    priorities = {
        "Red": "Immediate",
        "Yellow": "High",
        "Orange": "Medium",
        "Green": "Low",
    }

    return priorities.get(alert_color, "Unknown")


def get_max_consecutive_absences(student_id):
    data = get_data()
    records = data["attendance"]

    records = records[
        records["student_id"] == student_id
    ].copy()

    if records.empty:
        return 0

    records["date"] = pd.to_datetime(records["date"])
    records = records.sort_values("date")

    absent = records["status"].str.lower().eq("absent")
    groups = absent.ne(absent.shift()).cumsum()

    streaks = (
        records[absent]
        .groupby(groups[absent])
        .size()
    )

    return int(streaks.max()) if not streaks.empty else 0


def get_weeks_behind(student_id):
    progress = calculate_progress(student_id)

    if progress is None:
        return None

    if progress["current_module"] >= progress["expected_module"]:
        return 0.0

    return max(
        0.0,
        (
            progress["expected_module"]
            - progress["current_module"]
        ) * 3
    )


def get_minimum_exam_score(student_id, target_grade):
    grade = calculate_grade(student_id)

    if grade is None:
        return None

    current_without_exam = (
        grade["quiz_avg"] * 0.20
        + grade["assignment_avg"] * 0.30
        + grade["project_avg"] * 0.25
    )

    required = (
        target_grade - current_without_exam
    ) / 0.25

    return max(0.0, min(100.0, required))


def get_minimum_assignment_score(student_id, passing_grade):
    grade = calculate_grade(student_id)

    if grade is None:
        return None

    current_without_assignment = (
        grade["quiz_avg"] * 0.20
        + grade["project_avg"] * 0.25
        + grade["exam_avg"] * 0.25
    )

    required = (
        passing_grade - current_without_assignment
    ) / 0.30

    return max(0.0, min(100.0, required))


def get_grade_points_needed(student_id, target_grade):
    grade = calculate_grade(student_id)

    if grade is None:
        return None

    return max(
        0.0,
        target_grade - grade["overall_grade"]
    )


def get_letter_grade(grade):
    if grade >= 90:
        return "A+"
    elif grade >= 80:
        return "A"
    elif grade >= 70:
        return "B"
    elif grade >= 60:
        return "C"
    elif grade >= 50:
        return "D"
    else:
        return "F"


def get_alert_code(alert):
    codes = {
        "Red": 1,
        "Yellow": 2,
        "Orange": 3,
        "Green": 4,
    }

    return codes.get(alert)


def get_parent_call_absence_trigger(student_id):
    max_streak = get_max_consecutive_absences(student_id)

    return {
        "max_consecutive_absences": max_streak,
        "threshold": 3,
        "triggered": max_streak >= 3,
    }
