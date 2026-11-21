
import os
from pathlib import Path

import pandas as pd


# Default location of the student CSV files
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
    """
    Load the five student datasets.

    Returns a dictionary containing Pandas DataFrames.
    """

    folder = Path(data_dir) if data_dir else DATA_DIR

    datasets = {}

    for name, filename in REQUIRED_FILES.items():

        file_path = folder / filename

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing dataset: {file_path}"
            )

        datasets[name] = pd.read_csv(file_path)

    return datasets
