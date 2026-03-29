"""
PatientDataTransformer — sklearn-compatible transformer that cleans and
feature-engineers the mental health patient dataset.

Usage
-----
from data_cleaning import PatientDataTransformer
import pandas as pd

df_raw = pd.read_csv("data/data.csv")
transformer = PatientDataTransformer()
df_clean = transformer.fit_transform(df_raw)
"""

import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

warnings.filterwarnings("ignore")

# ── Lookup maps (domain knowledge — not learned from data) ────────────────────

SLEEP_MAP = {
    "Less than 5 hours": 1,
    "5-6 hours": 2,
    "7-8 hours": 3,
    "More than 8 hours": 4,
}

EDUCATION_MAP = {
    "Class 12": 0,
    # Bachelor's
    "BA": 1,
    "B.Com": 1,
    "B.Ed": 1,
    "B.Pharm": 1,
    "B.Tech": 1,
    "B.Arch": 1,
    "BBA": 1,
    "BCA": 1,
    "BE": 1,
    "BHM": 1,
    "BSc": 1,
    "LLB": 1,
    "MBBS": 1,
    # Master's
    "MA": 2,
    "M.Com": 2,
    "M.Ed": 2,
    "M.Pharm": 2,
    "M.Tech": 2,
    "MBA": 2,
    "MCA": 2,
    "ME": 2,
    "MHM": 2,
    "MSc": 2,
    "LLM": 2,
    # PhD
    "PhD": 3,
    "MD": 3,
}

PROFESSION_MAP = {
    "Software Engineer": "Technology",
    "Data Scientist": "Technology",
    "UX/UI Designer": "Technology",
    "Graphic Designer": "Technology",
    "Digital Marketer": "Technology",
    "Accountant": "Finance",
    "Financial Analyst": "Finance",
    "Investment Banker": "Finance",
    "Business Analyst": "Finance",
    "Research Analyst": "Finance",
    "Doctor": "Healthcare",
    "Pharmacist": "Healthcare",
    "Chemist": "Healthcare",
    "Lawyer": "Legal",
    "Judge": "Legal",
    "Civil Engineer": "Engineering",
    "Mechanical Engineer": "Engineering",
    "Architect": "Engineering",
    "Electrician": "Engineering",
    "Plumber": "Engineering",
    "Teacher": "Education",
    "Educational Consultant": "Education",
    "Researcher": "Education",
    "Manager": "Management",
    "HR Manager": "Management",
    "Marketing Manager": "Management",
    "Entrepreneur": "Management",
    "Consultant": "Management",
    "Travel Consultant": "Management",
    "Content Writer": "Creative & Services",
    "Chef": "Creative & Services",
    "Customer Support": "Creative & Services",
    "Sales Executive": "Creative & Services",
    "Pilot": "Creative & Services",
}

COLUMN_RENAME = {
    "working_professional_or_student": "status",
    "have_you_ever_had_suicidal_thoughts_": "suicidal_thoughts",
    "family_history_of_mental_illness": "family_history",
}

NUMERIC_COLS = [
    "age",
    "academic_pressure",
    "work_pressure",
    "cgpa",
    "study_satisfaction",
    "job_satisfaction",
    "workstudy_hours",
    "financial_stress",
]

ESSENTIAL_COLS = [
    "age",
    "pressure",
    "satisfaction",
    "sleep_duration",
    "workstudy_hours",
    "financial_stress",
    "depression",
]

# ══════════════════════════════════════════════════════════════════════════════


class PatientDataTransformer(BaseEstimator, TransformerMixin):
    """
    Cleans and feature-engineers the patient mental health dataset.

    fit()          — learns imputation statistics and thresholds from training data
    transform()    — applies the full cleaning + engineering pipeline
    fit_transform() — inherited from TransformerMixin

    Fitted attributes (available after fit)
    ----------------------------------------
    pressure_medians_       : dict {status → median pressure}
    satisfaction_medians_   : dict {status → median satisfaction}
    cgpa_median_            : float, median CGPA for students
    workstudy_median_       : float
    financial_stress_median_: float
    sleep_median_           : float
    education_median_       : float, fallback for unmapped degree strings
    max_hours_              : float, used to normalise worklife_balance
    stress_75_              : float, 75th-pct threshold for high_risk flag
    edu_unmapped_           : Series, degree values not in EDUCATION_MAP
    prof_unmapped_          : Series, profession values not in PROFESSION_MAP
    city_coords_            : dict {city → (lat, lon)}, geocoded during fit
    profession_categories_  : list of profession_category values seen in fit
    """

    def fit(self, X, y=None):
        df = self._prepare(X.copy())

        # ── Geocode cities (done once on unique values) ───────────────────────
        self.city_coords_ = self._fit_geocoding(df)

        # ── Imputation statistics ─────────────────────────────────────────────
        self.pressure_medians_ = df.groupby("status")["pressure"].median().to_dict()
        self.satisfaction_medians_ = (
            df.groupby("status")["satisfaction"].median().to_dict()
        )
        self.cgpa_median_ = df.loc[df["status"] == "Student", "cgpa"].median()
        self.workstudy_median_ = df["workstudy_hours"].median()
        self.financial_stress_median_ = df["financial_stress"].median()
        self.sleep_median_ = df["sleep_duration"].map(SLEEP_MAP).median()

        # ── Education level fallback ──────────────────────────────────────────
        edu = df["degree"].map(EDUCATION_MAP)
        self.education_median_ = edu.median()

        # ── Mapping audit (stored for reporting, not used in transform) ───────
        self.edu_unmapped_ = df.loc[edu.isna(), "degree"].value_counts()
        prof_mapped = df["profession"].map(PROFESSION_MAP)
        self.prof_unmapped_ = df.loc[
            prof_mapped.isna() & (df["status"] != "Student"), "profession"
        ].value_counts()

        # ── Threshold statistics (need imputed + encoded data) ────────────────
        df = self._impute(df)
        df = self._encode(df)
        self.profession_categories_ = sorted(
            df["profession_category"].dropna().unique().tolist()
        )
        self.max_hours_ = df["workstudy_hours"].max()
        stress = (
            df["pressure"] + df["financial_stress"] + (5 - df["sleep_duration"])
        ) / 3
        self.stress_75_ = stress.quantile(0.75)

        return self

    def transform(self, X, y=None):
        check_is_fitted(self)
        df = self._prepare(X.copy())
        df = self._impute(df)
        df = self._encode(df)
        df = self._apply_geocoding(df)
        df = self._engineer(df)
        df = self._finalize(df)
        return df

    # ── Private pipeline steps ────────────────────────────────────────────────

    def _prepare(self, df):
        """Rename columns, standardise strings, merge split student/professional cols."""
        # Normalise column names
        df.columns = (
            df.columns.str.strip()
            .str.replace(r"[?/]", "", regex=True)
            .str.replace(r"\s+", "_", regex=True)
            .str.lower()
        )
        df.rename(columns=COLUMN_RENAME, inplace=True)

        # Drop identifier
        if "name" in df.columns:
            df.drop(columns=["name"], inplace=True)

        # Strip string whitespace
        str_cols = df.select_dtypes(str).columns
        df[str_cols] = df[str_cols].apply(lambda s: s.str.strip())

        # Standardise free-text categoricals
        for col in [
            "gender",
            "status",
            "dietary_habits",
            "suicidal_thoughts",
            "family_history",
            "depression",
        ]:
            if col in df.columns:
                df[col] = df[col].str.title()

        # Fix known typos
        if "profession" in df.columns:
            df["profession"] = df["profession"].replace(
                "Finanancial Analyst", "Financial Analyst"
            )

        # Enforce numeric types
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Merge split student / professional columns into unified ones
        if "academic_pressure" in df.columns and "work_pressure" in df.columns:
            df["pressure"] = df["academic_pressure"].fillna(df["work_pressure"])
        if "study_satisfaction" in df.columns and "job_satisfaction" in df.columns:
            df["satisfaction"] = df["study_satisfaction"].fillna(df["job_satisfaction"])

        df["pressure_source"] = np.where(df["status"] == "Student", "academic", "work")

        return df

    def _impute(self, df):
        """Fill NaNs using statistics learned during fit."""
        # Group-specific medians for pressure and satisfaction
        for col, medians in [
            ("pressure", self.pressure_medians_),
            ("satisfaction", self.satisfaction_medians_),
        ]:
            df[col] = df.apply(
                lambda row: (
                    medians.get(row["status"], np.nan)
                    if pd.isna(row[col])
                    else row[col]
                ),
                axis=1,
            )

        # CGPA: meaningful for students only
        df["cgpa"] = np.where(
            df["status"] == "Student",
            df["cgpa"].fillna(self.cgpa_median_),
            np.nan,
        )

        df["workstudy_hours"] = df["workstudy_hours"].fillna(self.workstudy_median_)
        df["financial_stress"] = df["financial_stress"].fillna(
            self.financial_stress_median_
        )

        # Sleep duration: map to ordinal then fill unknowns
        df["sleep_duration_raw"] = df["sleep_duration"]
        df["sleep_duration"] = (
            df["sleep_duration"].map(SLEEP_MAP).fillna(self.sleep_median_)
        )

        return df

    def _encode(self, df):
        """Binary, ordinal, and categorical encoding."""
        binary_map = {"Yes": 1, "No": 0}
        for col in ["suicidal_thoughts", "family_history", "depression"]:
            if col in df.columns:
                df[col] = df[col].map(binary_map)

        df["gender_enc"] = (df["gender"] == "Male").astype(int)
        df["is_professional"] = (df["status"] == "Working Professional").astype(int)
        df["dietary_enc"] = (
            df["dietary_habits"]
            .map({"Unhealthy": 0, "Moderate": 1, "Healthy": 2})
            .fillna(1)
        )
        df["unhealthy_diet"] = (df["dietary_habits"] == "Unhealthy").astype(int)

        # Education level (ordinal 0–3)
        df["education_level"] = (
            df["degree"].map(EDUCATION_MAP).fillna(self.education_median_)
        )

        # Profession category
        df["profession_category"] = df["profession"].map(PROFESSION_MAP)
        df["profession_category"] = df["profession_category"].where(
            df["profession_category"].notna(),
            np.where(df["status"] == "Student", "Student", "Other"),
        )

        return df

    def _fit_geocoding(self, df: pd.DataFrame) -> dict:
        """Geocode unique city names; returns city → (lat, lon) mapping."""
        if "city" not in df.columns:
            return {}
        geolocator = Nominatim(user_agent="patient_data_transformer")
        geocode = RateLimiter(
            geolocator.geocode, min_delay_seconds=1, error_wait_seconds=2
        )
        coords = {}
        for city in df["city"].dropna().unique():
            try:
                loc = geocode(city)
                coords[city] = (
                    (loc.latitude, loc.longitude) if loc else (np.nan, np.nan)
                )
            except Exception:
                coords[city] = (np.nan, np.nan)
        return coords

    def _apply_geocoding(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add city_lat and city_lon using the mapping learned in fit."""
        if not self.city_coords_:
            return df
        df["city_lat"] = df["city"].map(
            lambda c: self.city_coords_.get(c, (np.nan, np.nan))[0]
        )
        df["city_lon"] = df["city"].map(
            lambda c: self.city_coords_.get(c, (np.nan, np.nan))[1]
        )
        return df

    def _engineer(self, df):
        """Derive composite features."""
        df["stress_index"] = (
            df["pressure"] + df["financial_stress"] + (5 - df["sleep_duration"])
        ) / 3

        df["wellbeing_score"] = (
            df["satisfaction"] + df["sleep_duration"] + df["dietary_enc"]
        ) / 3

        df["physical_score"] = ((5 - df["sleep_duration"]) + df["dietary_enc"]) / 2

        df["emotional_score"] = (
            (5 - df["satisfaction"])
            + df["pressure"]
            + (5 * df["suicidal_thoughts"])
            + (4 * df["depression"])
        ) / 4

        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 22, 30, 40, 50, 100],
            labels=["≤22", "23-30", "31-40", "41-50", "51+"],
            right=True,
        )

        df["worklife_balance"] = 1 - (
            (df["workstudy_hours"] / self.max_hours_)
            * (1 - (df["sleep_duration"] - 1) / 3)
        )

        df["high_risk"] = (
            (df["stress_index"] >= self.stress_75_)
            & (
                (df["family_history"] == 1)
                | (df["suicidal_thoughts"] == 1)
                | (df["depression"] == 1)
            )
        ).astype(int)

        df["cgpa_band"] = pd.cut(
            df["cgpa"],
            bins=[0, 5, 6, 7, 8, 10],
            labels=["<5", "5-6", "6-7", "7-8", "8+"],
            right=True,
        )
        df["cgpa_band"] = df["cgpa_band"].astype(str).fillna("non_student")

        df["treatment_not_needed"] = (
            (df["depression"] == 0)
            & (df["pressure"] <= 2)
            & (df["satisfaction"] >= 4)
            & (df["suicidal_thoughts"] == 0)
            & (df["sleep_duration"] >= 2)
            & (df["dietary_enc"] >= 1)
        )

        return df

    def _finalize(self, df):
        """Drop rows with missing essentials, duplicates, and intermediate columns."""
        before = len(df)
        df.dropna(subset=ESSENTIAL_COLS, inplace=True)
        n_dropped_na = before - len(df)

        before = len(df)
        df.drop_duplicates(inplace=True)
        n_dropped_dup = before - len(df)

        df.index.name = "patient_id"

        self.n_dropped_na_ = n_dropped_na
        self.n_dropped_dup_ = n_dropped_dup

        return df

    def print_audit(self):
        """Print a summary of what fit() learned and any data quality issues."""
        check_is_fitted(self)
        print("=== PatientDataTransformer Audit ===")
        print(f"\nImputation medians (pressure)    : {self.pressure_medians_}")
        print(f"Imputation medians (satisfaction): {self.satisfaction_medians_}")
        print(f"CGPA median (students)           : {self.cgpa_median_:.3f}")
        print(f"Workstudy hours median           : {self.workstudy_median_}")
        print(f"Financial stress median          : {self.financial_stress_median_}")
        print(f"Sleep duration median            : {self.sleep_median_}")
        print(f"Education level median (fallback): {self.education_median_}")
        print(f"Max workstudy hours              : {self.max_hours_}")
        print(f"Stress index 75th pct (threshold): {self.stress_75_:.3f}")

        if self.edu_unmapped_.empty:
            print("\neducation_level : no unmapped degree strings.")
        else:
            print(
                f"\neducation_level : {self.edu_unmapped_.sum()} rows unmapped → imputed with median:"
            )
            print(self.edu_unmapped_.to_string())

        if self.prof_unmapped_.empty:
            print("\nprofession_category : no unmapped profession strings.")
        else:
            print(
                f"\nprofession_category : {self.prof_unmapped_.sum()} rows unmapped → labelled 'Other':"
            )
            print(self.prof_unmapped_.to_string())

        n_cities = len(self.city_coords_)
        n_geocoded = sum(1 for v in self.city_coords_.values() if not np.isnan(v[0]))
        print(f"\ncity geocoding : {n_geocoded}/{n_cities} cities resolved")
        if n_cities > n_geocoded:
            failed = [c for c, v in self.city_coords_.items() if np.isnan(v[0])]
            print(f"  unresolved   : {failed}")

        print(f"\nprofession_categories : {self.profession_categories_}")

        print(f"\nRows dropped (missing essentials): {self.n_dropped_na_}")
        print(f"Rows dropped (duplicates)        : {self.n_dropped_dup_}")


# ══════════════════════════════════════════════════════════════════════════════
# Script entry-point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    DATA_DIR = Path("data")
    DATA_DIR.mkdir(exist_ok=True)

    df_raw = pd.read_csv(DATA_DIR / "data.csv")
    print(f"Loaded  : {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
    print(f"Missing :\n{df_raw.isnull().sum().to_string()}\n")

    transformer = PatientDataTransformer()
    df_clean = transformer.fit_transform(df_raw)

    transformer.print_audit()

    print(f"\nClean shape : {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
    print(f"Columns     : {df_clean.columns.tolist()}")
    print(f"\n{df_clean.describe(include='all').to_string()}")

    out_path = DATA_DIR / "data_clean.csv"
    df_clean.to_csv(out_path, index=True)
    print(f"\nSaved → {out_path}")
