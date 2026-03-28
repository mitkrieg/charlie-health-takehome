"""
EDA plots for the cleaned patient mental health dataset.

Expects data/data_clean.csv to exist (produced by data_cleaning.py).

Run standalone:
    python eda.py
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore")

PLOTS_DIR = Path("eda_plots")
PLOTS_DIR.mkdir(exist_ok=True)

PALETTE = "Set2"
sns.set_theme(style="whitegrid", palette=PALETTE)

EDU_LABELS = {0: "High School", 1: "Bachelor's", 2: "Master's", 3: "PhD"}


def save(fig, name):
    path = PLOTS_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


# ── 01  Target distribution ───────────────────────────────────────────────────
def plot_target(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Depression (Target) Distribution", fontsize=13, fontweight="bold")

    counts = df["depression"].value_counts().rename({0: "No", 1: "Yes"})
    axes[0].pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=sns.color_palette(PALETTE, 2), startangle=140,
    )
    axes[0].set_title("Proportion")

    sns.countplot(
        x="depression",
        data=df.assign(depression=df["depression"].map({0: "No", 1: "Yes"})),
        ax=axes[1], palette=PALETTE,
    )
    axes[1].set_title("Count")
    axes[1].set_xlabel("Depression")
    save(fig, "01_target_distribution")


# ── 02  Age distribution ──────────────────────────────────────────────────────
def plot_age(df):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Age Distribution", fontsize=13, fontweight="bold")

    sns.histplot(df["age"], bins=30, kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Overall")

    sns.kdeplot(
        data=df, x="age",
        hue=df["depression"].map({0: "No", 1: "Yes"}),
        fill=True, alpha=0.4, ax=axes[1], palette=PALETTE,
    )
    axes[1].set_title("By Depression")
    save(fig, "02_age_distribution")


# ── 03  Depression rate by key categoricals ───────────────────────────────────
def plot_depression_by_categoricals(df):
    cat_cols = {
        "gender": "Gender",
        "status": "Student / Professional",
        "sleep_duration_raw": "Sleep Duration",
        "dietary_habits": "Dietary Habits",
        "age_group": "Age Group",
    }

    fig, axes = plt.subplots(1, len(cat_cols), figsize=(20, 5))
    fig.suptitle(
        "Depression Rate by Demographic / Lifestyle Factor",
        fontsize=13, fontweight="bold",
    )

    for ax, (col, title) in zip(axes, cat_cols.items()):
        rate = df.groupby(col)["depression"].mean().sort_values(ascending=False) * 100
        rate.plot(kind="bar", ax=ax, color=sns.color_palette(PALETTE, len(rate)))
        ax.set_title(title)
        ax.set_ylabel("Depression rate (%)")
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        for bar in ax.patches:
            ax.annotate(
                f"{bar.get_height():.1f}%",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    save(fig, "03_depression_by_categoricals")


# ── 04  Numeric features by depression (violins) ─────────────────────────────
def plot_numeric_by_depression(df):
    num_features = [
        "pressure", "satisfaction", "financial_stress",
        "workstudy_hours", "stress_index", "wellbeing_score",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "Numeric Feature Distributions by Depression Status",
        fontsize=13, fontweight="bold",
    )

    dep_label = df["depression"].map({0: "No", 1: "Yes"})
    for ax, feat in zip(axes.flat, num_features):
        sns.violinplot(
            x=dep_label, y=df[feat], ax=ax, palette=PALETTE, inner="quartile"
        )
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel("Depression")

    plt.tight_layout()
    save(fig, "04_numeric_by_depression")


# ── 05  Correlation heatmap ───────────────────────────────────────────────────
def plot_correlation(df):
    corr_cols = [
        "age", "pressure", "satisfaction", "sleep_duration",
        "financial_stress", "workstudy_hours", "suicidal_thoughts",
        "family_history", "stress_index", "wellbeing_score",
        "worklife_balance", "gender_enc", "is_professional",
        "dietary_enc", "education_level", "depression",
    ]
    corr = df[corr_cols].corr()

    fig, ax = plt.subplots(figsize=(13, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
        center=0, linewidths=0.5, ax=ax, annot_kws={"size": 8},
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save(fig, "05_correlation_heatmap")


# ── 06  Stress vs wellbeing scatter ──────────────────────────────────────────
def plot_stress_vs_wellbeing(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        df["stress_index"], df["wellbeing_score"],
        c=df["depression"], cmap="RdYlGn_r",
        alpha=0.4, edgecolors="none", s=20,
    )
    plt.colorbar(scatter, ax=ax, label="Depression (1=Yes)")
    ax.set_xlabel("Stress Index")
    ax.set_ylabel("Wellbeing Score")
    ax.set_title(
        "Stress vs Wellbeing (coloured by Depression)", fontsize=12, fontweight="bold"
    )
    save(fig, "06_stress_vs_wellbeing")


# ── 07  High-risk flag analysis ───────────────────────────────────────────────
def plot_high_risk(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("High-Risk Flag", fontsize=13, fontweight="bold")

    counts_hr = df["high_risk"].value_counts().rename({0: "Normal", 1: "High-Risk"})
    axes[0].pie(
        counts_hr, labels=counts_hr.index, autopct="%1.1f%%",
        colors=sns.color_palette(PALETTE, 2), startangle=140,
    )
    axes[0].set_title("Proportion")

    dep_by_risk = df.groupby("high_risk")["depression"].mean() * 100
    dep_by_risk.index = ["Normal", "High-Risk"]
    dep_by_risk.plot(kind="bar", ax=axes[1], color=sns.color_palette(PALETTE, 2))
    axes[1].set_title("Depression Rate by Risk Flag")
    axes[1].set_ylabel("Depression rate (%)")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=0)
    save(fig, "07_high_risk_analysis")


# ── 08  Pairplot of core clustering features ─────────────────────────────────
def plot_pairplot(df):
    cluster_feats = [
        "age", "pressure", "satisfaction", "sleep_duration",
        "financial_stress", "stress_index", "wellbeing_score",
    ]
    pair_df = df[cluster_feats + ["depression"]].copy()
    pair_df["depression"] = pair_df["depression"].map({0: "No", 1: "Yes"})

    pg = sns.pairplot(
        pair_df, hue="depression", palette=PALETTE,
        plot_kws={"alpha": 0.3, "s": 10}, diag_kind="kde", corner=True,
    )
    pg.figure.suptitle(
        "Pairplot — Core Clustering Features", y=1.01, fontsize=13, fontweight="bold"
    )
    save(pg.figure, "08_pairplot_clustering_features")


# ── 09  Work/study hours by status & depression ───────────────────────────────
def plot_hours_by_status(df):
    dep_label = df["depression"].map({0: "No", 1: "Yes"})
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(
        x="status", y="workstudy_hours", hue=dep_label,
        data=df, ax=ax, palette=PALETTE,
    )
    ax.set_title(
        "Work/Study Hours by Status & Depression", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("")
    ax.set_ylabel("Hours per day")
    save(fig, "09_hours_by_status")


# ── 10  Financial stress heatmap (age group × status) ────────────────────────
def plot_financial_stress_heatmap(df):
    pivot = df.pivot_table(
        values="financial_stress", index="age_group",
        columns="status", aggfunc="mean",
    )
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="YlOrRd", linewidths=0.5, ax=ax)
    ax.set_title(
        "Avg Financial Stress by Age Group & Status", fontsize=12, fontweight="bold"
    )
    save(fig, "10_financial_stress_heatmap")


# ── 11  Education level ───────────────────────────────────────────────────────
def plot_education(df):
    df = df.copy()
    df["edu_label"] = df["education_level"].map(EDU_LABELS)
    order = list(EDU_LABELS.values())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Education Level", fontsize=13, fontweight="bold")

    edu_counts = df["edu_label"].value_counts().reindex(order)
    sns.barplot(x=edu_counts.index, y=edu_counts.values, ax=axes[0],
                palette=PALETTE, order=order)
    axes[0].set_title("Patient Count")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=20)

    edu_dep = df.groupby("edu_label")["depression"].mean().reindex(order) * 100
    sns.barplot(x=edu_dep.index, y=edu_dep.values, ax=axes[1],
                palette=PALETTE, order=order)
    axes[1].set_title("Depression Rate (%)")
    axes[1].set_ylabel("Depression rate (%)")
    axes[1].tick_params(axis="x", rotation=20)
    for bar in axes[1].patches:
        axes[1].annotate(
            f"{bar.get_height():.1f}%",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9,
        )

    edu_stress = df.groupby("edu_label")["stress_index"].mean().reindex(order)
    sns.barplot(x=edu_stress.index, y=edu_stress.values, ax=axes[2],
                palette=PALETTE, order=order)
    axes[2].set_title("Avg Stress Index")
    axes[2].set_ylabel("Stress index")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    save(fig, "11_education_level")


# ── 12  Profession category ───────────────────────────────────────────────────
def plot_profession_category(df):
    prof_df = df[df["profession_category"] != "Student"].copy()
    cat_order = (
        prof_df.groupby("profession_category")["depression"].mean()
        .sort_values(ascending=False).index.tolist()
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Profession Category (Working Professionals Only)",
        fontsize=13, fontweight="bold",
    )

    cat_counts = prof_df["profession_category"].value_counts().reindex(cat_order)
    sns.barplot(x=cat_counts.values, y=cat_counts.index, ax=axes[0],
                palette=PALETTE, orient="h")
    axes[0].set_title("Patient Count")
    axes[0].set_xlabel("Count")

    cat_dep = (
        prof_df.groupby("profession_category")["depression"].mean() * 100
    ).reindex(cat_order)
    sns.barplot(x=cat_dep.values, y=cat_dep.index, ax=axes[1],
                palette=PALETTE, orient="h")
    axes[1].set_title("Depression Rate (%)")
    axes[1].set_xlabel("Depression rate (%)")
    for i, v in enumerate(cat_dep.values):
        axes[1].text(v + 0.3, i, f"{v:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    save(fig, "12_profession_category")


# ── 13  Stress index by profession category ───────────────────────────────────
def plot_stress_by_profession(df):
    prof_df = df[df["profession_category"] != "Student"].copy()
    stress_order = (
        prof_df.groupby("profession_category")["stress_index"].median()
        .sort_values(ascending=False).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(
        data=prof_df, x="profession_category", y="stress_index",
        order=stress_order, palette=PALETTE, ax=ax,
    )
    ax.set_title("Stress Index by Profession Category", fontsize=12, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("Stress Index")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "13_stress_by_profession")


# ── 14  Depression rate: education × profession heatmap ──────────────────────
def plot_education_profession_heatmap(df):
    edu_prof = df[df["profession_category"] != "Student"].copy()
    edu_prof["edu_label"] = edu_prof["education_level"].map(EDU_LABELS)

    pivot = edu_prof.pivot_table(
        values="depression", index="edu_label",
        columns="profession_category", aggfunc="mean",
    ) * 100
    pivot = pivot.reindex(list(EDU_LABELS.values()))

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot, annot=True, fmt=".1f", cmap="RdYlGn_r",
        linewidths=0.5, ax=ax, annot_kws={"size": 8},
    )
    ax.set_title(
        "Depression Rate (%) by Education Level × Profession Category",
        fontsize=12, fontweight="bold",
    )
    ax.set_ylabel("Education Level")
    ax.set_xlabel("")
    plt.tight_layout()
    save(fig, "14_education_x_profession_heatmap")


# ══════════════════════════════════════════════════════════════════════════════

def run_all(df):
    plot_target(df)
    plot_age(df)
    plot_depression_by_categoricals(df)
    plot_numeric_by_depression(df)
    plot_correlation(df)
    plot_stress_vs_wellbeing(df)
    plot_high_risk(df)
    plot_pairplot(df)
    plot_hours_by_status(df)
    plot_financial_stress_heatmap(df)
    plot_education(df)
    plot_profession_category(df)
    plot_stress_by_profession(df)
    plot_education_profession_heatmap(df)
    print(f"\nAll plots saved to: {PLOTS_DIR.resolve()}")


if __name__ == "__main__":
    df = pd.read_csv("data/data_clean.csv")
    print(f"Loaded clean dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    run_all(df)
