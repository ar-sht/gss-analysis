import matplotlib.pyplot as plt
import pandas as pd

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency for t-tests
    stats = None


def summarize(series: pd.Series) -> dict:
    return {
        "count": int(series.count()),
        "mean": series.mean(),
        "median": series.median(),
        "std": series.std(),
        "min": series.min(),
        "25%": series.quantile(0.25),
        "75%": series.quantile(0.75),
        "max": series.max(),
    }


def build_election_data(
    df: pd.DataFrame, label: str, column: str, candidates: list[str], years: list[int]
) -> pd.DataFrame:
    subset = df[df["year"].isin(years)].copy()
    subset[column] = subset[column].astype(str)
    subset = subset[subset[column].isin(candidates)]
    subset = subset[["wordsum", column]].rename(columns={column: "voter"})
    subset["election"] = label
    return subset


def main() -> None:
    df = pd.read_excel("pres-data.xlsx")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["wordsum"] = pd.to_numeric(df["wordsum"], errors="coerce")
    df = df[df["wordsum"].between(1, 10, inclusive="both")]

    election_configs = [
        {
            "label": "2012 Election",
            "column": "pres12",
            "candidates": ["Obama", "Romney"],
            "years": [2014, 2018],
        },
        {
            "label": "2016 Election",
            "column": "pres16",
            "candidates": ["Clinton", "Trump"],
            "years": [2018, 2022],
        },
        {
            "label": "2020 Election",
            "column": "pres20",
            "candidates": ["Biden", "Trump"],
            "years": [2022],
        },
    ]

    election_frames = []
    for config in election_configs:
        election_df = build_election_data(
            df,
            config["label"],
            config["column"],
            config["candidates"],
            config["years"],
        )
        if election_df.empty:
            print(f"No data available for {config['label']}.")
            continue
        election_frames.append((config, election_df))

    if not election_frames:
        print("No election data available to analyze.")
        return

    fig, axes = plt.subplots(
        len(election_frames),
        2,
        figsize=(10, 4 * len(election_frames)),
        sharex=True,
        sharey=True,
    )
    if len(election_frames) == 1:
        axes = [axes]

    threshold_counts = {}

    for row_idx, (config, election_df) in enumerate(election_frames):
        election_label = config["label"]
        candidates = config["candidates"]
        period_wordsum = election_df["wordsum"].dropna()
        mean = period_wordsum.mean()
        std = period_wordsum.std()
        threshold = mean + 2 * std
        print(
            f"{election_label} population wordsum mean={mean:.4f}, std={std:.4f}, "
            f"threshold={threshold:.4f}"
        )
        if threshold > 10:
            print(
                f"{election_label} threshold exceeds 10; treating wordsum == 10 as meeting it."
            )

        print(f"{election_label} wordsum stats by voter:")
        for col_idx, candidate in enumerate(candidates):
            subset = election_df[election_df["voter"] == candidate]["wordsum"].dropna()
            print(f"Voter {candidate}: {summarize(subset)}")
            ax = axes[row_idx][col_idx]
            ax.hist(subset, bins=range(1, 11), edgecolor="black")
            ax.set_title(f"{election_label} - {candidate}")
            ax.set_xlabel("wordsum")
            ax.set_ylabel("count")

        if stats is None:
            print("scipy is not installed; skipping t-tests.")
        else:
            groups = {
                candidate: election_df[election_df["voter"] == candidate]["wordsum"].dropna()
                for candidate in candidates
            }
            if all(len(values) > 0 for values in groups.values()):
                t_stat, p_value = stats.ttest_ind(
                    groups[candidates[0]],
                    groups[candidates[1]],
                    equal_var=False,
                    nan_policy="omit",
                )
                print(
                    f"T-test for {election_label}: {candidates[0]} vs {candidates[1]} "
                    f"t={t_stat:.4f}, p={p_value:.4g}, "
                    f"n1={len(groups[candidates[0]])}, n2={len(groups[candidates[1]])}"
                )
            else:
                print(f"Insufficient data for t-test in {election_label}.")

        election_counts = {}
        for candidate in candidates:
            candidate_values = election_df[election_df["voter"] == candidate][
                "wordsum"
            ].dropna()
            if threshold > 10:
                meets_threshold = candidate_values >= 10
            else:
                meets_threshold = candidate_values > threshold
            election_counts[candidate] = int(meets_threshold.sum())
        threshold_counts[election_label] = election_counts
        print(f"{election_label} counts above 2-sigma: {election_counts}")

        fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
        ax_bar.bar(list(election_counts.keys()), list(election_counts.values()))
        ax_bar.set_ylabel("Count above 2-sigma threshold")
        ax_bar.set_title(f"{election_label}: 2-sigma exceeders by voter")

        fig_pie, ax_pie = plt.subplots(figsize=(6, 4))
        counts = list(election_counts.values())
        if sum(counts) == 0:
            ax_pie.text(0.5, 0.5, "No exceeders", ha="center", va="center")
            ax_pie.set_title(f"{election_label}: 2-sigma exceeders by voter")
            ax_pie.axis("off")
        else:
            ax_pie.pie(
                counts,
                labels=list(election_counts.keys()),
                autopct="%1.1f%%",
                startangle=90,
            )
            ax_pie.set_title(f"{election_label}: 2-sigma exceeders by voter")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
