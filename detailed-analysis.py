import itertools
import math

import pandas as pd
import matplotlib.pyplot as plt

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


def main() -> None:
    df = pd.read_excel("clean-data.xlsx")
    df["wordsum"] = pd.to_numeric(df["wordsum"], errors="coerce")
    party_aliases = {
        "Independent, close to democrat": "Independent, near democrat",
        "Independent, close to republican": "Independent, near republican",
        "Independent (neither, no response)": "Independent",
    }
    df["partyid"] = df["partyid"].replace(party_aliases)

    party_order = [
        "Strong democrat",
        "Not very strong democrat",
        "Independent, near democrat",
        "Independent",
        "Independent, near republican",
        "Not very strong republican",
        "Strong republican",
        "Other",
    ]
    party_colors = {
        "Strong republican": "#8B0000",
        "Not very strong republican": "#D33B3B",
        "Strong democrat": "#0B3D91",
        "Not very strong democrat": "#2E6FD8",
        "Independent, near republican": "#F7DADA",
        "Independent, near democrat": "#DCEBFF",
        "Independent": "#BDBDBD",
        "Other": "#D2B48C",
    }

    period_map = {
        2010: "Pre-Trump",
        2012: "Pre-Trump",
        2014: "Pre-Trump",
        2018: "Post-Trump",
        2022: "Post-Trump",
        2024: "Post-Trump",
    }
    df["period"] = df["year"].map(period_map)
    periods = ["Pre-Trump", "Post-Trump"]

    parties = df["partyid"].dropna().astype(str).unique().tolist()
    ordered_parties = [p for p in party_order if p in parties]
    other_parties = [p for p in parties if p not in party_order]
    if other_parties:
        df["partyid"] = df["partyid"].apply(lambda value: "Other" if value in other_parties else value)
        ordered_parties.append("Other")
    parties = ordered_parties

    threshold_counts = {period: {} for period in periods}

    for period in periods:
        period_df = df[df["period"] == period]
        period_wordsum = period_df["wordsum"].dropna()
        mean = period_wordsum.mean()
        std = period_wordsum.std()
        threshold = mean + 2 * std
        print(
            f"{period} population wordsum mean={mean:.4f}, std={std:.4f}, threshold={threshold:.4f}"
        )
        if threshold > 10:
            print(f"{period} threshold exceeds 10; treating wordsum == 10 as meeting it.")

        print(f"Period {period} wordsum stats by party:")
        for party in parties:
            subset = period_df[period_df["partyid"] == party]["wordsum"].dropna()
            if subset.empty:
                continue
            print(f"Party {party}: {summarize(subset)}")

        if stats is None:
            print("scipy is not installed; skipping t-tests.")
        else:
            print(f"T-tests for period {period}:")
            party_groups = {
                party: period_df[period_df["partyid"] == party]["wordsum"].dropna()
                for party in parties
                if not period_df[period_df["partyid"] == party]["wordsum"].dropna().empty
            }
            for left, right in itertools.combinations(party_groups.keys(), 2):
                t_stat, p_value = stats.ttest_ind(
                    party_groups[left], party_groups[right], equal_var=False, nan_policy="omit"
                )
                print(
                    f"{left} vs {right}: t={t_stat:.4f}, p={p_value:.4g}, "
                    f"n1={len(party_groups[left])}, n2={len(party_groups[right])}"
                )

        period_counts = {}
        for party in parties:
            party_values = period_df[period_df["partyid"] == party]["wordsum"].dropna()
            if threshold > 10:
                meets_threshold = party_values >= 10
            else:
                meets_threshold = party_values > threshold
            period_counts[party] = int(meets_threshold.sum())
        threshold_counts[period] = period_counts
        print(f"{period} counts above 2-sigma: {period_counts}")

        if parties:
            cols = 3
            rows = math.ceil(len(parties) / cols)
            fig, axes = plt.subplots(
                rows, cols, figsize=(12, max(4, rows * 3)), sharex=True, sharey=True
            )
            axes_list = axes.ravel() if hasattr(axes, "ravel") else [axes]
            for ax, party in zip(axes_list, parties):
                subset = period_df[period_df["partyid"] == party]["wordsum"].dropna()
                ax.hist(
                    subset,
                    bins=range(1, 11),
                    edgecolor="black",
                    color=party_colors.get(party),
                )
                ax.set_title(str(party))
                ax.set_xlabel("wordsum")
                ax.set_ylabel("count")
            for ax in axes_list[len(parties):]:
                ax.axis("off")
            fig.suptitle(f"{period} wordsum histograms by party")

    fig2, ax2 = plt.subplots(figsize=(10, 5))
    x_positions = range(len(parties))
    bar_width = 0.35
    pre_counts = [threshold_counts["Pre-Trump"].get(party, 0) for party in parties]
    post_counts = [threshold_counts["Post-Trump"].get(party, 0) for party in parties]
    bar_colors = [party_colors.get(party, "#CCCCCC") for party in parties]
    ax2.bar(
        [x - bar_width / 2 for x in x_positions],
        pre_counts,
        width=bar_width,
        label="Pre-Trump",
        color=bar_colors,
    )
    ax2.bar(
        [x + bar_width / 2 for x in x_positions],
        post_counts,
        width=bar_width,
        label="Post-Trump",
        color=bar_colors,
    )
    ax2.set_xticks(list(x_positions))
    ax2.set_xticklabels(parties, rotation=45, ha="right")
    ax2.set_ylabel("Count above 2-sigma threshold")
    ax2.set_title("Wordsum 2-sigma exceeders by party and era")
    ax2.legend()

    fig3, ax3 = plt.subplots(figsize=(8, 4))
    if sum(pre_counts) == 0:
        ax3.text(0.5, 0.5, "No exceeders", ha="center", va="center")
        ax3.set_title("Pre-Trump: 2-sigma exceeders by party")
        ax3.axis("off")
    else:
        ax3.pie(
            pre_counts,
            labels=parties,
            autopct="%1.1f%%",
            startangle=90,
            colors=bar_colors,
        )
        ax3.set_title("Pre-Trump: 2-sigma exceeders by party")

    fig4, ax4 = plt.subplots(figsize=(8, 4))
    if sum(post_counts) == 0:
        ax4.text(0.5, 0.5, "No exceeders", ha="center", va="center")
        ax4.set_title("Post-Trump: 2-sigma exceeders by party")
        ax4.axis("off")
    else:
        ax4.pie(
            post_counts,
            labels=parties,
            autopct="%1.1f%%",
            startangle=90,
            colors=bar_colors,
        )
        ax4.set_title("Post-Trump: 2-sigma exceeders by party")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
