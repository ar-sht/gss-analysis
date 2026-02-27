import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy import stats
except ImportError:  # pragma: no cover - optional dependency for t-tests
    stats = None

def main() -> None:
    df = pd.read_excel("simple-data.xlsx")
    df["wordsum"] = pd.to_numeric(df["wordsum"], errors="coerce")

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
    parties = ["D", "R", "I"]
    fig, axes = plt.subplots(len(periods), len(parties), figsize=(12, 6), sharex=True, sharey=True)

    for row_idx, period in enumerate(periods):
        for col_idx, party in enumerate(parties):
            ax = axes[row_idx, col_idx]
            subset = df[(df["period"] == period) & (df["partyid"] == party)]["wordsum"].dropna()
            print(f"Period {period} Party {party} wordsum stats:")
            summary = {
                "count": int(subset.count()),
                "mean": subset.mean(),
                "median": subset.median(),
                "std": subset.std(),
                "min": subset.min(),
                "25%": subset.quantile(0.25),
                "75%": subset.quantile(0.75),
                "max": subset.max(),
            }
            print(summary)
            ax.hist(subset, bins=range(1, 11), edgecolor="black")
            ax.set_title(f"{period} - {party}")
            ax.set_xlabel("wordsum")
            ax.set_ylabel("count")

    if stats is None:
        print("scipy is not installed; skipping t-tests.")
    else:
        for period in periods:
            print(f"T-tests for period {period}:")
            period_df = df[df["period"] == period]
            groups = {
                party: period_df[period_df["partyid"] == party]["wordsum"].dropna()
                for party in parties
            }
            for left, right in [("D", "R"), ("D", "I"), ("R", "I")]:
                t_stat, p_value = stats.ttest_ind(
                    groups[left], groups[right], equal_var=False, nan_policy="omit"
                )
                print(
                    f"{left} vs {right}: t={t_stat:.4f}, p={p_value:.4g}, "
                    f"n1={len(groups[left])}, n2={len(groups[right])}"
                )

    threshold_counts = {}
    for period in periods:
        period_wordsum = df[df["period"] == period]["wordsum"].dropna()
        mean = period_wordsum.mean()
        std = period_wordsum.std()
        threshold = mean + 2 * std
        print(f"{period} population wordsum mean={mean:.4f}, std={std:.4f}, threshold={threshold:.4f}")
        if threshold > 10:
            print(f"{period} threshold exceeds 10; treating wordsum == 10 as meeting it.")
        period_counts = {}
        for party in parties:
            party_values = df[
                (df["period"] == period) & (df["partyid"] == party)
            ]["wordsum"].dropna()
            if threshold > 10:
                meets_threshold = party_values >= 10
            else:
                meets_threshold = party_values > threshold
            period_counts[party] = int(meets_threshold.sum())
        threshold_counts[period] = period_counts
        print(f"{period} counts above 2-sigma: {period_counts}")

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    bar_width = 0.35
    x_positions = range(len(parties))
    pre_counts = [threshold_counts["Pre-Trump"][party] for party in parties]
    post_counts = [threshold_counts["Post-Trump"][party] for party in parties]
    ax2.bar(
        [x - bar_width / 2 for x in x_positions],
        pre_counts,
        width=bar_width,
        label="Pre-Trump",
    )
    ax2.bar(
        [x + bar_width / 2 for x in x_positions],
        post_counts,
        width=bar_width,
        label="Post-Trump",
    )
    ax2.set_xticks(list(x_positions))
    ax2.set_xticklabels(parties)
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
        )
        ax4.set_title("Post-Trump: 2-sigma exceeders by party")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
