import pandas as pd
import matplotlib.pyplot as plt


def main() -> None:
    df = pd.read_excel("simple-educ-data.xlsx")
    df["educ"] = df["educ"].astype(str)

    parties = ["D", "R", "I"]
    for party in parties:
        subset = df[df["partyid"] == party]["educ"].dropna()
        print(f"Party {party} education distribution (count={len(subset)}):")
        counts = subset.value_counts()
        percents = (counts / counts.sum() * 100).round(2)
        summary = pd.DataFrame({"count": counts, "percent": percents})
        print(summary)

    top_categories = df["educ"].value_counts().head(10).index.tolist()
    if not top_categories:
        print("No education categories available to plot.")
        return

    x_positions = range(len(top_categories))
    bar_width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    for idx, party in enumerate(parties):
        party_counts = (
            df[df["partyid"] == party]["educ"]
            .value_counts()
            .reindex(top_categories, fill_value=0)
        )
        ax.bar(
            [x + (idx - 1) * bar_width for x in x_positions],
            party_counts.values,
            width=bar_width,
            label=party,
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(top_categories, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title("Top education categories by party (simple labels)")
    ax.legend(title="Party")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
