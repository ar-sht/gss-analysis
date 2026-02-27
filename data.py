import pandas as pd
import re


def is_invalid_cell(value) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str):
        normalized = re.sub(r"\s+", " ", value.strip())
        return normalized in {
            ".n: No answer",
            ".i: Inapplicable",
            ".d: Do not Know/Cannot Choose",
            "",
        }

    return False


def main() -> None:
    df = pd.read_excel("data.xlsx")
    invalid_rows = df.applymap(is_invalid_cell).any(axis=1)
    cleaned = df.loc[~invalid_rows]
    cleaned["year"] = pd.to_numeric(cleaned["year"], errors="coerce")
    cleaned = cleaned[cleaned["year"].isin([2010, 2012, 2014, 2018, 2022, 2024])]
    cleaned["wordsum"] = pd.to_numeric(cleaned["wordsum"], errors="coerce")
    cleaned = cleaned[cleaned["wordsum"].between(1, 10, inclusive="both")]
    cleaned.to_excel("clean-data.xlsx", index=False)

    simple_df = pd.read_excel("clean-data.xlsx")
    simple_df.iloc[:, 2] = simple_df.iloc[:, 2].apply(map_party_label)
    simple_df.to_excel("simple-data.xlsx", index=False)


def map_party_label(value) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"not very strong republican", "strong republican"}:
            return "R"
        if normalized in {"not very strong democrat", "strong democrat"}:
            return "D"
    return "I"


if __name__ == "__main__":
    main()
