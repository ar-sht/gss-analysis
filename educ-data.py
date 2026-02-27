import pandas as pd
import re


def is_invalid_party(value) -> bool:
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


def is_invalid_educ(value) -> bool:
    if pd.isna(value):
        return True
    if isinstance(value, str):
        normalized = re.sub(r"\s+", " ", value.strip())
        return normalized in {".n: No answer", ""}
    return False


def map_party_label(value) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"not very strong republican", "strong republican"}:
            return "R"
        if normalized in {"not very strong democrat", "strong democrat"}:
            return "D"
    return "I"


def main() -> None:
    df = pd.read_excel("educ-data.xlsx")
    invalid_party_rows = df["partyid"].apply(is_invalid_party)
    invalid_educ_rows = df["educ"].apply(is_invalid_educ)
    cleaned = df.loc[~(invalid_party_rows | invalid_educ_rows)]
    cleaned.to_excel("clean-educ-data.xlsx", index=False)

    simple_df = cleaned.copy()
    simple_df["partyid"] = simple_df["partyid"].apply(map_party_label)
    simple_df.to_excel("simple-educ-data.xlsx", index=False)


if __name__ == "__main__":
    main()
