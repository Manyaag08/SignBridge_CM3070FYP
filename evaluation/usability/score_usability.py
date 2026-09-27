"""Score the SignBridge usability study: SUS (0-100, pass 68), task success and comprehension.
Usage: python score_usability.py responses.csv   (same columns as sus_responses_template.csv)"""
import sys
import pandas as pd

def sus(row):
    odd = sum(row[f"q{i}"] - 1 for i in (1, 3, 5, 7, 9))
    even = sum(5 - row[f"q{i}"] for i in (2, 4, 6, 8, 10))
    return (odd + even) * 2.5

if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1])
    if df.empty:
        sys.exit("No responses yet: the study has not been run.")
    df["sus"] = df.apply(sus, axis=1)
    tasks = [c for c in df.columns if c.endswith("_success")]
    print(df.groupby("group").agg(n=("sus", "size"), sus_mean=("sus", "mean"), sus_sd=("sus", "std")))
    print("SUS >= 68:", (df.sus >= 68).mean().round(3))
    print("task success:", df[tasks].mean().round(3).to_dict())
    h = df[df.group == "hearing"]
    if len(h):
        print("comprehension:", round(h.comprehension_correct.sum() / h.comprehension_total.sum(), 3))
