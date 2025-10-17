import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"date", "description", "amount", "category"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df["type"] = np.where(df["amount"] >= 0, "income", "expense")
    return df

def monthly_trends(df: pd.DataFrame) -> pd.DataFrame:
    monthly = df.groupby(["month", "type",])["amount"].sum().unstack(fill_value=0)
    if "income" not in monthly.columns: monthly["income"] = 0
    if "expense" not in monthly.columns: monthly["expense"] = 0
    monthly = monthly.sort_index()
    monthly["net"] = monthly["income"] + monthly["expense"]
    return monthly

def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    exp = df[df["amount"] < 0].copy()
    by_cat = exp.groupby("category")["amount"].sum().sort_values()
    return by_cat

def save_charts(monthly: pd.DataFrame, by_cat: pd.Series):
    plt.figure()
    monthly[["income", "expense", "net"]].plot(marker="o")
    plt.title("Monthly Income / Expenses / Net")
    plt.xlabel("Month")
    plt.ylabel("Amount")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_trend.png")
    plt.close()

    plt.figure()
    top10 = by_cat.abs().sort_values(ascending=False).head(10) * -1  
    top10.plot(kind="barh")
    plt.title("Top 10 Spending Categories")
    plt.xlabel("Amount Spent")
    plt.ylabel("Category")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "category_spend.png")
    plt.close()

    plt.figure()
    cum_net = monthly["net"].cumsum()
    cum_net.plot(marker="o")
    plt.title("Cumulative Net Cashflow")
    plt.xlabel("Month")
    plt.ylabel("Cumulative Net")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cumulative_net.png")
    plt.close()

def write_summary(monthly: pd.DataFrame, by_cat: pd.Series):
    total_income = monthly["income"].sum()
    total_expense = monthly["expense"].sum()
    total_net = monthly["net"].sum()
    top5 = (by_cat.abs().sort_values(ascending=False).head(5) * -1).round(2)

    lines = [
        "# Personal Finance Summary",
        "",
        f"- **Total income**: ${total_income:,.2f}",
        f"- **Total expenses**: ${-total_expense:,.2f}",
        f"- **Net**: ${total_net:,.2f}",
        "",
        "## Top 5 Spending Categories",
    ]
    for cat, amt in top5.items():
        lines.append(f"- {cat}: ${amt:,.2f}")

    (OUTPUT_DIR / "summary_report.md").write_text("\n".join(lines), encoding="utf-8")

def main():
    parser = argparse.ArgumentParser(description="Analyze personal finance CSV.")
    parser.add_argument("--csv", required=True, help="Path to transactions CSV.")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    df = load_data(csv_path)
    monthly = monthly_trends(df)
    by_cat = category_breakdown(df)
    save_charts(monthly, by_cat)
    write_summary(monthly, by_cat)
    print("Done! Check the 'output/' folder for charts and summary_report.md.")

if __name__ == "__main__":
    main()
