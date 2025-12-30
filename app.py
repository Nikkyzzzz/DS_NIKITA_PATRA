import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

SENTIMENT_ORDER = ["Extreme Fear", "Fear", "Neutral", "Greed", "Extreme Greed"]
COLORS_MAP = {
    "Extreme Fear": "#d62728",
    "Fear": "#ff7f0e",
    "Neutral": "#8c8c8c",
    "Greed": "#2ca02c",
    "Extreme Greed": "#1f77b4",
}


def classify_behavior(row: pd.Series) -> str:
    side = str(row.get("Side", "")).upper()
    sentiment = row.get("classification", "")
    if side == "BUY":
        if sentiment in ["Extreme Fear", "Fear"]:
            return "Contrarian"
        if sentiment in ["Greed", "Extreme Greed"]:
            return "Conformist"
    elif side == "SELL":
        if sentiment in ["Greed", "Extreme Greed"]:
            return "Contrarian"
        if sentiment in ["Extreme Fear", "Fear"]:
            return "Conformist"
    return "Neutral"


@st.cache_data(show_spinner=False)
def load_local_data(base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = Path(base_dir) / "csv_files"
    fg_path = data_dir / "fear_greed_index.csv"
    trades_path = data_dir / "historical_data.csv"

    fg_df = pd.read_csv(fg_path)
    trades_df = pd.read_csv(trades_path)

    return fg_df, trades_df


def validate_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def preprocess_data(fg_df: pd.DataFrame, trades_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    validate_columns(fg_df, ["date", "value", "classification"], "Fear & Greed CSV")
    validate_columns(
        trades_df,
        ["Trade ID", "Account", "Coin", "Side", "Size USD", "Closed PnL", "Timestamp IST"],
        "Historical Trades CSV",
    )

    fg_df["date"] = pd.to_datetime(fg_df["date"], errors="coerce")
    trades_df["Timestamp IST"] = pd.to_datetime(
        trades_df["Timestamp IST"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    trades_df["date"] = pd.to_datetime(trades_df["Timestamp IST"].dt.date, errors="coerce")

    for col in ["Size USD", "Closed PnL"]:
        trades_df[col] = pd.to_numeric(trades_df[col], errors="coerce")

    merged_df = trades_df.merge(
        fg_df[["date", "value", "classification"]], on="date", how="inner"
    )
    merged_df["sentiment_numeric"] = merged_df["classification"].map(
        {
            "Extreme Fear": 1,
            "Fear": 2,
            "Neutral": 3,
            "Greed": 4,
            "Extreme Greed": 5,
        }
    )
    merged_df["behavior_type"] = merged_df.apply(classify_behavior, axis=1)

    return fg_df, merged_df


def load_data_with_option(base_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    st.sidebar.header("Data Source")
    source = st.sidebar.radio("Choose input source", ["Local files", "Upload CSV files"])

    if source == "Upload CSV files":
        fg_upload = st.sidebar.file_uploader(
            "Upload fear_greed_index.csv", type=["csv"], key="fg_upload"
        )
        trades_upload = st.sidebar.file_uploader(
            "Upload historical_data.csv", type=["csv"], key="trades_upload"
        )

        if fg_upload is None or trades_upload is None:
            st.info("Upload both CSV files to continue, or switch to Local files in the sidebar.")
            st.stop()

        fg_df = pd.read_csv(fg_upload)
        trades_df = pd.read_csv(trades_upload)
        st.sidebar.success("Using uploaded CSV files")
        return preprocess_data(fg_df, trades_df)

    fg_df, trades_df = load_local_data(base_dir)
    st.sidebar.success("Using local files from csv_files/")
    return preprocess_data(fg_df, trades_df)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    sentiments = sorted(df["classification"].dropna().unique().tolist())
    selected_sentiments = st.sidebar.multiselect(
        "Sentiments", sentiments, default=sentiments
    )

    coins = sorted(df["Coin"].dropna().astype(str).unique().tolist())
    selected_coins = st.sidebar.multiselect("Coins", coins, default=coins)

    filtered = df[
        (df["date"].dt.date >= start_date)
        & (df["date"].dt.date <= end_date)
        & (df["classification"].isin(selected_sentiments))
        & (df["Coin"].astype(str).isin(selected_coins))
    ].copy()

    st.sidebar.caption(f"Rows after filtering: {len(filtered):,}")
    return filtered


def render_overview(filtered_df: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)

    closed = filtered_df[filtered_df["Closed PnL"].fillna(0) != 0]
    win_rate = ((closed["Closed PnL"] > 0).mean() * 100) if len(closed) else np.nan

    c1.metric("Total Trades", f"{len(filtered_df):,}")
    c2.metric("Unique Traders", f"{filtered_df['Account'].nunique():,}")
    c3.metric("Total Volume (USD)", f"{filtered_df['Size USD'].sum():,.2f}")
    c4.metric("Win Rate (%)", "-" if np.isnan(win_rate) else f"{win_rate:.2f}")

    st.dataframe(filtered_df.head(20), use_container_width=True)


def render_sentiment_views(fg_df: pd.DataFrame, filtered_df: pd.DataFrame) -> None:
    st.subheader("Sentiment and Trading Activity")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sentiment_counts = (
        filtered_df["classification"].value_counts().reindex(SENTIMENT_ORDER).dropna()
    )
    axes[0].pie(
        sentiment_counts.values,
        labels=sentiment_counts.index,
        autopct="%1.1f%%",
        colors=[COLORS_MAP[s] for s in sentiment_counts.index],
    )
    axes[0].set_title("Trade Distribution by Sentiment")

    fg_plot = fg_df.sort_values("date")
    axes[1].plot(fg_plot["date"], fg_plot["value"], linewidth=1.5, color="steelblue")
    axes[1].axhline(y=50, color="black", linestyle="--", alpha=0.5)
    axes[1].axhline(y=25, color="red", linestyle="--", alpha=0.3)
    axes[1].axhline(y=75, color="green", linestyle="--", alpha=0.3)
    axes[1].set_title("Fear & Greed Index Over Time")
    axes[1].set_xlabel("Date")
    axes[1].set_ylabel("Index Value")

    plt.tight_layout()
    st.pyplot(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6))

    side_counts = filtered_df["Side"].value_counts()
    axes2[0].bar(side_counts.index, side_counts.values, color=["green", "red"])
    axes2[0].set_title("BUY vs SELL Count")

    log_size = np.log10(filtered_df["Size USD"].fillna(0) + 1)
    axes2[1].hist(log_size, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes2[1].set_title("Trade Size Distribution (Log10 USD)")
    axes2[1].set_xlabel("log10(Size USD + 1)")

    plt.tight_layout()
    st.pyplot(fig2)


def render_behavior_profitability(filtered_df: pd.DataFrame) -> None:
    st.subheader("Behavior and Profitability")

    buy_sell = filtered_df.groupby(["classification", "Side"]).size().unstack(fill_value=0)
    buy_sell_pct = buy_sell.div(buy_sell.sum(axis=1), axis=0).mul(100)
    buy_sell_pct = buy_sell_pct.reindex(SENTIMENT_ORDER).dropna(how="all")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    buy_sell_pct.plot(kind="bar", stacked=True, ax=axes[0], color=["green", "red"])
    axes[0].set_title("Buy vs Sell by Sentiment (%)")
    axes[0].set_ylabel("Percent")
    axes[0].tick_params(axis="x", rotation=45)

    if {"BUY", "SELL"}.issubset(buy_sell.columns):
        ratio = (buy_sell["BUY"] / buy_sell["SELL"].replace(0, np.nan)).reindex(SENTIMENT_ORDER)
        valid_idx = [s for s in ratio.index if pd.notna(ratio.loc[s])]
        axes[1].bar(
            valid_idx,
            ratio.dropna().values,
            color=[COLORS_MAP[s] for s in valid_idx],
        )
        axes[1].axhline(y=1, color="black", linestyle="--")
    axes[1].set_title("Buy/Sell Ratio by Sentiment")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    closed = filtered_df[filtered_df["Closed PnL"].fillna(0) != 0].copy()
    if closed.empty:
        st.info("No closed positions in current filter for profitability analysis.")
        return

    closed["Profitable"] = closed["Closed PnL"] > 0
    profitability = closed.groupby("classification").agg(
        Total_PnL=("Closed PnL", "sum"),
        Avg_PnL=("Closed PnL", "mean"),
        Median_PnL=("Closed PnL", "median"),
        Std_PnL=("Closed PnL", "std"),
        Profitable_Trades=("Profitable", "sum"),
        Total_Closed_Trades=("Profitable", "count"),
    )
    profitability["Win_Rate_%"] = (
        profitability["Profitable_Trades"] / profitability["Total_Closed_Trades"] * 100
    )

    st.dataframe(profitability.round(2), use_container_width=True)

    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

    wr = profitability["Win_Rate_%"].reindex(SENTIMENT_ORDER)
    wr_idx = [s for s in wr.index if pd.notna(wr.loc[s])]
    axes2[0, 0].bar(wr_idx, wr.dropna().values, color=[COLORS_MAP[s] for s in wr_idx])
    axes2[0, 0].axhline(50, color="black", linestyle="--")
    axes2[0, 0].set_title("Win Rate by Sentiment")
    axes2[0, 0].tick_params(axis="x", rotation=45)

    total_pnl = profitability["Total_PnL"].reindex(SENTIMENT_ORDER)
    pnl_idx = [s for s in total_pnl.index if pd.notna(total_pnl.loc[s])]
    axes2[0, 1].bar(
        pnl_idx, total_pnl.dropna().values, color=[COLORS_MAP[s] for s in pnl_idx]
    )
    axes2[0, 1].axhline(0, color="black", linestyle="--")
    axes2[0, 1].set_title("Total PnL by Sentiment")
    axes2[0, 1].tick_params(axis="x", rotation=45)

    avg_pnl = profitability["Avg_PnL"].reindex(SENTIMENT_ORDER)
    std_pnl = profitability["Std_PnL"].reindex(SENTIMENT_ORDER)
    avg_idx = [s for s in avg_pnl.index if pd.notna(avg_pnl.loc[s])]
    axes2[1, 0].bar(
        avg_idx,
        avg_pnl.dropna().values,
        yerr=std_pnl.dropna().values,
        capsize=5,
        color=[COLORS_MAP[s] for s in avg_idx],
        alpha=0.75,
    )
    axes2[1, 0].axhline(0, color="black", linestyle="--")
    axes2[1, 0].set_title("Average PnL by Sentiment")
    axes2[1, 0].tick_params(axis="x", rotation=45)

    filtered_pnl = closed[closed["Closed PnL"].between(-1000, 1000)]
    sns.boxplot(
        data=filtered_pnl,
        x="classification",
        y="Closed PnL",
        order=SENTIMENT_ORDER,
        palette=COLORS_MAP,
        ax=axes2[1, 1],
    )
    axes2[1, 1].axhline(0, color="black", linestyle="--")
    axes2[1, 1].set_title("PnL Distribution by Sentiment")
    axes2[1, 1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig2)


def render_risk_and_time_series(filtered_df: pd.DataFrame) -> None:
    st.subheader("Risk, Correlation, and Time Series")

    risk_stats = (
        filtered_df.groupby("classification")["Size USD"].describe().reindex(SENTIMENT_ORDER)
    )
    st.dataframe(risk_stats.round(2), use_container_width=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    median_sizes = risk_stats["50%"].dropna()
    axes[0].bar(
        median_sizes.index,
        median_sizes.values,
        color=[COLORS_MAP[s] for s in median_sizes.index],
    )
    axes[0].set_title("Median Trade Size by Sentiment")
    axes[0].tick_params(axis="x", rotation=45)

    violin_df = filtered_df[filtered_df["Size USD"].between(0, 5000)]
    sns.violinplot(
        data=violin_df,
        x="classification",
        y="Size USD",
        order=SENTIMENT_ORDER,
        palette=COLORS_MAP,
        ax=axes[1],
    )
    axes[1].set_title("Trade Size Distribution (0-5000 USD)")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    corr_cols = ["sentiment_numeric", "value", "Size USD", "Closed PnL"]
    corr = filtered_df[corr_cols].corr()

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, square=True, ax=ax2)
    ax2.set_title("Correlation Heatmap")
    st.pyplot(fig2)

    daily = (
        filtered_df.groupby("date")
        .agg(
            total_volume=("Size USD", "sum"),
            trade_count=("Trade ID", "count"),
            fg_value=("value", "first"),
            total_pnl=("Closed PnL", "sum"),
        )
        .reset_index()
        .sort_values("date")
    )

    fig3, ax1 = plt.subplots(figsize=(16, 6))
    ax1.bar(daily["date"], daily["total_volume"], alpha=0.35, color="steelblue", label="Volume")
    ax1.set_ylabel("Trading Volume (USD)", color="steelblue")
    ax2 = ax1.twinx()
    ax2.plot(daily["date"], daily["fg_value"], color="darkred", linewidth=2, label="F&G")
    ax2.set_ylabel("Fear & Greed Index", color="darkred")
    ax2.axhline(50, color="black", linestyle="--", alpha=0.5)
    ax1.set_title("Trading Volume vs Fear & Greed")
    st.pyplot(fig3)

    window = st.slider("Rolling correlation window (days)", 3, 30, 7, 1)
    rolling_corr = daily["total_volume"].rolling(window).corr(daily["fg_value"])

    fig4, ax4 = plt.subplots(figsize=(16, 5))
    ax4.plot(daily["date"], rolling_corr, color="purple", linewidth=2)
    ax4.axhline(0, color="black", linestyle="--")
    ax4.set_title(f"{window}-Day Rolling Correlation: Volume vs Fear & Greed")
    st.pyplot(fig4)

    st.caption(f"Average rolling correlation: {rolling_corr.mean():.3f}")


def render_trader_insights(filtered_df: pd.DataFrame) -> None:
    st.subheader("Trader Activity and Top Performers")

    trader_activity = filtered_df.groupby("classification").agg(
        Unique_Traders=("Account", "nunique"),
        Total_Trades=("Trade ID", "count"),
    )
    trader_activity["Avg_Trades_per_Trader"] = (
        trader_activity["Total_Trades"] / trader_activity["Unique_Traders"]
    )

    st.dataframe(trader_activity.reindex(SENTIMENT_ORDER).round(2), use_container_width=True)

    closed = filtered_df[filtered_df["Closed PnL"].fillna(0) != 0].copy()
    if closed.empty:
        st.info("No closed positions in current filter for top trader analysis.")
        return

    top_n = st.slider("Top traders per sentiment", 3, 20, 5, 1)

    perf = (
        closed.groupby(["Account", "classification"]).agg(
            Total_PnL=("Closed PnL", "sum"), Trade_Count=("Trade ID", "count")
        )
        .reset_index()
        .rename(columns={"classification": "Sentiment"})
    )

    selected_sentiment = st.selectbox(
        "Inspect sentiment",
        [s for s in SENTIMENT_ORDER if s in perf["Sentiment"].unique()],
    )
    top = perf[perf["Sentiment"] == selected_sentiment].nlargest(top_n, "Total_PnL")
    st.dataframe(top, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["Account"].astype(str), top["Total_PnL"], color=COLORS_MAP[selected_sentiment])
    ax.set_title(f"Top {top_n} Traders in {selected_sentiment} by Total PnL")
    ax.set_xlabel("Total PnL (USD)")
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(
        page_title="Trading Behavior vs Market Sentiment",
        page_icon="📈",
        layout="wide",
    )
    st.title("Trading Behavior vs Market Sentiment")
    st.caption(
        "Interactive Streamlit version of the notebook analysis for Hyperliquid trades and Fear & Greed Index."
    )

    base_dir = "."
    try:
        fg_df, merged_df = load_data_with_option(base_dir)
    except FileNotFoundError as exc:
        st.error(f"Data file not found: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Failed to load data: {exc}")
        st.stop()

    filtered_df = apply_filters(merged_df)

    if filtered_df.empty:
        st.warning("No rows match current filters. Adjust selections in the sidebar.")
        st.stop()

    tabs = st.tabs(
        [
            "Overview",
            "Sentiment & Activity",
            "Behavior & Profitability",
            "Risk & Time Series",
            "Traders",
        ]
    )

    with tabs[0]:
        render_overview(filtered_df)

    with tabs[1]:
        render_sentiment_views(fg_df, filtered_df)

    with tabs[2]:
        render_behavior_profitability(filtered_df)

    with tabs[3]:
        render_risk_and_time_series(filtered_df)

    with tabs[4]:
        render_trader_insights(filtered_df)


if __name__ == "__main__":
    main()
