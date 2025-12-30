# Trading Behavior vs Market Sentiment Analysis

A comprehensive data science project analyzing the relationship between Hyperliquid trader behavior and Bitcoin Fear & Greed Index to uncover profitable trading patterns and sentiment-driven insights.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-orange)
![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-red)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset Description](#dataset-description)
- [Installation](#installation)
- [Usage](#usage)
- [Analysis Sections](#analysis-sections)
- [Key Findings](#key-findings)
- [Visualizations](#visualizations)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

This project investigates how cryptocurrency traders on the Hyperliquid platform behave under different market sentiment conditions, as measured by the Bitcoin Fear & Greed Index. By analyzing thousands of trades across various sentiment states (Extreme Fear to Extreme Greed), we identify:

- **Behavioral patterns** that correlate with market sentiment
- **Profitability differences** across sentiment categories
- **Risk appetite changes** during fear vs greed periods
- **Contrarian vs conformist strategies** and their effectiveness
- **Optimal trading strategies** based on historical data

The analysis provides actionable insights for traders looking to incorporate sentiment analysis into their trading strategies.

---

## ✨ Features

### Comprehensive Analysis
- **Multi-dimensional Data Analysis**: Combines trading data with market sentiment metrics
- **Statistical Profiling**: Detailed statistical analysis of trading patterns by sentiment
- **Behavioral Classification**: Identifies contrarian and conformist trading behaviors
- **Performance Metrics**: Win rates, PnL analysis, and risk-adjusted returns

### Advanced Visualizations
- **Interactive Charts**: 20+ high-quality visualizations using Matplotlib and Seaborn
- **Correlation Heatmaps**: Identify relationships between sentiment and trading metrics
- **Time Series Analysis**: Track sentiment and trading volume trends over time
- **Distribution Analysis**: Box plots, violin plots, and histograms for pattern recognition

### Strategic Insights
- **Sentiment-Based Trading Rules**: Data-driven recommendations for different market conditions
- **Risk Management Guidelines**: Position sizing strategies based on sentiment
- **Top Performer Analysis**: Identify successful traders and their strategies
- **Temporal Pattern Recognition**: Discover time-based trading opportunities

---

## 📊 Dataset Description

### 1. Fear & Greed Index Dataset (`fear_greed_index.csv`)

Contains daily Bitcoin Fear & Greed Index values and classifications.

**Columns:**
- `date`: Date of the measurement (datetime)
- `value`: Fear & Greed Index value (0-100)
  - 0-24: Extreme Fear
  - 25-49: Fear
  - 50: Neutral
  - 51-75: Greed
  - 76-100: Extreme Greed
- `classification`: Categorical sentiment classification

**Source**: Bitcoin Fear & Greed Index historical data

### 2. Historical Trading Data (`historical_data.csv`)

Contains detailed Hyperliquid trading records with timestamps, positions, and P&L.

**Columns:**
- `Trade ID`: Unique identifier for each trade
- `Account`: Trader account identifier (anonymized)
- `Timestamp IST`: Trade execution timestamp in IST
- `Coin`: Cryptocurrency symbol traded
- `Side`: Trade direction (BUY/SELL)
- `Size USD`: Position size in USD
- `Closed PnL`: Profit/Loss for closed positions
- Additional trading metrics

**Coverage**: Multiple days of trading activity across various sentiment conditions

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab (for notebook workflow)
- Streamlit (for interactive app workflow)
- pip package manager

### Setup Instructions

1. **Clone or Download the Project**
   ```bash
   cd ds_Nikita_Patra
   ```

2. **Install Required Dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn streamlit
   ```

   Or install from requirements file:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify Data Files**
   Ensure the following files are in the project directory:
   - `fear_greed_index.csv`
   - `historical_data.csv`

4. **Launch Jupyter Notebook (Optional)**
   ```bash
   jupyter notebook notebook_1.ipynb
   ```

5. **Launch Streamlit App**
   ```bash
   streamlit run app.py
   ```

---

## 🚀 Usage

### Running the Analysis

1. **Open the Notebook**: Launch `notebook_1.ipynb` in Jupyter
2. **Run All Cells**: Execute cells sequentially or use "Run All" from the menu
3. **Review Outputs**: Each section provides visualizations and statistical summaries
4. **Customize Analysis**: Modify parameters to focus on specific aspects

### Running the Streamlit App

1. Install dependencies with `pip install -r requirements.txt`
2. Start app with `streamlit run app.py`
3. Open the local URL shown in terminal (typically `http://localhost:8501`)
4. Use sidebar filters to analyze date range, sentiment, and coin-specific behavior

### Quick Start Example

```python
# Load the datasets
import pandas as pd

fg_df = pd.read_csv('fear_greed_index.csv')
trades_df = pd.read_csv('historical_data.csv')

# Basic exploration
print(f"Fear & Greed data points: {len(fg_df)}")
print(f"Total trades: {len(trades_df)}")
print(f"Sentiment distribution:\n{fg_df['classification'].value_counts()}")
```

### Customization Options

- **Date Range Filtering**: Analyze specific time periods
- **Sentiment Focus**: Deep dive into particular sentiment categories
- **Trader Segmentation**: Focus on top performers or specific cohorts
- **Coin-Specific Analysis**: Examine individual cryptocurrency behavior

---

## 📈 Analysis Sections

The notebook is organized into comprehensive sections:

### 1. Data Loading & Preprocessing
- Import and clean datasets
- Handle missing values and data types
- Merge trading data with sentiment data
- Create derived features

### 2. Exploratory Data Analysis (EDA)
- **Sentiment Distribution**: Pie charts and time series of Fear & Greed Index
- **Trading Activity**: Volume, frequency, and trader participation
- **PnL Distribution**: Analyze profitability patterns
- **Trade Characteristics**: Side distribution, size analysis

### 3. Trading Behavior by Sentiment
- **Volume Analysis**: Trading activity by sentiment state
- **Buy/Sell Patterns**: Ratio analysis across sentiments
- **Trade Size Behavior**: Risk appetite variations

### 4. Profitability Analysis
- **Win Rate Metrics**: Success rates by sentiment
- **Average PnL**: Mean and median profitability
- **Total Returns**: Aggregate P&L by sentiment
- **Distribution Analysis**: Box plots and violin plots

### 5. Risk Analysis
- **Position Sizing**: Trade size statistics by sentiment
- **Volatility Patterns**: Standard deviation and dispersion metrics
- **Risk-Adjusted Returns**: Sharpe-like ratios

### 6. Advanced Insights
- **Correlation Analysis**: Heatmaps showing relationships
- **Time Series Patterns**: Rolling correlations and trends
- **Contrarian vs Conformist**: Strategy comparison
- **Top Performer Identification**: Best traders by sentiment

### 7. Strategic Recommendations
- **Entry/Exit Timing**: Optimal conditions for trades
- **Position Sizing Rules**: Sentiment-based guidelines
- **Risk Management**: Stop-loss and take-profit strategies
- **Behavior Optimization**: Align strategies with profitable patterns

---

## 🔍 Key Findings

### Trading Activity Patterns
- Trading volume and frequency vary significantly across sentiment states
- Extreme sentiment periods (Extreme Fear/Greed) show distinct trading patterns
- Unique trader participation fluctuates with market sentiment

### Profitability Insights
- **Win rates differ by sentiment**: Certain sentiments historically show higher success rates
- **Risk-reward profiles vary**: Average PnL and volatility change with sentiment
- **Sentiment extremes offer opportunities**: Both extreme fear and greed can be profitable with right strategy

### Behavioral Analysis
- **Contrarian Strategy**: Buying during fear and selling during greed
- **Conformist Strategy**: Buying during greed and selling during fear
- Performance comparison reveals effectiveness of each approach
- Mixed strategies may be optimal depending on sentiment intensity

### Risk Management Observations
- Position sizes tend to vary with sentiment confidence
- Traders adjust risk exposure based on market conditions
- Successful traders show consistent risk management across sentiments

### Temporal Patterns
- Rolling correlation between volume and sentiment reveals lead-lag relationships
- Certain time periods show stronger sentiment-behavior alignment
- Sentiment transitions may present unique opportunities

---

## 📊 Visualizations

The analysis generates 20+ professional visualizations including:

### Distribution Plots
- Sentiment classification pie charts
- Fear & Greed Index time series with threshold markers
- Trade size histograms (log scale)
- PnL distribution plots

### Comparative Analysis
- Buy/Sell ratio bar charts by sentiment
- Win rate comparisons across sentiment categories
- Average PnL with error bars (standard deviation)
- Box plots for PnL distribution

### Risk Visualizations
- Median trade size by sentiment
- Violin plots for trade size distribution
- Position sizing patterns

### Advanced Charts
- Correlation heatmaps (sentiment vs trading metrics)
- Dual-axis plots (volume vs sentiment over time)
### 4. Profitability Analysis
- **Win Rate Metrics**: Success rates by sentiment
- **Average PnL**: Mean and median profitability
- **Total Returns**: Aggregate P&L by sentiment
- **Distribution Analysis**: Box plots and violin plots

---

## 🎯 Risk & Drawdown Analysis

### 5. Risk & Drawdown
- **Drawdown Analysis**: Peak-to-trough drawdowns by sentiment
- **Risk-Adjusted Metrics**: Sharpe/Sortino-like comparisons
- **Volatility Profiling**: Trade size variability by sentiment
- **Tail Risk**: Extreme loss analysis

---

## 🤝 Trader Behavior Profiles

### 6. Trader Segmentation
- **Top Performers**: Identify traders with consistent wins
- **Contrarian vs Conformist**: Behavior classification by sentiment trend
- **Position Sizing**: Risk appetite indicators
- **Temporal Patterns**: Time-of-day and day-of-week effects

---

## 🧭 Strategy Insights

### 7. Strategy Recommendations
- **Sentiment-Based Playbooks**: What to do in Fear vs Greed
- **Risk Controls**: Stop-loss, take-profit guidance per sentiment
- **Sizing Rules**: Position sizing heuristics by sentiment
- **Signal Combinations**: Pairing sentiment with volume or volatility

---

## 🖼️ Visualizations

- Volume vs Fear & Greed Index
- Buy/Sell ratios by sentiment
- PnL distributions across sentiment states
- Heatmaps of correlations between sentiment and trading metrics
- Rolling correlations of trading volume vs sentiment
- Trader activity segmentation plots

---

## 📦 Project Structure

```bash
.
├── README.md
├── notebook_1.ipynb          # Main analysis notebook
├── csv_files/
│   ├── fear_greed_index.csv  # Sentiment dataset
│   └── historical_data.csv   # Trading records
└── outputs/                  # Generated plots and figures
```

---

## 📦 Dependencies

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn

---

## 🔮 Future Enhancements

- Add trading strategy backtests
- Include more exchanges and asset pairs
- Integrate live sentiment feeds
- Expand statistical testing for significance

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a PR with a clear description of changes.

---

## 🛡️ License

This project is licensed under the MIT License.
---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### Ways to Contribute
1. **Bug Reports**: Open an issue describing the bug
2. **Feature Requests**: Suggest new analysis ideas or visualizations
3. **Code Improvements**: Submit pull requests with enhancements
4. **Documentation**: Improve README or add code comments
5. **Data Contributions**: Share additional datasets for analysis

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is available for educational and research purposes. Please ensure compliance with data usage policies of Hyperliquid and Fear & Greed Index sources.

**Disclaimer**: This analysis is for educational and research purposes only. Past performance does not guarantee future results. Always conduct thorough due diligence and risk assessment before making trading decisions. The authors are not responsible for any financial losses incurred from using this analysis.

---

## 📧 Contact & Support

For questions, suggestions, or collaboration opportunities:
- Open an issue in the repository
- Reach out via email (if applicable)
- Connect on professional networks

---

## 🙏 Acknowledgments

- **Bitcoin Fear & Greed Index**: For providing historical sentiment data
- **Hyperliquid Platform**: For trading data access
- **Open Source Community**: For the excellent Python data science ecosystem
- **Contributors**: Thanks to everyone who has contributed to this project

---

## 📚 References & Resources

### Academic Papers
- Behavioral Finance and Cryptocurrency Markets
- Sentiment Analysis in Financial Trading
- Contrarian Investment Strategies

### Tools & Libraries
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
- [Seaborn Tutorial](https://seaborn.pydata.org/tutorial.html)

### Data Sources
- [Alternative.me Fear & Greed Index](https://alternative.me/crypto/fear-and-greed-index/)
- [Hyperliquid Documentation](https://hyperliquid.xyz/)

---

<div align="center">

**⭐ If you find this project useful, please consider giving it a star! ⭐**

Made with ❤️ and Python 🐍

</div>
>>>>>>> 8530080 (Initial commit)
