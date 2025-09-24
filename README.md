# clustering-based-relative-valuation

## A Modern Approach to Equities Investing: Clustering-Based Relative Valuation
This repository contains the code, data workflows, and results from my Bachelor Thesis, where I explored how unsupervised machine learning can improve equity valuation and trading strategies.

### Project Overview
Traditional equity valuation often relies on peer group comparisons, which are usually selected manually. This process is:
* Time-consuming
* Biased (analysts may select peers that support their thesis)
This project automates peer group selection using clustering algorithms (K-means and DBSCAN) to identify comparable firms objectively. It then tests whether these machine learning–derived peer groups can improve trading strategy performance.

### Methodology
The project follows the CRISP-DM data science framework:
1. Business Understanding – Problem: peer group selection in equity valuation is biased and inefficient
2. Data Understanding – Stock data (2017–2022) from CapitalIQ & Compustat, split into value vs. growth universes
3. Data Preparation – Cleaning, outlier reduction, scaling, feature engineering
4. Modeling – Applied:
* K-means clustering (partitioning)
* DBSCAN (density-based, better with outliers)
5. Evaluation – Compared models on:
* Cluster quality (silhouette scores, size, runtime)
* Trading performance (returns, Sharpe ratios, volatility)
6. Deployment – Trading strategy simulation with implied share prices and buy/sell signals

### Key Findings
* DBSCAN: Better at handling outliers and produced higher trading returns, sometimes beating benchmarks
* K-means: Formed tighter, more cohesive clusters but was more sensitive to outliers
* Value vs. Growth Stocks: Contrary to conventional finance theory, growth stocks outperformed value stocks in this dataset
* Feature Selection: Domain knowledge–driven features (e.g., profitability, growth, risk) worked better than fully automated selection

## Repository Structure
├── 01 data/             # Processed financial data (2017–2022)
├── 02 notebooks/        # Jupyter notebooks with clustering & trading strategy
├── 03 src/              # Core Python scripts (preprocessing, modeling, evaluation)
├── 04 results/          # Plots, cluster stats, trading performance outputs
└── README.md         # Project documentation (this file)

## Technologies Used
* Python (pandas, numpy, scikit-learn, matplotlib, seaborn)
* Kneedle (to determine optimal cluster count for K-means)
* WRDS/Compustat/CapitalIQ (financial data sources)

## Citation
If you use this code or methodology, please cite:
Daniel Eder (2024). A Modern Approach to Equities Investing: Clustering-Based Relative Valuation. Bachelor Thesis, WU Vienna.
