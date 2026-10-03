# NIFTY 50 Stock Co-movement Analysis (PCA & Clustering)

**Question:** Do NIFTY 50 stocks move with their official sector, or are there hidden groups an investor should know about for diversification?

**Short answer:** Mostly with their sector, with some exceptions. IT, private banks, pharma and FMCG form clean, single-sector clusters. Several others don't: insurers move with new-age names rather than with banks, hospital chains move separately from pharma, PSU power and energy trade like metals, and Reliance and Adani Enterprises move with the domestic cyclicals.

## Data
| | |
|---|---|
| Universe | NSE's official NIFTY 50 list (`data/nifty50_raw.csv`, downloaded from niftyindices.com) → `data/nifty50_sectors.csv` (ticker, sector) |
| Prices | Daily adjusted closes, 1 Oct 2021 – 30 Sep 2026, via `yfinance` |
| Sample | 49 stocks × 1,239 trading days of returns |
| Dropped | **JIOFIN** (38% of days missing; it listed in Aug 2023) |
| Data fixes | Two one-day returns were set to 0. **TMPV** −40% on 14 Oct 2025 was the Tata Motors demerger, not a market move. **TRENT** −33% on 1 Jan 2026 is a Yahoo price-scaling error: news put Trent near ₹4,060 on 6 Jan, while Yahoo shows ₹2,700. Genuine shocks, such as Adani after the Hindenburg report (−28%), are kept. |

## Key results

**PCA**
- **PC1 explains 29.0% of variance.** All 49 loadings are positive, so PC1 is the market factor: the whole index moving together. It took 25 components to reach 80%, so after the market factor most variance is stock-specific.
  - Most market-sensitive: Grasim, L&T, JSW Steel, Bajaj Finserv, SBI.
  - Least market-sensitive: Cipla, Max Healthcare, Dr Reddy's, ONGC, HUL. These are defensives.
- **PC2 (5.6%) splits defensives from cyclicals.** IT (TCS, Infosys, HCL, TechM) plus HUL and Asian Paints sit at **+**. PSU energy and power (ONGC, Coal India, NTPC, Power Grid), BEL and Tata Steel sit at **−**. This matches the expected defensive vs cyclical split.
- **PC3 (5.0%) splits export IT from domestic consumption.** IT (plus ONGC and Hindalco, which are dollar-linked) sits at **+**. Maruti, HUL, Asian Paints, Eicher, Nestlé and Bajaj Finance sit at **−**. This is effectively a rupee/global-vs-India factor.

**Clustering** (Ward linkage on `sqrt(2(1−ρ))`, cut at **k = 7**)

| # | Cluster | Members |
|---|---|---|
| 1 | IT services | TCS, INFY, HCLTECH, TECHM |
| 2 | Commodities & PSU energy | HINDALCO, JSWSTEEL, TATASTEEL, COALINDIA, ONGC, NTPC, POWERGRID |
| 3 | Core banks & lenders | HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK, SBIN, BAJFINANCE, BAJAJFINSV |
| 4 | New-age / insurance / hospitals | HDFCLIFE, SBILIFE, APOLLOHOSP, MAXHEALTH, BSE, ETERNAL |
| 5 | Domestic cyclicals | Autos (MARUTI, M&M, BAJAJ-AUTO, EICHERMOT, TMPV), cement (GRASIM, ULTRACEMCO), LT, BEL, ADANIENT, ADANIPORTS, RELIANCE, BHARTIARTL, TITAN, TRENT, ASIANPAINT, INDIGO, SHRIRAMFIN |
| 6 | Pharma | SUNPHARMA, CIPLA, DRREDDY |
| 7 | FMCG staples | HINDUNILVR, ITC, NESTLEIND, TATACONSUM |

**Hidden co-movers: 8 stocks** sit outside the cluster where most of their own sector lives:

| Stock | Official sector | Moves with |
|---|---|---|
| HDFCLIFE, SBILIFE | Financial Services | New-age/insurance/hospitals, not banks |
| BSE | Financial Services | New-age/insurance/hospitals |
| SHRIRAMFIN | Financial Services | Domestic cyclicals (vehicle finance follows auto demand) |
| APOLLOHOSP, MAXHEALTH | Healthcare | New-age/insurance/hospitals, not pharma |
| RELIANCE | Oil, Gas & Fuels | Domestic cyclicals, not PSU energy |
| ADANIENT | Metals & Mining | Domestic cyclicals, paired tightly with ADANIPORTS |

The two Consumer Services stocks (ETERNAL, TRENT) also split 1–1. At the sector level, **Power and PSU oil and gas trade like metals**: all four of their stocks sit in the commodities cluster.

**Robustness:** After removing PC1, re-clustering the market-neutral residuals gives a similar partition (ARI = 0.61 against the raw clusters). Insurers and hospitals still separate from banks, and the Adani names join the PSU energy group. Agreement with official sectors is only moderate either way (ARI 0.27 raw, 0.33 market-neutral). So sector labels are a useful starting point but leave out real structure.

**Resume bullet**
> Analysed 5 years of daily returns for the NIFTY 50 using PCA and Ward hierarchical clustering. The market factor (PC1) explains 29% of variance. PC2 and PC3 separate defensives from cyclicals and export IT from domestic consumption. Found 8 "hidden co-movers" that trade with a different group than their NSE sector, including HDFC Life, SBI Life, Apollo Hospitals and Reliance.

## Charts
| | |
|---|---|
| ![](charts/1_scree.png) | ![](charts/2_biplot_pc1_pc2.png) |
| ![](charts/3_corr_heatmap_by_cluster.png) | ![](charts/4_dendrogram.png) |

## Files
- `nifty50_comovement.ipynb`: the full analysis, executed with outputs. **Start here.**
- `build_notebook.py`: generates the notebook. Edit the cells here, then rebuild.
- `results.json`: every headline number. `data/loadings_and_clusters.csv` holds PC1–3 loadings plus the cluster for each stock.

### Re-run
```bash
pip install -r requirements.txt
```
```bash
python build_notebook.py && jupyter nbconvert --to notebook --execute --inplace nifty50_comovement.ipynb
```
Prices are re-downloaded live, so the numbers can shift slightly if Yahoo revises its data.

## Interview prep

**Why standardise before PCA?** PCA maximises variance. Without scaling, high-volatility stocks such as Adani Enterprises or BSE would dominate the first components just because their returns are bigger, not because they drive co-movement. Scaling each stock to unit variance means PCA runs on the **correlation** matrix, so every stock gets an equal vote.

**What does PC1 represent?** The market factor. All 49 loadings share one sign, so a positive PC1 score means "the whole market is up today". It is basically NIFTY's beta. Loading size measures market sensitivity: Grasim, L&T and SBI are high-beta, while Cipla, Max Healthcare and HUL are defensive. At 29%, it is a little below the 30–40% you often see. Over this window, stock-specific and sector events (Adani/Hindenburg, PSU rally, IT slowdown) took a large share of the variance.

**Why returns and not prices?** Prices are non-stationary: they trend. Any two stocks that both rose over five years will show a high price correlation even with no daily link, which is spurious regression. Returns are roughly stationary, so their correlation measures whether stocks actually move together day to day. That co-movement is what matters for diversification.

**How did you choose the number of clusters?** I combined three things:
1. The brief's 5–7 range.
2. The silhouette score. It is low and flat across that range (0.07–0.08), which tells me structure beyond the market factor is real but weak. That is normal for large caps in one market.
3. Interpretability on the dendrogram. At k = 5, a single 31-stock "everything domestic" cluster remains. k = 7 is the first cut that separates banks, the insurance/hospital/new-age group and domestic cyclicals.

I then checked stability by re-clustering market-neutral residuals (ARI 0.61).

**Why sqrt(2(1−ρ)) and Ward?** For standardised series, this distance equals the Euclidean distance between them, up to a constant. That makes it a proper metric, which Ward linkage needs because it minimises within-cluster variance. ρ = 1 gives distance 0, and ρ = −1 gives the maximum distance of 2.

**How would a portfolio manager use this?**
- **Real diversification:** owning HDFC Bank, ICICI and Kotak is one bet, not three. Holding HDFC Life alongside a bank adds more diversification than its "Financials" label suggests.
- **Risk budgets by cluster, not sector:** for example, cap total exposure to the commodities/PSU energy cluster, because NTPC and Power Grid behave like metals.
- **Hedging and pairs trades:** tight pairs such as Bajaj Finance/Finserv, JSW/Tata Steel and Adani Ent/Ports are natural candidates for pairs trades or relative-value positions.
- **Factor tilts:** PC2 and PC3 scores give simple defensive-vs-cyclical and export-vs-domestic tilts.

**Limitations to mention:** Correlations change over time. They spike in sell-offs, which is exactly when diversification matters. A rolling-window version, or a separate 2022 drawdown sub-period, would test whether the clusters hold up. Pearson correlation is also sensitive to outliers such as the Adani crash, and the cluster structure here is weak overall (silhouette about 0.08).
