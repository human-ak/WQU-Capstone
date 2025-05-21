# **Clustering with Convolutional AutoEncoders for Pairs Selection in a Pairs Trading framework**

(MScFE 690 CP Track 8 - Machine Learning (Deep) Investment Strategies:- Topic 3 - Machine Learning (Deep) Application to Long/Short Pair Trading Strategies)

## Problem Statement

Our project consists of improving the two-stage Pairs Trading framework as demonstrated in the book “A Machine Learning-Based Pairs Trading Investment Strategy” (Sarmento and Horta). 

In the first stage of pairs selection (21–35), dimensionality reduction and clustering techniques were applied to identify co-moving assets. We propose to evaluate how the deep-learning  technique of Convolutional Auto-Encoders (CAE) could improve the clustering of the asset universe. 

Similarly to the authors , when eligible pairs are identified, we will apply the set of 4 rules as defined by the authors, i.e., cointegration, Hurst metric , half-life, and average mean reversions (31–33). 

Then we build train forecasting models (MLP & LSTM) to do the pair trading.

We plan to implement some of the suggestions mentioned in the book (104), by building a cross-market asset universe, focusing on technology-related topics, whether for ETFs, stocks, or commodities. In our 1st draft submitted on 30-04-25, we used the 44 ETFs that have volume above 1 million from Yahoo Finance.

## Results

The following are our results for the 1st draft. We are also trying out other forecasting models for trading.

**CAE**

**Validation**

| | MLP  | LSTM |
| ------------- | ------------- | ------------- |
| Portfolio sharpe ratio | 0.24 | 3.58 |
| Maximum drawdown of portfolio  | -0.64% | -0.29% |
| Total Drawdown Days | 140 days | 110 days |
| Max DD period | 0 days | 0 days |
| Total number of trades | 5183 | 1221 |
| Positive trades | 2590 | 752 |
| Negative trades | 2593 | 469 |
| Annual ROI | 0.18 | 2.15 |
 
**Test unrestricted**

| | MLP  | LSTM |
| ------------- | ------------- | ------------- |
| Portfolio sharpe ratio | -1.06 | 0.41 |
| Maximum drawdown of portfolio  | -3.85% | -2.56% |
| Total Drawdown Days | 130 days | 122 days |
| Max DD period | 3 days | 1 days |
| Total number of trades | 10012 | 3128 |
| Positive trades | 4558 | 1830 |
| Negative trades | 5454 | 1298 |
| Annual ROI | -3.31 | 1.43 |

**Test with active pairs on validation set**

| | MLP  | LSTM |
| ------------- | ------------- | ------------- |
| Portfolio sharpe ratio | -0.51 | 0.40 |
| Maximum drawdown of portfolio  | -4.08% | -3.19% |
| Total Drawdown Days | 129 days | 109 days |
| Max DD period | 1 days | 1 days |
| Total number of trades | 4358 | 1150 |
| Positive trades | 2006 | 630 |
| Negative trades | 2352 | 520 |
| Annual ROI | -1.79 | 1.45 |


## Litterature Review

For an exhaustive review of the literature, please refer to the project's Zotero library: [https://bit.ly/WQUzotero](https://bit.ly/WQUzotero).

### We have also identified the following resources as potentially reusable or improvable:

Daehkiml. Pair Trading: A Market-Neutral Trading Strategy with Integrated Machine Learning. https://daehkim.github.io/pair-trading/. Accessed 30 Mar. 2025.

Jansen, Stefan. Stefan-Jansen/Machine-Learning-for-Trading. 2018. 30 Mar. 2025. GitHub, https://github.com/stefan-jansen/machine-learning-for-trading.

Polakow, Oleg. VectorBT. https://vectorbt.dev/.

Roychoudhury, Raktim. Pairs Trading Using Unsupervised Clustering and Deep Reinforcement Learning. 2023. 8 Apr. 2025. GitHub, https://github.com/raktim-roychoudhury/pairs_trading.

tensortrade.org. TensorTrade. https://github.com/tensortrade-org. Accessed 14 Apr. 2025.

Yan, Zijian. Yan1015/Pairs-Trading-Using-Copula. 2018. 9 Mar. 2025. GitHub, https://github.com/Yan1015/Pairs-Trading-using-Copula.

Yang, Hongshen. Cryptocurrency Trading with Reinforcement Learning Based on Backtrader. 2023. 30 Mar. 2025. GitHub, https://github.com/Hongshen-Yang/pair-trading-envs.*

