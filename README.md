# **Clustering with Convolutional AutoEncoders for Pairs Selection in a Pairs Trading framework**

(MScFE 690 CP Track 8 - Machine Learning (Deep) Investment Strategies:- Topic 3 - Machine Learning (Deep) Application to Long/Short Pair Trading Strategies)

## Problem Statement

Our project consists of improving the two-stage Pairs Trading framework as demonstrated in the book “A Machine Learning-Based Pairs Trading Investment Strategy” (Sarmento and Horta). 

In the first stage of pairs selection (21–35), dimensionality reduction and clustering techniques were applied to identify co-moving assets. We propose to evaluate how the deep-learning  technique of Convolutional Auto-Encoders (CAE) could improve the clustering of the asset universe. 

Similarly to the authors , when eligible pairs are identified, we will apply the set of 4 rules as defined by the authors, i.e., cointegration, Hurst metric , half-life, and average mean reversions (31–33). 

Then we build train forecasting models (MLP & LSTM) to do the pair trading.

We plan to implement some of the suggestions mentioned in the book (104), by building a cross-market asset universe, focusing on technology-related topics, whether for ETFs, stocks, or commodities. In our 1st draft submitted on 30-04-25, we used the 44 ETFs that have volume above 1 million from Yahoo Finance.

## Results

The following are our results for the 1st draft. We are not satisfied with these results, therefore we are trying out other forecasting models & thinking of using RL models instead of forecasting models for trading.

**Validation**

| | MLP  | LSTM |
| ------------- | ------------- | ------------- |
| Portfolio sharpe ratio | 0.26 | 1.55 |
| Maximum drawdown of portfolio  | -0.39% | -0.48% |
| Total Drawdown Days | 126 days | 114 days |
| Max DD period | 1 days | 0 days |
| Total number of trades | 583 | 219 |
| Positive trades | 169 | 61 |
| Negative trades | 414 | 158 |
| Annual ROI | 0.18 | 1.37 |
 
**Test unrestricted**

| | MLP  | LSTM |
| ------------- | ------------- | ------------- |
| Portfolio sharpe ratio | -1.36 | 0.04 |
| Maximum drawdown of portfolio  | -1.42% | -1.10% |
| Total Drawdown Days | 125 days | 117 days |
| Max DD period | 3 days | 1 days |
| Total number of trades | 546 | 363 |
| Positive trades | 208 | 145 |
| Negative trades | 338 | 218 |
| Annual ROI | -1.38 | 0.05 |

**Test with active pairs on validation set**

| | MLP  | LSTM |
| ------------- | ------------- | ------------- |
| Portfolio sharpe ratio | -0.27 | 0.67 |
| Maximum drawdown of portfolio  | -2.10% | -3.70% |
| Total Drawdown Days | 114 days | 109 days |
| Max DD period | 1 days | 1 days |
| Total number of trades | 269 | 188 |
| Positive trades | 122 | 110 |
| Negative trades | 147 | 78 |
| Annual ROI | -0.64 | 2.43 |


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

