# **Nonlinear Clustering for Pairs Trading: A Deep Learning Approach with Convolutional Autoencoders**

This work is a part Capstone project work for the completion of MSc in Financial Engineering from WorldQuant University. Among the varous topics, we chose, MScFE 690 CP Track 8 Machine Learning (Deep) Investment Strategies:- Topic 3 - Machine Learning (Deep) Application to Long/Short Pair Trading Strategies

## Abstract

This paper enhances the pairs trading machine learning framework proposed by Sarmento and Horta by integrating deep learning techniques within the original two-stage structure. Specifically, we replace the traditional Principal Component Analysis (PCA) used in the pairs selection stage with Convolutional Auto-Encoders (CAE) to capture nonlinear dependencies among financial assets. The CAE-extracted features are clustered using agglomerative clustering to identify candidate trading pairs, which are then filtered using a rule based selection involving cointegration and mean-reversion metrics. Preliminary results on a universe of ETFs show that CAEs achieve low reconstruction error and produce compact and interpretable clusters. We benchmark our framework against both classical and modern strategies, demonstrating the potential for improved clustering quality and trading performance. This work supports the growing interest in leveraging deep learning for financial time series and contributes a scalable, data-driven approach to pairs trading.

## Hypothesis 

We propose the following hypothesis:

CAEs can extract latent representations of financial assets that reveal nonlinear clustering structures, leading to more effective pair selection and improved trading performance compared to PCA-based methods.


## Results

The following are our results for the 1st draft. It shows that clustering using CAE hleps us identify more number of pairs, & also beter trading perfomace. We are also trying out other forecasting models for trading.

### **Trading on 900 pairs selected from 10 CAE clusters**

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


### **Trading on 9 pairs selected from 6 PCA clusters**

**Validation**

| | MLP  | LSTM |
| ------------- | ------------- | ------------- |
| Portfolio sharpe ratio | 1.02 | 0.99 |
| Maximum drawdown of portfolio  | -1.65% | -1.12% |
| Total Drawdown Days | 119 days | 63 days |
| Max DD period | 1 days | 0 days |
| Total number of trades | 26 | 9 |
| Positive trades | 16 | 5 |
| Negative trades | 10 | 4 |
| Annual ROI | 2.15 | 1.16 |
 
**Test unrestricted**

| | MLP  | LSTM |
| ------------- | ------------- | ------------- |
| Portfolio sharpe ratio | -1.51 | -1.17 |
| Maximum drawdown of portfolio  | -6.83% | -5.47% |
| Total Drawdown Days | 130 days | 126 days |
| Max DD period | 3 days | 2 days |
| Total number of trades | 22 | 12 |
| Positive trades | 7 | 7 |
| Negative trades | 15 | 5 |
| Annual ROI | -6.08 | -4.74 |

**Test with active pairs on validation set**

| | MLP  | LSTM |
| ------------- | ------------- | ------------- |
| Portfolio sharpe ratio | 0.03 | 1.19 |
| Maximum drawdown of portfolio  | -4.85% | -5.92% |
| Total Drawdown Days | 112 days | 99 days |
| Max DD period | 1 days | 0 days |
| Total number of trades | 11 | 7 |
| Positive trades | 5 | 6 |
| Negative trades | 6 | 1 |
| Annual ROI | 0.16 | 11.45 |


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

