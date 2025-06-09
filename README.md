# **Nonlinear Clustering for Pairs Trading: A Deep Learning Approach with Convolutional Autoencoders**

This work is a part Capstone project work for the completion of MSc in Financial Engineering from WorldQuant University. Among the varous topics, we chose, MScFE 690 CP Track 8 Machine Learning (Deep) Investment Strategies:- Topic 3 - Machine Learning (Deep) Application to Long/Short Pair Trading Strategies

## Abstract

This paper enhances the pairs trading machine learning framework proposed by Sarmento and Horta by integrating deep learning techniques within the original two-stage structure. Specifically, we replace the traditional Principal Component Analysis (PCA) used in the pairs selection stage with Convolutional Auto-Encoders (CAE) to capture nonlinear dependencies among financial assets. The CAE-extracted features are clustered using agglomerative clustering to identify candidate trading pairs, which are then filtered using a rule based selection involving cointegration and mean-reversion metrics. Preliminary results on a universe of ETFs show that CAEs achieve low reconstruction error and produce compact and interpretable clusters. We benchmark our framework against both classical and modern strategies, demonstrating the potential for improved clustering quality and trading performance. This work supports the growing interest in leveraging deep learning for financial time series and contributes a scalable, data-driven approach to pairs trading.

## Hypothesis 

We propose the following hypothesis:

CAEs can extract latent representations of financial assets that reveal nonlinear clustering structures, leading to more effective pair selection and improved trading performance compared to PCA-based methods.


## Results

The following are our results for the 1st draft. It shows that clustering using CAE hleps us identify more number of pairs, & also beter trading perfomace. We are also trying out other forecasting models for trading.

### **Trading on 99 pairs selected from 5 CAE clusters**

**Validation**

| | MLP  | LSTM | Encoder_Decoder |
| ------------- | ------------- | ------------- | ------------- |
| Portfolio sharpe ratio (Daily) | -0.41 | -0.38 | -0.16 |
| Sharpe Ratio assumming IID returns | -6.63 | -6.36 | -2.63 |
| Maximum drawdown of portfolio  | -2.34% | -0.30% | -1.86% |
| Total Drawdown Days | 134 days | 111 days | 125 days |
| Max DD period | 3 days | 0 days | 1 days |
| Total number of trades | 699 | 143 | 608 |
| Positive trades | 323 | 100 | 358 |
| Negative trades | 376 | 43 | 250 |
| Annual ROI | -2.19 | 1.35 | 0.71 |
 
**Test unrestricted**

| | MLP  | LSTM | Encoder_Decoder |
| ------------- | ------------- | ------------- | ------------- |
| Portfolio sharpe ratio (Daily) | -0.10 | -0.01 | -0.09 |
| Sharpe Ratio assumming IID returns | -1.62 | -0.19 | -1.55 |
| Maximum drawdown of portfolio  | -4.70% | -3.51% | -4.78% |
| Total Drawdown Days | 132 days | 122 days | 121 days |
| Max DD period | 2 days | 1 days | 1 days |
| Total number of trades | 1350 | 391 | 882 |
| Positive trades | 648 | 213 | 477 |
| Negative trades | 702 | 178 | 405 |
| Annual ROI | -3.82 | -0.66 | -2.14 |

**Test with active pairs on validation set**

| | MLP  | LSTM | Encoder_Decoder |
| ------------- | ------------- | ------------- | ------------- |
| Portfolio sharpe ratio (Daily) | -0.07 | -0.08 | -0.05 |
| Sharpe Ratio assumming IID returns | -1.20 | -1.36 | -0.87 |
| Maximum drawdown of portfolio  | -3.42% | -4.57% | -4.17% |
| Total Drawdown Days | 117 days | 110 days | 114 days |
| Max DD period | 1 days | 1 days | 1 days |
| Total number of trades | 401 | 153 | 580 |
| Positive trades | 214 | 73 | 315 |
| Negative trades | 187 | 80 | 265 |
| Annual ROI | 1.54 | 0.68 |  0.66 |


### **Trading on 15 pairs selected from 5 PCA clusters**

**Validation**

| | MLP  | LSTM |
| ------------- | ------------- | ------------- |
| Portfolio sharpe ratio (Daily) | 1.02 | 0.99 |
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
| Portfolio sharpe ratio (Daily) | -1.51 | -1.17 |
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
| Portfolio sharpe ratio (Daily) | 0.03 | 1.19 |
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

