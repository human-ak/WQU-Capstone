# **Nonlinear Clustering for Pairs Trading: A Deep Learning Approach with Convolutional Autoencoders**

This work is a part Capstone project work for the completion of MSc in Financial Engineering from WorldQuant University. Among the varous topics, we chose, MScFE 690 CP Track 8 Machine Learning (Deep) Investment Strategies:- Topic 3 - Machine Learning (Deep) Application to Long/Short Pair Trading Strategies

## Abstract

This paper enhances the pairs trading machine learning framework proposed by Sarmento and Horta by integrating deep learning techniques within the original two-stage structure. Specifically, we replace the traditional Principal Component Analysis (PCA) used in the pairs selection stage with Convolutional Auto-Encoders (CAE) to capture nonlinear dependencies among financial assets. The CAE-extracted features are clustered using agglomerative clustering to identify candidate trading pairs, which are then filtered using a rule based selection involving cointegration and mean-reversion metrics. Preliminary results on a universe of ETFs show that CAEs achieve low reconstruction error and produce compact and interpretable clusters. We benchmark our framework against both classical and modern strategies, demonstrating the potential for improved clustering quality and trading performance. This work supports the growing interest in leveraging deep learning for financial time series and contributes a scalable, data-driven approach to pairs trading.

## Hypothesis 

We propose the following hypothesis:

CAEs can extract latent representations of financial assets that reveal nonlinear clustering structures, leading to more effective pair selection and improved trading performance compared to PCA-based methods.


## Results

The following results shows that clustering using CAE hleps us identify more number of pairs, & also beter trading performance.

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

| | MLP  | LSTM | Encoder_Decoder |
| ------------- | ------------- | ------------- | ------------- |
| Portfolio sharpe ratio (Daily) | -0.40 | -0.24 | -0.24 |
| Sharpe Ratio assumming IID returns | -6.37 | -3.93 | -3.96 |
| Maximum drawdown of portfolio  | -8.35% | -0.84% | -6.30% |
| Total Drawdown Days | 157 days | 114 days | 145 days |
| Max DD period | 3 days | 0 days | 2 days |
| Total number of trades | 217 | 62 | 174 |
| Positive trades | 86 | 36 | 81 |
| Negative trades | 131 | 26 | 93 |
| Annual ROI | -8.18 | 0.29 | -5.03 |
 
**Test unrestricted**

| | MLP  | LSTM | Encoder_Decoder |
| ------------- | ------------- | ------------- | ------------- |
| Portfolio sharpe ratio (Daily) | -0.29 | -0.18 | -0.30 |
| Sharpe Ratio assumming IID returns | -4.58 | -2.91 | -4.78 |
| Maximum drawdown of portfolio  | -7.84% | -5.16% | -10.78% |
| Total Drawdown Days | 133 days | 127 days | 140 days |
| Max DD period | 3 days | 2 days | 3 days |
| Total number of trades | 217 | 81 | 267 |
| Positive trades | 86 | 40 | 108 |
| Negative trades | 131 | 41 | 159 |
| Annual ROI | -7.53 | -4.57 | -10.77 |

**Test with active pairs on validation set**

| | MLP  | LSTM | Encoder_Decoder |
| ------------- | ------------- | ------------- | ------------- |
| Portfolio sharpe ratio (Daily) | 0.01 | -0.03 | -0.09 |
| Sharpe Ratio assumming IID returns | 0.27 | -0.54 | -1.48 |
| Maximum drawdown of portfolio  | -5.65% | -5.31% | -10.51% |
| Total Drawdown Days | 98 days | 116 days | 120 days |
| Max DD period | 0 days | 1 days | 2 days |
| Total number of trades | 15 | 36 | 48 |
| Positive trades | 8 | 18 | 27 |
| Negative trades | 7 | 18 | 21 |
| Annual ROI | 7.70 | 0.69 |  -5.47 |


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

