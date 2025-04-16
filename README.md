# WQU-Capstone
MScFE 690 CP Track 8 - Machine Learning (Deep) Investment Strategies:- Topic 3 - Machine Learning (Deep) Application to Long/Short Pair Trading Strategies


## Problem Statement

Our project consists of improving the two-stage Pairs Trading framework as demonstrated in the book “A Machine Learning-Based Pairs Trading Investment Strategy” (Sarmento and Horta). 
In the first stage of pairs selection (21–35), dimensionality reduction and clustering techniques were applied to identify co-moving assets. We propose to evaluate how the deep-learning  technique of Convolutional Auto-Encoders (CAE) could improve the clustering of the asset universe. 
Similarly to the authors , when eligible pairs are identified, we will apply the set of 4 rules as defined by the authors, i.e., cointegration, Hurst metric , half-life, and average mean reversions (31–33). 
An additional step that we propose in this pairs selection process is to build a second selection of pairs using Copula  for comparison. 
A hybrid approach combining both Rule-Based and Copula could be explored in an eventual scenario.
For the second stage, which consists of predicting the trades, we will implement a Reinforcement Learning for more flexibility to tackle Sarmento and Hort’s forecasting-based limitation of fixed threshold (86–94).
We plan to implement some of the suggestions mentioned in the book (104), by building a cross-market asset universe, focusing on technology-related topics, whether for ETFs, stocks, or commodities. We will also attempt to train the Reinforcement Learning algorithm on sentiment data.



### We have also identified the following resources as potentially reusable or improvable:

Daehkiml. Pair Trading: A Market-Neutral Trading Strategy with Integrated Machine Learning. https://daehkim.github.io/pair-trading/. Accessed 30 Mar. 2025.
Jansen, Stefan. Stefan-Jansen/Machine-Learning-for-Trading. 2018. 30 Mar. 2025. GitHub, https://github.com/stefan-jansen/machine-learning-for-trading.
Polakow, Oleg. VectorBT. https://vectorbt.dev/.
Roychoudhury, Raktim. Pairs Trading Using Unsupervised Clustering and Deep Reinforcement Learning. 2023. 8 Apr. 2025. GitHub, https://github.com/raktim-roychoudhury/pairs_trading.
tensortrade.org. TensorTrade. https://github.com/tensortrade-org. Accessed 14 Apr. 2025.
Yan, Zijian. Yan1015/Pairs-Trading-Using-Copula. 2018. 9 Mar. 2025. GitHub, https://github.com/Yan1015/Pairs-Trading-using-Copula.
Yang, Hongshen. Cryptocurrency Trading with Reinforcement Learning Based on Backtrader. 2023. 30 Mar. 2025. GitHub, https://github.com/Hongshen-Yang/pair-trading-envs.
