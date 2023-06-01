# Crypto DRL trading DEMO

**Nothing in this demo constitutes professional and/or financial advice, nor does any information on this repo constitute a comprehensive or complete statement of the matters discussed or the law relating thereto.**

## Descriptions

This demo aims to create an intelligent and dynamic trading platform prototype by harnessing the power of deep reinforcement learning (DRL). Using the Stable baseline 3 framework, this demo showcases the capabilities of DRL in multi-stock trading scenarios, with a special focus on crypto markets.

![plot](./plot/crypto_trading.jpg)

## Notebooks

1. fetchs crypto-trading data
2. EDA
3. DRL training and inference
   
## code structure

``` 
crypto_trading/
├── agent
├── dataset
│   ├── daily-level
│   └── hour-level
├── model_saves
├── notebooks
│   ├── data_retrieve
│   ├── EDA
│   └── training
├── plot
│   ├── dataset
│   ├── fig_save
│   └── test_plot
├── processor
├── results
└── trading_env
```
