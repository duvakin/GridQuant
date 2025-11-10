# ⚡GridQuant
European market **spot price forecasts** via Monte Carlo analysis and Geometric Brownian Motion.  

## Overview
Today, with cloud storage and AI placing unprecedented strain on the grid, energy security and building operational costs are not only an urban planning concern but a growing business risk.

This python-based package introduces a crucial co-simulation and energy market forecasting method leveraging [Geometric Brownian Motion](https://www.sciencedirect.com/topics/mathematics/geometric-brownian-motion) (GBM) and [Monte Carlo](https://www.sciencedirect.com/science/article/pii/S2212567114004638) (MC) analysis to forecast energy prices with uncertainty in increasingly more volatile energy markets.

The workflow consists of three key stages:
1. **Data Retrival** — Historical hourly energy spot prices from bidding zones in the European Union are accessed via the [European Network of Transmission System Operators for Electricity](https://www.entsoe.eu/) (ENTSOE) API.
2. **Empirical Volatility Calculation** — The model computes both hourly and annualized volatility based on historical spot prices, forming the foundation of the stochastic price predictions.
3. **Stochastic Forecast Simulation** — The system performs Monte Carlo simulations based on GBM dynamics to generate forward-looking energy price scenarios and uncertainty bounds.

## Installation
`python3 -m pip install entsoe-py`

*NOTE: To use `EntsoeRawClient` You will need to request your own API key by emailing [transparency@entsoe.eu](transparency@entsoe.eu).*

## Usage
TODO
