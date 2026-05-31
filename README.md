# Entropy Pooling – View Blending Engine (Meucci 2008)

Implements minimum relative entropy (cross‑entropy) to blend prior market distribution with investor views. Generalises Black‑Litterman to any prior (e.g., empirical returns) and any views (linear or nonlinear). Acts as a natural aggregation layer for outputs from 200+ engines.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Prior: kernel density estimate or bootstrap from historical returns
- Views: e.g., "top 3 ETFs outperform mean by 0.2%"
- Solves convex optimisation (KL divergence) with equality constraints
- Output: posterior expected returns (per‑ETF score)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-entropy-pooling-views-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py`
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- The posterior distribution is the one that is closest (in relative entropy) to the prior while satisfying the views.
- High posterior expected return indicates that an ETF is favoured by both prior history and the expressed views.
- This engine can be easily extended to accept outputs from any other engine (e.g., spiking NN, quantum walk, etc.) as views.

## Requirements

See `requirements.txt`.
