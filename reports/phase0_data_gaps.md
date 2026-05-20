# Phase 0 — Data Gap Analysis

## What we have

| Market | Closing lines available | Source | Coverage |
|---|---|---|---|
| Moneyline | ✅ Yes | tennis-data.co.uk (Pinnacle/Avg/B365) | ATP 2001+, WTA 2007+ |
| Total Games O/U | ❌ No | — | Outcomes only (from Sackmann set scores) |
| Set Spread | ❌ No | — | Outcomes only (from Sackmann set counts) |

## Implication for Phase 0

**Moneyline** backtests are full P&L backtests — we have closing ML odds from Pinnacle (the sharpest book) and can compute exact CLV and ROI.

**Games O/U and Set Spread** run in *price-discovery mode*: we analyse the historical outcome distributions and compute fair prices, but we cannot backtest P&L because we don't have the lines the books actually offered.

## What price-discovery mode tells us

Even without closing lines, analysing outcomes gives us:

1. **Fair price anchor** — the empirical over rate at any line tells us what a break-even bet costs. A model that predicts totals more accurately than the empirical distribution has structural edge.

2. **Surface / context variation** — clay produces more games than grass; Bo5 totals cluster around 35-36; tight favourites produce more sets than heavy chalks. These variations are what a generative serve/return model will learn to exploit.

3. **The right line to target** — books typically offer Bo3 totals at 21.5–23.5 and Bo5 totals at 33.5–38.5. Our distribution analysis pins down which line is hardest for the market to price accurately.

## Closing odds sources to investigate in Phase 3

For historical **totals and spread** closing lines:

- **OddsPapi** — claims 350+ books including Pinnacle and Betfair Exchange, plus historical odds. Claims to cover alt markets (totals, spreads). Free tier available. **Highest priority to verify.**
- **The Odds API** — ATP/WTA Slams + 1000/500 coverage. Primarily ML; totals coverage unconfirmed for tennis. Credit-based pricing.
- **Betfair Exchange historical data** — Betfair's Smart Data product sells historical prices; may include totals. Price: $$$, but Betfair is the market of record for sharp tennis prices.
- **Pinnacle historical API** — restricted; not publicly available without a partnership.

**Recommendation**: When beginning Phase 3, first check OddsPapi's coverage for `tennis_totals` and `tennis_spreads` endpoints before any API spend is committed.
