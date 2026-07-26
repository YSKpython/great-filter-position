# Where Is the Filter? — analysis code

Code for **"Where Is the Filter? Joint Observational Underdetermination
from the Fermi Silence, the Doomsday Tail, and the Anthropic Shadow"**
(Yavuz Selim Kılınç, v2, 2026).

The paper embeds three observation-selection channels — the Fermi null
sky (SOM), the Doomsday birth rank (TWD), and the anthropic shadow — in
one Bayesian model parameterized by the filter position `phi`, and
measures what the data can say about `phi` in bits of information gain.

## Reproducing everything

    pip install numpy matplotlib        # numpy >= 1.20, matplotlib >= 3.4
    python filter_position.py           # 7-item verification suite, then all tables + figures
    python filter_position.py --checks  # verification suite only
    python filter_position.py --figures # tables and figures only

`filter_position.py` runs the verification suite first and refuses to
produce output if any check fails. On success it writes every number in
the manuscript to `results.txt` and `results.json`, and the six figures
to `figures/`.

## Modules

| file | role |
|---|---|
| `filter_core.py` | model: couplings, channel likelihoods, posterior, information metrics, coupling class |
| `filter_checks.py` | verification suite (7 checks, each prints what it proves) |
| `filter_experiments.py` | Tables 1–5 and Figures 1–6 |
| `filter_position.py` | single driver |

## What changed from v1

See `REPO_NOTES.md`. In short: `phi` is redefined as a log-improbability
share; all couplings are bounded by construction (the v1 `Q`-clip at
`phi = 0.808` is gone); the survival channel uses the likelihood
`1 - P(1-Q)` rather than the posterior `PQ/(1-P+PQ)`; the birth-rank
channel marginalises over `N`; information is reported in bits with
analytic ceilings.

## License

MIT. See `LICENSE`.