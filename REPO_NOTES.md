# What changed from v1 to v2

1. **`phi` redefined.** It is now the share of the chain's total
   log-improbability carried by steps already passed. Couplings are
   exponential in `phi`, and the `phi`-invariance of `lambda` becomes a
   substantive, falsifiable condition (`Lambda_B = Lambda_A`) instead of
   an algebraic identity.

2. **Bounded by construction.** Every coupled quantity stays inside its
   physical range. v1 clipped `Q(phi)` at 1.0, producing the unexplained
   kink at `phi = 0.808` in v1 Figures 1, 3d, 5a and 6b. Nothing is
   clipped here; `check_bounded` verifies this over the whole coupling
   class.

3. **Correct survival likelihood.** The channel uses
   `P(E | P,Q) = 1 - P(1-Q)`, not the posterior catastrophe probability
   `PQ/(1-P+PQ)` that v1 multiplied into the joint likelihood. `eta` is
   retained as a reported diagnostic only. The identified combination is
   `P(1-Q)`, not `eta`.

4. **Birth rank marginalised.** The TWD channel marginalises over `N`
   with the power-law prior, so both `alpha` and `beta` survive into the
   joint model; the sum degeneracy does not transfer verbatim.

5. **Information in bits.** Results are reported as KL divergence from
   the uniform prior, with analytic per-channel ceilings, replacing an
   un-normalised "posterior variance" that was in fact `E[(phi-0.5)^2]`.

Each item is reproduced and diagnosed by `check_v1_artefacts` and the
other checks in `filter_checks.py`.