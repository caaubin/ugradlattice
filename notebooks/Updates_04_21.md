# Updates 04/21

## What was done

### NB08 — Quark Propagators
- Added explanation cell after the propagator-vs-time plot (Section 4)
  explaining why the cosh-like shape is correct: forward + backward
  propagation from antiperiodic BCs gives
  Sigma|S|^2 ~ A*exp(-2m*t) + B*exp(-2m*(Lt-t)), minimum at t=Lt/2,
  upturn at t=3. Graph confirmed to agree with expectations.

### NB09 — Meson Correlators and Masses
- Fixed gamma matrix display formatting in cell 3 (matrices now align
  properly under "Gamma =")
- Added explanation cell after pion correlator plot (Section 2):
  cosh shape from APBC, effective mass plateau with only 3 points,
  and note that the interacting plot in Section 5 will look similar
  (not a bug — small lattice dominates the visual shape)
- Added explanation cell after spectrum bar chart (Section 3):
  free-field degeneracy expectation, rho degeneracy requirement,
  gamma convention note
- Added explanation cell after GMOR plot (Section 4):
  why M_pi shows a U-shape (not flat) on the free field — Wilson
  mass dominates, fitting systematics on tiny lattice
- Fixed stray `<cell_type>markdown</cell_type>` tags (none found in NB09
  but checked)

### NB10 — Chiral Physics and GMOR
- Fixed GMOR plot (right panel, Section 4): removed the m_crit vertical
  line that stretched the x-axis to -2.0 and squished data into one
  corner. Now zoomed in on data region with m_crit reported as text
  annotation instead
- Fixed chiral extrapolation plot (Section 5): extended fit line from
  m_q=0 through the data so the extrapolation is actually visible.
  Fixed y-value computation that was inconsistently mixing bare and
  shifted masses
- Added explanation cells after both plots
- Removed stray `<cell_type>markdown</cell_type>` tags from cells 0 and 12

### NB11 — Ensemble Analysis
- Reviewed all outputs. Pion/sigma hierarchy (M_pi < M_sigma) is correct.
  Noisy correlator and lack of clean plateau are expected at this
  lattice size and statistics. No code changes needed.
- Identified rho_z data issue (see below).

### Rho gamma convention fix (MesonBase.py, RhoCalculator.py, Propagator.py)
- **Bug:** the meson operator code assumed physics convention where
  gamma0 = temporal, gamma1,2,3 = spatial. But the Dirac operator in
  build_wilson_dirac_matrix maps gamma[mu] directly to lattice direction
  mu, with directions [x,y,z,t] = [0,1,2,3]. So gamma3 is temporal,
  not spatial.
- **Effect:** rho_z was using the temporal gamma instead of the z-spatial
  gamma, giving a different (wrong) mass. rho_x and rho_y were
  mislabeled but still used spatial gammas, so their masses are valid.
- **Fix applied to:**
  - `su2/meson_correlator/MesonBase.py` — channel definitions + comment
  - `su2/meson_correlator/RhoCalculator.py` — polarization mapping + docstring
  - `su2/meson_correlator/Propagator.py` — channel definitions + comment
- **New mapping:** rho_x->gamma0, rho_y->gamma1, rho_z->gamma2, rho_t->gamma3

### NB11 stray tags
- Removed `<cell_type>markdown</cell_type>` tags from 4 cells in NB11

---

## Outstanding: rho_z batch data needs regeneration

### The problem
The `.dat` files in `su2/meson_correlator/batch_results/` were generated
with the OLD gamma convention. The `rho_z_correlator_*.dat` files used
gamma3 (temporal gamma) instead of gamma2 (z-spatial gamma), so they
contain wrong correlator data.

- **rho_x data**: used gamma1 (y-spatial) — mislabeled but valid spatial
  polarization, mass is correct
- **rho_y data**: used gamma2 (z-spatial) — mislabeled but valid spatial
  polarization, mass is correct
- **rho_z data**: used gamma3 (temporal) — WRONG, needs regeneration

### What to do next time
1. Check how the original batch was run — look for a batch script or
   command history in `su2/meson_correlator/` (e.g. `run_batch.sh` or
   similar)
2. Regenerate only the rho_z channel:
   ```bash
   # For each gauge config used in the original batch:
   python3 PropagatorModular.py --channel rho_z \
       --input-config <config_file> --output <matching_output_dir> \
       --mass <same_mass> --ls <Ls> --lt <Lt> --save-correlators
   ```
3. Alternatively, regenerate all rho channels (rho_x, rho_y, rho_z) for
   clean labeling — masses for x/y won't change but labels will be correct
4. After regeneration, re-run NB11 to verify rho_z is now degenerate
   with rho_x and rho_y (within statistical errors)
5. Expected result: all three rho polarizations should give similar masses
   on the ensemble, with M_rho > M_pi

---

## Dr. Aubin meeting notes (raw, 04/21)

```
# Dr. Aubin edits notes

## [08]

- **[01]** Dr. Aubin edits in [1]. Pt source(s)
- **[02]** → more edits
- **[03]** → changed, not 3 propagators, the pt. source props;
  - *{ for correlator typically wall-point is best, but to really find masses you should do all of 'em (possible other cases). * }*

What I wrote was wall source... wall sink, but we're not doing that quite yet.

- → Change plot label to what's previous (he might've done it).
- → By squaring we're doing a meson*, include discussion here if both quarks were ours / would they flip sign. The meson just doesn't flip sign.

**Exercise ideas** → this is for t⁴, so even hand-drawn or code, what do you expect for t ↔ -t, k ↔ -t. Watch out which pts will be equal vs higher. Student exercises after all.

**Exercises:** Change "8 propagators" wording, like he did above already in another section.

---

## [09]

**Exercise:** if made of same quarks, why different spin?

- → Write γ in prettier way as we did in prev. file.
- **[1]** Subscript ij for flavors. Detail matters. When working calculations we only do charged pion, not neutral, make it clear. Even as we treat them the same, flavors do differ.
- Keep eqn 1 in [2], be added stuff, neaten it more yourself.
```

---

## 04/21 second pass — Aubin edits applied

### NB08 — items Dr. Aubin had already applied in-file (verified, left alone)
- [x] Section 1: point-source notation uses $\delta_P$
- [x] Section 2: "least expensive part of lattice QCD calculations,
      but is done far more often than generating the gauge fields"
- [x] Section 3 header: renamed to "The point-source--point-sink (pt-pt) Propagator"
- [x] Section 3 code: variable `propagator` (singular) + output "Total pt-pt propagator: 8"
- [x] Section 4 plot label: `|S(\vec{0},t;\, \vec{0},0)|^2` convention

### NB08 — items I applied this pass
- [x] **Section 3 markdown**: rewrote body to clarify "these are not 3 (or 8)
      different propagators — same $S$, inverted against 8 different point sources."
      Added the wall-source vs point-source note and the caveat that this
      notebook uses point sources only.
- [x] **Section 4 code**: fixed `prop0 = propagators[0]` → `prop0 = propagator[0]`
      (was inconsistent with singular variable name set by Aubin in Cell 7).
- [x] **Section 4 (new markdown cell)**: "Aside: squaring $|S|^2$ is already mesonic"
      — discussion of $S\,S^\dagger$ as meson kernel; sign-flip of both quarks
      leaves the meson unchanged.
- [x] **Exercises**: renamed Ex 5 from "All 8 propagators" to "pt-pt propagator
      matrix" and updated indexing from `propagators[...]` → `propagator[...]`.
- [x] **Exercises**: added new Ex 7 on $4^4$ lattice symmetry — predict
      behavior under $t \leftrightarrow -t$ and $\vec{x} \leftrightarrow -\vec{x}$,
      identify which lattice points must give equal magnitudes, sketch or code.

### NB09 — items I applied this pass
- [x] **Section 1 (Cell 3)**: introduced flavor indices in the meson operator
      $M_\Gamma^{ij}(x) = \bar\psi_i(x)\,\Gamma\,\psi_j(x)$. Added paragraph
      explaining that the charged pion $\pi^+ = \bar d\,\gamma_5\,u$ (not $\pi^0$)
      is what we compute, why charged pion has simpler Wick contractions
      (no disconnected diagrams), and that flavor distinction matters even
      with numerically degenerate flavors.
- [x] **Gamma matrix display (Cell 4)**: rewrote to match NB06's `gammaprint`
      style — real integer matrix for pion/sigma, "j *" prefix + integer
      matrix for imaginary-entry channels like rho_x (γ₀). Added alignment
      logic so rows line up cleanly even with the prefix.
- [x] **Section 2 (Cell 5)**: kept the first correlator equation as-is (Aubin
      asked to preserve it). Neatened the point-source equation into its own
      display line
      $S^{\rm pt\text{-}src}(\vec{x}, t) = S(\vec{x}, t;\, \vec{0}, 0)$,
      added a one-line mention of $\gamma_5$-hermiticity that collapses
      the correlator to the single trace, and tidied spacing throughout.
- [x] **Exercises (Cell 19)**: inserted new Ex 2 "Same quarks, different spin"
      — asks students to explain physically why $\gamma_5 \to \gamma_i \to \mathbb{1}$
      changes the meson's $J^{PC}$ and how Wick contraction preserves those
      quantum numbers. Existing exercises shifted down (3–10).

---

## Next time (pick up here)

1. **Re-run NB08 and NB09 end-to-end** to visually confirm today's edits:
   - NB08 plot renders with `propagator[0]` (singular) without error
   - NB09 Cell 4 gamma matrix display shows integer entries aligned cleanly
     with `j *` prefix for imaginary-entry channels (no stray `+0.j`)
   - Nothing else broke from the cell restructuring

2. **Regenerate rho_z batch data** — see the "Outstanding" section above.
   After regeneration, re-run NB11 to verify all three ρ polarizations
   are degenerate within errors and M_ρ > M_π.

Once those are done, the NB06 → NB11 pipeline is internally consistent
and ready for the next physics step.
