# Expected Outputs for Lattice QCD Modular System

## Overview

This document describes the expected outputs and directory structure for the modular lattice QCD propagator system when run with sample configurations.

## Sample Configurations

### 1. Identity Configuration (`identity_4x4x4x4.pkl`)
- **Type**: Free field theory
- **Description**: All gauge links set to identity matrix
- **Plaquette**: 1.0
- **Physics**: No gauge interactions, pure kinetic fermion theory

**Expected Meson Masses**:
- All mesons should have similar masses (degeneracy)
- Typical range: 0.3 - 0.8 (depending on quark mass and Wilson parameter)
- **M_π ≈ M_ρ ≈ M_σ** (no gauge field splitting)

### 2. Random Configuration (`random_4x4x4x4.pkl`)
- **Type**: Strong coupling regime
- **Description**: Random SU(2) gauge links
- **Plaquette**: ~0.5
- **Physics**: Strong gauge interactions, confinement effects

**Expected Meson Masses**:
- Large mass splittings expected
- **M_π < M_ρ < M_σ** (typical hierarchy)
- Possible rotational symmetry breaking: M_ρₓ ≠ M_ρᵧ ≠ M_ρᵧ

## Command Line Usage

### Single Channel Calculations

```bash
# Pion channel
python3 PropagatorModular.py --mass 0.1 --ls 4 --lt 4 --channel pion \
    --input-config sample_inputs/identity_4x4x4x4.pkl --output results_pion

# Sigma channel
python3 PropagatorModular.py --mass 0.1 --ls 4 --lt 4 --channel sigma \
    --input-config sample_inputs/identity_4x4x4x4.pkl --output results_sigma

# Rho channels (individual polarizations)
python3 PropagatorModular.py --mass 0.1 --ls 4 --lt 4 --channel rho_x \
    --input-config sample_inputs/identity_4x4x4x4.pkl --output results_rho_x

python3 PropagatorModular.py --mass 0.1 --ls 4 --lt 4 --channel rho_y \
    --input-config sample_inputs/identity_4x4x4x4.pkl --output results_rho_y

python3 PropagatorModular.py --mass 0.1 --ls 4 --lt 4 --channel rho_z \
    --input-config sample_inputs/identity_4x4x4x4.pkl --output results_rho_z
```

### Full Spectrum Calculation

```bash
# All channels at once
python3 PropagatorModular.py --mass 0.1 --ls 4 --lt 4 --channel all \
    --input-config sample_inputs/identity_4x4x4x4.pkl --output results_spectrum
```

## Expected Directory Structure

When running the above commands, the following directory structure is created:

```
results_[channel]_[timestamp]/
├── data/
│   ├── results_[timestamp].txt      # Human-readable summary
│   ├── results_[timestamp].json     # Machine-readable data
│   └── correlators_[timestamp].npz  # Raw correlator data (if saved)
├── plots/                           # (Optional) Generated plots
│   ├── correlator_[channel].png
│   └── effective_mass_[channel].png
└── logs/
    └── calculation_[timestamp].log  # Detailed execution log
```

### Example Output Files

#### 1. Text Summary (`results_[timestamp].txt`)

```
LATTICE QCD MESON MASS RESULTS
=================================
Generated: 2025-09-23 14:15:30

IMPORTANT PHYSICS NOTES:
------------------------
These results are from a SINGLE gauge configuration.
Physical meson masses require ensemble averaging over
many configurations (typically 100-1000).

Single-config results may show:
- Statistical fluctuations (e.g., M_ρ < M_π)
- Broken rotational symmetry (M_ρx ≠ M_ρy ≠ M_ρz)
- Poor signal-to-noise in some channels
- Deviations from expected mass hierarchy

RESULTS:
--------
Channel: pion (0-+)
Mass: 0.456789 ± 0.045679
Chi²/dof: 1.23
Fit range: t = 2 to 6

Parameters:
  Quark mass: 0.100000
  Wilson r: 0.500000
  Effective mass: 2.100000
  Lattice: [4, 4, 4, 4]
  Solver: gmres

Execution time: 23.45 seconds
```

#### 2. JSON Data (`results_[timestamp].json`)

```json
[
  {
    "channel": "pion",
    "input_mass": 0.1,
    "meson_mass": 0.456789,
    "meson_error": 0.045679,
    "chi_squared": 1.23,
    "fit_range": [2, 6],
    "correlator": [0.6234, 0.3456, 0.1789, ...],
    "effective_mass": [null, null, 0.456, 0.467, ...],
    "mass_errors": [null, null, 0.046, 0.047, ...],
    "lattice_dims": [4, 4, 4, 4],
    "wilson_r": 0.5,
    "solver": "gmres",
    "execution_time": 23.45,
    "channel_info": {
      "JPC": "0-+",
      "name": "Pion",
      "description": "Pseudoscalar meson"
    }
  }
]
```

## Physics Interpretation

### Identity Configuration Results

For free field theory (identity configuration), expect:

1. **Mass Degeneracy**: All meson masses should be approximately equal
   - Physical reason: No gauge field interactions to split degenerate states
   - Typical values: M ≈ 0.4-0.7 for m_quark = 0.1, r = 0.5

2. **Clean Exponential Decay**: Correlators should show smooth exponential behavior
   - Good signal-to-noise ratio
   - Clean plateau in effective mass

3. **Rotational Symmetry**: M_ρₓ = M_ρᵧ = M_ρᵧ (exact equality)

### Random Configuration Results

For strong coupling (random configuration), expect:

1. **Mass Hierarchy**: M_π < M_ρ < M_σ (approximately)
   - Gauge interactions lift degeneracies
   - Pion is typically lightest (approximate Goldstone boson)

2. **Symmetry Breaking**: M_ρₓ ≠ M_ρᵧ ≠ M_ρᵧ
   - Single configuration breaks rotational symmetry
   - Differences can be significant (10-50%)

3. **Noisy Correlators**: Stronger gauge fluctuations → more noise
   - Larger error bars
   - Possible fitting challenges

## Typical Execution Times

On a standard laptop (assuming 4×4×4×4 lattice):

- **Single channel**: 15-45 seconds
- **All channels**: 60-180 seconds
- **Identity config**: Faster (simpler propagators)
- **Random config**: Slower (complex linear systems)

## Quality Checks

### Good Results Indicators:
- ✅ Chi²/dof ≈ 1.0 (good fit quality)
- ✅ Fit range spans 3-5 time slices
- ✅ Error bars < 20% of mass value
- ✅ Positive definite masses

### Warning Signs:
- ⚠️ Chi²/dof >> 2.0 (poor fit)
- ⚠️ Very large error bars (> 50%)
- ⚠️ Negative effective masses
- ⚠️ No plateau region found

### Common Issues:
- **"No convergence"**: Linear solver failed → try different solver
- **"No plateau found"**: Poor signal → increase statistics or change parameters
- **"Negative correlator"**: Normal for vector channels, handled automatically

## Testing Verification

To verify the system works correctly:

1. **Run identity config, pion channel**: Should get M ≈ 0.4-0.7
2. **Run identity config, all channels**: All masses should be similar
3. **Run random config, all channels**: Should see mass hierarchy
4. **Check directory structure**: All expected files created
5. **Verify JSON format**: Machine-readable data intact

This completes the expected outputs documentation for systematic testing and verification.