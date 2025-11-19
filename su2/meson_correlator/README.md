# Meson Correlator Calculations - SU(N) Implementation

**Author**: Zeke Mohammed
**Institution**: Fordham University
**Date**: Fall 2025

## Overview

Modular Python implementation of meson correlator calculations for lattice QCD with full SU(N) support. Calculates meson masses (pion, sigma, rho) from Wilson fermion propagators on thermalized gauge configurations.

## Features

### SU(N) Generalization
- **Arbitrary Color Support**: Works for SU(2), SU(3), or any SU(N)
- **Backward Compatible**: Default `n_colors=2` maintains SU(2) behavior
- **Fully Tested**: Test suite for N=2 and N=3

### Meson Channels
- **Pion (π)**: Pseudoscalar, γ₅ operator
- **Sigma (σ)**: Scalar, identity operator
- **Rho (ρ)**: Vector, γᵢ operators (x, y, z polarizations)

### Analysis Tools
- Ensemble averaging across gauge configurations
- Effective mass extraction
- Plateau fitting with error estimation
- Batch processing for overnight runs

## Quick Start

### Calculate Pion Mass (SU(2))
```python
import PionCalculator

mass = PionCalculator.calculate_pion_mass(
    U,                          # Gauge configuration
    mass=0.1,                   # Quark mass
    lattice_dims=[6,6,6,20],    # Lattice size
    wilson_r=0.5,               # Wilson parameter
    verbose=True
)
```

### Calculate with SU(3)
```python
mass = PionCalculator.calculate_pion_mass(
    U, mass=0.1, lattice_dims=[6,6,6,20],
    n_colors=3,  # SU(3) QCD
    verbose=True
)
```

### Run Full Spectrum Analysis
```python
python3 Propagator.py \
  --input-config gauge_configs/config.pkl \
  --channel all \
  --mass 0.2 \
  --verbose
```

### Batch Process Multiple Configurations
```python
python3 run_correlators_overnight.py \
  --config-dir gauge_configs \
  --output-dir correlator_outputs \
  --every-n 5
```

### Analyze Ensemble
```python
python3 analyze_correlators.py \
  --input-dir correlator_outputs/m020_b240 \
  --output-dir plots
```

## Module Structure

### Core Modules
- **MesonBase.py**: Wilson-Dirac matrix, propagator solvers, common utilities
- **PionCalculator.py**: Pion (pseudoscalar) correlator calculations
- **SigmaCalculator.py**: Sigma (scalar) correlator calculations
- **RhoCalculator.py**: Rho (vector) correlator calculations
- **sun.py**: SU(N) gauge theory operations and utilities

### Scripts
- **Propagator.py**: Main driver for single/spectrum calculations
- **analyze_correlators.py**: Ensemble averaging and analysis
- **run_correlators_overnight.py**: Batch processing automation

### Tests
- **test_sun_quick.py**: Function signature verification
- **test_sun_compatibility.py**: SU(2) backward compatibility
- **test_su3_demo.py**: SU(3) infrastructure validation

## Usage Examples

### Example 1: Single Configuration
```bash
python3 Propagator.py \
  --input-config config.pkl \
  --channel pion \
  --mass 0.1 \
  --ls 6 --lt 20 \
  --save-correlators
```

### Example 2: Mass Scan
```bash
python3 Propagator.py \
  --input-config config.pkl \
  --mass-scan "0.05,0.10,0.15,0.20" \
  --channel pion
```

### Example 3: Full Spectrum
```bash
python3 Propagator.py \
  --input-config config.pkl \
  --channel all \
  --mass 0.2
```

## Technical Details

### Physics Implementation
- **Wilson Fermions**: Standard lattice discretization
- **Antiperiodic BCs**: In time direction (quenched approximation)
- **Point Source**: At t=0 for propagator calculation
- **Gamma Matrices**: Dirac representation

### Propagator Indexing
Global index formula: `idx = site_idx × (N_c × 4) + 4 × color + spin`
- Correct ordering ensures proper Dirac/color structure
- Critical for multi-color calculations

### Ensemble Averaging
- Statistical errors via jackknife method
- Per-configuration YAML output format
- Automated plateau identification

## Requirements

```python
numpy >= 1.20
scipy >= 1.7
matplotlib >= 3.3
pyyaml >= 5.4
```

## Testing

Run all tests:
```bash
python3 test_sun_quick.py
python3 test_sun_compatibility.py
python3 test_su3_demo.py
```

Expected output: All tests passing ✓

## File Format

### Input: Gauge Configuration (pickle)
```python
{
    'U': array,           # Gauge links
    'plaquette': float,   # Average plaquette
    'beta': float,        # Coupling
    'Lx', 'Ly', 'Lz', 'Lt': int
}
```

### Output: Correlator Data (YAML)
```yaml
PION_5:
  0: 0.6909...
  1: 0.0493...
  ...
SIGMA:
  0: 0.6910...
  ...
```

## Performance Notes

- **SU(2)**: ~15 seconds per configuration (6³×20 lattice)
- **SU(3)**: ~45 seconds per configuration (3× slower)
- **Memory**: ~1 GB for 6³×20 lattice
- **Solver**: BiCGSTAB (iterative, faster) or direct (exact, slower)

## Known Issues

### Timeouts
Some configurations may timeout (>10 min) due to difficult matrix inversions. This is normal - typically get 15-25 successful configs out of 40 attempted.

### Statistical Fluctuations
Single-configuration results may show:
- Large errors (20-50%)
- Unphysical orderings (e.g., M_ρ < M_π)
- Failed channels (poor signal-to-noise)

**Solution**: Ensemble averaging over multiple configurations.

## Future Work

- [ ] SU(3) thermal generator implementation
- [ ] Smeared sources for improved signal
- [ ] Stochastic sources for variance reduction
- [ ] APE smearing for gauge noise reduction

## References

- Wilson fermions: [Lattice QCD textbook, Ch. 10]
- Meson correlators: Standard LQCD methodology
- SU(N) gauge theory: Group theory implementation
