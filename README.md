# ugradlattice
Lattice QCD code written by and for undergraduates

This is a suite of Lattice QCD code written by Fordham University undergraduates to simulate basic lattice systems. Currently available here are those based upon work primarily by Michael Creutz from the early 1980s (as those are simplest to simulate). Currently in the code are:

SU(2) QCD:
* Initialize 2-color quenched lattices (hot or cold start)
* Generate configurations (with metropolis or heat bath algorithm) with a simple Wilson plaquette action
* Determine the average plaquette
* Set up Wilson-Dirac operator for the quark propagator
* Extract meson correlators (pion, sigma, rho) and masses
* Ensemble analysis with jackknife error estimation
* Determine n x m Wilson loops to extract the string tension (under development)

SU(3) QCD:
* Initialize 3-color quenched lattices (hot or cold start)
* Generate configurations (with metropolis or Cabibbo-heat bath algorithm) with a simple Wilson plaquette action (being verified)
* Determine the average plaquette (being verified)
* Set up Wilson-Dirac operator for the quark propagator (under development)
* Extract the pion correlator for a quenched ensemble (under development)

## Quick Start

```bash
pip install numpy scipy matplotlib jupyter
cd notebooks/
jupyter notebook
```

Start with **Notebook 00** and work through them in order.

## Notebook Progression

| # | Notebook | Topics |
|---|---------|--------|
| 00 | Setup and Imports | Environment check, project structure |
| 01 | SU(2) Group Theory | Cayley-Klein parameterization, group axioms |
| 02 | Lattice Navigation | Coordinates, periodic boundaries, neighbor tables |
| 03 | Gauge Fields and Plaquettes | Links, Wilson loops, gauge action |
| 04 | Monte Carlo and Thermalization | Metropolis algorithm, plaquette vs beta |
| 05 | Wilson-Dirac Operator | Gamma matrices, Clifford algebra, sparsity |
| 06 | Quark Propagators | Point sources, Dirac inversion |
| 07 | Meson Correlators and Masses | Pion/sigma/rho operators, GMOR relation |
| 08 | Analysis of Existing Data | Ensemble averaging, jackknife errors |

All demo calculations use a **4x4x4x4 lattice** and run in seconds.
Notebook 08 analyzes correlator data produced by the batch processing script.

## Repository Structure

```
ugradlattice-main/
  notebooks/            Jupyter notebooks (start here)
    notebook_utils.py   Shared helpers for path setup and plotting
  configs/              Gauge configurations
    sample_4x4x4x4/    Identity and random test configs
    6x6x6x20_b2.40/    50 thermalized configs (beta=2.4)
  scripts/              Shell scripts for long runs
    generate_8x8x8x20.sh     8^3x20 config generation
    run_propagators_batch.sh  Batch correlator processing
  su2/                  Core SU(2) module
    su2.py              Gauge theory, Dirac operator
    pvb.py              Plaquette vs beta
    Thermal_Generator.py  Monte Carlo configuration generator
    jackknife.py        Statistical error analysis
    meson_correlator/   Meson mass extraction
      Propagator.py     Wilson fermion propagator calculator
  su3/                  SU(3) implementation
```

## Generating New Configurations

To generate 8x8x8x20 configurations (runs several hours):

```bash
bash scripts/generate_8x8x8x20.sh
```

To process correlators in batch:

```bash
bash scripts/run_propagators_batch.sh configs/8x8x8x20_b2.40 8 20
```

## Authors (in order of when they contributed)

* Charles Carver
* Sean Hannaford
* Danielle Moynihan
* George Carey
* Jackson Reynolds
* Alfred Ricker
* Anthony Girardi
* Molly Hayes
* Jan Bierowiec
* Zeke Mohammed
