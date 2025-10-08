Lattice QCD Modular System Documentation
=========================================

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python Version

.. image:: https://img.shields.io/badge/license-MIT-green.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License

A modular, high-performance lattice QCD propagator system for meson mass calculations
using Wilson fermions on discrete spacetime lattices.

Overview
--------

This documentation covers the complete modular lattice QCD system developed for
efficient meson propagator calculations. The system breaks down the monolithic
approach into specialized, optimized modules while maintaining full backward
compatibility with existing command-line interfaces.

**Key Features:**

* **Modular Architecture**: Specialized calculators for pion, rho, and sigma mesons
* **Performance Optimized**: Faster execution through targeted physics implementations
* **CLI Compatible**: Identical command-line interface to original system
* **Comprehensive Testing**: Full validation framework with sample configurations
* **Professional Documentation**: Complete API reference with physics context

Quick Start
-----------

.. code-block:: bash

   # Install dependencies
   pip install numpy scipy matplotlib

   # Calculate pion mass on 4×4×4×4 lattice
   python3 PropagatorModular.py --mass 0.1 --ls 4 --lt 4 --channel pion \
       --input-config sample_inputs/identity_4x4x4x4.pkl --output results_pion

   # Calculate full meson spectrum
   python3 PropagatorModular.py --mass 0.1 --ls 4 --lt 4 --channel all \
       --input-config sample_inputs/identity_4x4x4x4.pkl --output results_spectrum

System Architecture
-------------------

The modular system consists of these core components:

.. code-block:: text

   PropagatorModular.py     ← Main CLI interface (backward compatible)
   ├── MesonIntegration.py  ← Unified coordination layer
   ├── MesonBase.py         ← Shared infrastructure and algorithms
   ├── PionCalculator.py    ← Pseudoscalar meson calculations (J^PC = 0^{-+})
   ├── RhoCalculator.py     ← Vector meson calculations (J^PC = 1^{--})
   ├── SigmaCalculator.py   ← Scalar meson calculations (J^PC = 0^{++})
   └── test_modules.py      ← Comprehensive validation framework

Physics Background
------------------

**Lattice QCD** discretizes spacetime to study quantum chromodynamics non-perturbatively.
This system implements:

* **Wilson Fermions**: Fermion discretization with explicit chiral symmetry breaking
* **SU(2) Gauge Theory**: Simplified QCD with two-color gauge group
* **Meson Spectroscopy**: Hadron mass extraction from correlation functions
* **Point Sources**: Localized fermion sources for propagator calculations

The fundamental quantity calculated is the meson correlator:

.. math::

   C_\\Gamma(t) = \\langle \\bar{\\psi}(x_0) \\Gamma \\psi(x_0) \\cdot \\bar{\\psi}(x_0 + t\\hat{t}) \\Gamma \\psi(x_0 + t\\hat{t}) \\rangle

where :math:`\\Gamma` determines the meson quantum numbers:

* **Pion**: :math:`\\Gamma = \\gamma_5` (pseudoscalar)
* **Rho**: :math:`\\Gamma = \\gamma_i` (vector)
* **Sigma**: :math:`\\Gamma = I` (scalar)

.. toctree::
   :maxdepth: 3
   :caption: API Reference

   api_reference

