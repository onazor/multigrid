# [Codes] Numerical analysis for solving partial differential equations and optimal control problems

This repository contains all the MATLAB codes developed and used in my undergraduate thesis.

## Overview

This project investigates multigrid methods as accelerating basic iterative solvers for large-scale linear systems that arise in:
- **Elliptic and parabolic PDEs**
- **Optimal control problems (OCPs)**, including:
  - Linear OCPs
  - Bilinear OCPs
  - State-constrained bilinear OCPs

---

## Numerical Methods Implemented

### 1. Partial Differential Equations (PDEs)
- **Elliptic PDEs** (Multigrid.m)
  - Finite difference discretization

- **Parabolic PDEs** (Multigrid.m)
  - Implicit time-stepping (e.g., Backward Euler)
  - Space discretization via finite difference
  - Multigrid applied at each time step

### 2. Optimal Control Problems (OCPs)
- **Linear elliptic OCP** (multigrid_LOCP.m)
- **Bilinear elliptic OCP** (multigrid_BLOCP.m)
- **State-constrained bilinear OCP**, solved using:
  - **Lavrentiev-type regularization** (multigrid_LR.m)

---

## Multigrid Techniques

- **Cycles:** V-cycle, W-cycle, F-cycle
- **Smoothers:** Gauss-Seidel, residual-based Newton method smoothers (for bilinear systems)
- **Discretizations:** Structured grids with Dirichlet boundary conditions
- **Collective smoothing** for coupled systems

