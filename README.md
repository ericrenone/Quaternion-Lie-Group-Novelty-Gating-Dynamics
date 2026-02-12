# Quaternion Lie Group Novelty Gating Dynamics

## Overview

This repository implements a **Lie group adaptive filter** operating on **quaternions (S³)** with **fixed-point Q16.16 arithmetic**, designed to track evolving rotations while detecting **novelty events**. The system combines classical **adaptive filtering**, **Lie group geometry**, and **CORDIC-based fixed-point trigonometry** in a fully visualized, interactive Python framework.

Key features:

* **Quaternion-based Lie group filtering** for rotation tracking.
* **Fixed-point Q16.16 implementation**, demonstrating low-level numerical precision.
* **CORDIC sine/cosine computation** for efficient and hardware-compatible trigonometry.
* **Adaptive gain (α) with exponential decay** for robust tracking.
* **Novelty detection** based on angular distance thresholds.
* **Real-time GUI visualization** and trajectory PCA analysis on S³.

This framework is suitable for **robotics, embedded systems, sensor fusion, and advanced control experiments**.

---

## Core Components

1. **Fixed-Point Arithmetic (Q16.16)**

   * Efficient integer-based calculations.
   * `to_q16`, `from_q16`, `q16_mul`, `q16_div` helpers.

2. **CORDIC Algorithm**

   * Computes sine/cosine iteratively using only shifts and adds.
   * Ensures hardware-efficient trigonometry.

3. **Quaternion Operations**

   * Multiplication, conjugation, dot product, normalization.
   * Exponential map and geodesic angle computation for S³.

4. **Adaptive Filter (`FrameworkProver`)**

   * Tracks quaternion state with **adaptive gain α**.
   * Detects novelty events when angular deviation exceeds threshold.
   * Maintains history for analysis and visualization.

5. **Visualization (`ProofGUI`)**

   * Real-time plotting of angular distance, adaptive gain, and rotation axes.
   * PCA 3D visualization of trajectory on S³.
   * Novelty detection bar chart and axis evolution over time.

---

## Canonical References

* Shannon, C. E. (1948). *A Mathematical Theory of Communication.* Bell System Technical Journal, 27(3,4), 379–423, 623–656.
* Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001). *Estimation with Applications to Tracking and Navigation.* Wiley.
* Diebel, J. (2006). *Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors.* Stanford University Technical Report.
* Tishby, N., Pereira, F. C., & Bialek, W. (1999). *The Information Bottleneck Method.* Proceedings of the 37th Annual Allerton Conference on Communication, Control and Computing.
* Walther, A., et al. (2013). *CORDIC-Based FPGA Implementation of Trigonometric Functions.* IEEE Transactions on VLSI.
* Chirikjian, G. S., & Kyatkin, A. B. (2000). *Engineering Applications of Noncommutative Harmonic Analysis.* CRC Press.

---

## Notes & Recommendations

* Designed for **educational and research purposes**, demonstrating fixed-point Lie group filtering in Python.
* PCA is applied for visualization; note that S³ is non-Euclidean—linear PCA may slightly distort geodesic distances.
* Adjustable parameters:

  * `FX_ETA`: adaptive step size
  * `FX_GATE`: angular threshold for novelty detection
  * `FX_GAMMA`: decay factor for adaptive gain
* Extendable to embedded or real-time systems by porting the Q16/CORDIC implementation to C/C++.

