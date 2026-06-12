# Future Work Roadmap: UHS GNN Surrogate Modeling

This roadmap details seven prioritized research tasks to address the limitations of the current surrogate model and transition it from a case-specific emulator to a generalized, physics-consistent reservoir simulation tool.

---

## Priority 1: Additional Simulation Cases

* **Objective**: Generate at least 50 additional reservoir simulation cases with varying well operating schedules and geostatistical permeability realizations.
* **Scientific Value**: Resolves the severe training data deficit (currently only 4 training cases). Provides the data diversity required for deep neural networks to learn generalized flow mappings rather than memorizing cyclic schedules.
* **Expected Effort**: **Medium**. Requires setting up a batch execution script using ECLIPSE/tNavigator to run the simulations in parallel on a multi-core workstation. Est. time: 2--3 days.
* **Publication Impact**: **High**. Directly addresses the primary reviewer critique regarding dataset scale and generalizability, making the paper viable for prestigious geosciences journals (e.g. *Computational Geosciences*).

---

## Priority 2: Geological Variability

* **Objective**: Train and test the GNN on multiple geological templates, incorporating faults, structural facies changes, and varied mesh topologies (different cell counts).
* **Scientific Value**: Evaluates the mesh-independence and geological generalizability of the GNN. Proves that the GNN can handle arbitrary graph topologies and transfer learned transport physics to unseen fields.
* **Expected Effort**: **High**. Requires modifying the dataloader to process varying graph sizes and connectivity matrices, and re-running geostatistical generators. Est. time: 2 weeks.
* **Publication Impact**: **High**. Establishes the GNN as a true general-purpose surrogate solver rather than a local cyclic emulator.

---

## Priority 3: FNO Benchmark

* **Objective**: Implement a Fourier Neural Operator (FNO) benchmark. This requires conformal mapping or spatial interpolation of the PEBI mesh onto a high-resolution Cartesian grid ($256 \times 128$) to run FNO, and projecting the predictions back for comparison.
* **Scientific Value**: Provides a comparative benchmark against state-of-the-art operator learning methods, establishing whether graph convolutions are superior to frequency-domain operators on irregular grids.
* **Expected Effort**: **Medium-High**. Requires setting up the grid projection pipeline and tuning the FNO hyperparameters. Est. time: 1 week.
* **Publication Impact**: **Medium-High**. Satisfies machine learning reviewers' demands for modern SciML benchmarks.

---

## Priority 4: DeepONet Benchmark

* **Objective**: Implement a DeepONet benchmark, representing input permeability fields and initial states at fixed sensor locations and query coordinates.
* **Scientific Value**: Evaluates coordinate-based operator learning against graph-based message passing for unstructured geological domains.
* **Expected Effort**: **Medium-High**. Requires setting up sensor coordinate grids and training the Branch and Trunk networks. Est. time: 1 week.
* **Publication Impact**: **Medium-High**. Enriches the comparative analysis section.

---

## Priority 5: Physics-Informed Rollout

* **Objective**: Integrate physics-informed loss terms (PINN-style residuals representing mass conservation and Darcy's law) into the ST-GNN training objective.
* **Scientific Value**: Enforces physical consistency (mass balance) during autoregressive rollouts. Bounding predictions with physical constraints mitigates covariate shift, reducing the 59.19 bar pressure drift and saturation leaks.
* **Expected Effort**: **High**. Requires writing PyTorch differentiable operators representing irregular PEBI cell gradient calculations and transmissibility flux balances. Est. time: 3 weeks.
* **Publication Impact**: **Very High**. Physics-informed emulators are of massive interest to both the reservoir simulation and SciML communities.

---

## Priority 6: Multi-Output Surrogate

* **Objective**: Expand the surrogate target to predict hydrogen concentrations, biological consumption rates, and water saturation changes simultaneously.
* **Scientific Value**: Emulates the full physics of multi-phase flow, transport, and biochemical reactions critical to hydrogen containment and gas quality assessment.
* **Expected Effort**: **Medium**. Dataloader is already configured for 46 variables. Requires expanding the decoder output dimension. Est. time: 3 days.
* **Publication Impact**: **Medium**. Shows multi-physics capability.

---

## Priority 7: Optimization Workflow

* **Objective**: Integrate the trained ST-GNN v2 surrogate into a closed-loop schedule optimization framework (e.g. Bayesian Optimization or Genetic Algorithms) to optimize hydrogen recovery factor under geomechanical pressure constraints.
* **Scientific Value**: Demonstrates the practical utility of surrogate emulators, showing that the milliseconds inference speed enables optimization workflows that are computationally impossible with classical simulators.
* **Expected Effort**: **Medium**. Requires wrapping the PyTorch model inside an optimization loop (e.g. Scipy Optimize or Pygmo). Est. time: 4 days.
* **Publication Impact**: **High** (especially for engineering-focused venues like SPE Journal).
