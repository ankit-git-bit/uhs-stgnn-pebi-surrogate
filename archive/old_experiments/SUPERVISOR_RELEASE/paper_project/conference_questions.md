# Conference Presentation: 20 Defense Questions and Answers

This document prepares the presenter for 20 likely audience and reviewer questions during the Q&A session.

---

## 1. Dataset Scale & Generalizability

### Q1: Why was the training dataset restricted to only five simulation cases?
* **Answer**: The dataset was restricted due to the high computational cost of generating fine-scale, multi-phase reservoir simulations with geochemical and microbiological components. We present this study as a feasibility demonstration for graph-based surrogates on PEBI meshes. We acknowledge that five cases are insufficient for general geological emulators, and Section 12 explicitly details this limitation.

### Q2: How does the surrogate perform on a completely unseen geological model template?
* **Answer**: The model cannot currently generalize to a new geological template. Because the geological mesh topology (10,116 cells) and structural facies boundaries are identical across all five cases, the current GNN acts as a trajectory emulator rather than a generalized physical solver. Extending this to unseen geological layouts requires training on a diverse set of mesh topologies.

### Q3: If you have identical well locations, why not use simple 1D or 2D time-series emulators for the wells instead of modeling the entire 10,116-cell mesh?
* **Answer**: While well-scale pressure forecasting is useful, it does not provide the spatial distribution of pressure and gas saturation. Spatial pressure distributions are critical for geomechanical safety (detecting caprock fracture risk) and gas saturation distributions are necessary to track the hydrogen-water displacement front and gas recovery factor. Pure well-scale emulators lose this spatial resolution.

### Q4: How would the GNN perform if the injector and producer well locations were swapped?
* **Answer**: The model would fail to predict the dynamics accurately because well locations are static in the current dataset, meaning the GNN has only learned the pressure drawdown and buildup profiles relative to those specific cell centroids. Swapping well locations requires a dataset where well coordinates are systematically varied during training.

### Q5: Did you test the surrogate on permeability fields with different geostatistical correlation lengths?
* **Answer**: No. Case 5 was generated using a permeability realization drawn from the same variogram model (correlation range, nugget, orientation) as Cases 1--4. We have not tested the model on permeability fields representing different structural facies or depositional environments, which is listed as a major limitation in Section 12.

---

## 2. Operator Learning & Benchmarks

### Q6: Why did you not benchmark the GNN against the Fourier Neural Operator (FNO)?
* **Answer**: FNO requires structured Cartesian grids to compute the Fast Fourier Transform (FFT) in the frequency domain. Projecting our irregular PEBI mesh onto a Cartesian grid introduces significant truncation errors near the wellbores (RMSE up to 11.42 bar). Applying FNO directly to PEBI grids requires conformal mesh parameterization, which was outside the scope of this work. We explicitly state in Section 10.5 that no conclusions regarding relative performance against FNO should be drawn.

### Q7: Why not benchmark against DeepONet, which is mesh-independent?
* **Answer**: DeepONet evaluates predictions at arbitrary query coordinates but still requires a structured representation of the input functions (sensors). For irregular PEBI meshes, representing the input permeability and pressure fields at fixed sensor locations near complex wells is mathematically challenging. Graph Neural Operators (GNOs) are a more natural fit and will be evaluated in future work.

### Q8: What is the advantage of a GNN over interpolating the PEBI mesh to a very fine Cartesian grid and using a standard CNN?
* **Answer**: Fine-grid interpolation (e.g. $256 \times 128$) reduces truncation errors (RMSE = 8.59 bar) but introduces substantial memory and computational overhead. CNNs would spend compute on inactive grid cells outside the reservoir boundary. The GNN operates directly on active cells, saving computational resources and preserving the exact transmissibility interfaces.

---

## 3. Target Formulation & Feature Engineering

### Q9: Why does predicting pressure change ($\Delta P$) perform so much worse in terms of $R^2$ than predicting absolute pressure ($P_{t+1}$)?
* **Answer**: The drop in $R^2$ is not a sign of poorer model performance, but of a more honest metric. Direct pressure prediction is dominated by temporal autocorrelation, allowing a zero-parameter copy model ($\hat{P}_{t+1} = P_t$) to achieve $R^2 = 0.9931$. Predicting $\Delta P$ removes this autocorrelation floor, forcing the model to learn the physical flow derivatives.

### Q10: Why does the Random Forest model ($R^2 = 0.9827$, RMSE = 0.5487 bar) outperform the ST-GNN v2 in your results?
* **Answer**: These metrics are not directly comparable. The Random Forest is evaluated in a one-step-ahead prediction mode under teacher forcing, where it consumes the true, uncorrupted state at every step. The ST-GNN is evaluated in a 142-step fully autoregressive rollout mode, where it must feed its own predictions back as inputs. Rollout forecasting is a significantly harder task vulnerable to error propagation.

### Q11: What is the physical meaning behind the Gini importance of the $\Delta P_{t-1}$ lag feature (64.65%)?
* **Answer**: In transient pressure diffusion (Eq. 3), the pressure change at a cell is proportional to the net fluid flux. Within a given operating phase, the flux evolves smoothly, meaning $\Delta P_{t-1}$ acts as a local proxy for the spatial pressure gradient and diffusivity, allowing a local model to approximate the exponential pressure decay.

### Q12: Explain the "spatial grid factorization bug" and its implications.
* **Answer**: The preprocessing code attempted to reshape the 10,116-cell PEBI mesh into a 2D structured array by integer factorization. Because the grid is irregular and does not factor cleanly, the code fell back to treating the cells as a 1D vector. This caused finite-difference spatial gradient calculations to return zero, suppressing the spatial signal. It highlights that structured-grid assumptions can silently corrupt spatial features in GNN pipelines.

---

## 4. GNN Architecture & Rollout Stability

### Q13: Why did you incorporate Layer Normalization in the graph convolution layers?
* **Answer**: GNN message passing accumulates features from neighboring nodes. In deep stacks (4 layers), this causes feature representation scale growth, leading to gradient explosion. Applying LayerNorm after each convolution stabilizes the feature variance and enables stable gradient flow.

### Q14: What is the role of scheduled sampling in your training curriculum?
* **Answer**: Scheduled sampling bridges the gap between training (teacher forcing) and inference (autoregressive rollout). By progressively exposing the GNN to its own predictions during training (probability $p_{\text{rollout}} \to 0.5$), the model learns to recover from small prediction errors, mitigating rollout drift.

### Q15: Why does the pressure RMSE grow to a peak of 59.19 bar at step 50 before recovering?
* **Answer**: Step 50 occurs near the transition from active injection to shut-in. During this phase transition, the reservoir undergoes rapid pressure redistribution (shut-in equilibration). The GNN struggles to capture this sharp transient change, leading to a peak in error. The error "recovers" later as the reservoir reaches a stabilized, uniform pressure state.

### Q16: Why did you clip gas saturation predictions to $[0.0, 1.0]$?
* **Answer**: Purely data-driven neural networks carry no physical constraints and can predict negative saturations or values greater than 1.0, which violates basic physics. Clipping maintains physical feasibility and ensures the water saturation closure relation ($S_w = 1 - S_g$) remains valid.

---

## 5. Physical Consistency & Future Work

### Q17: Your model has a peak pressure RMSE of 59.19 bar. Isn't this too large for practical reservoir management?
* **Answer**: Yes. As discussed in Section 11, an error of 59.19 bar (49% of the operating mean) is unacceptable for safety-critical operations like geomechanical caprock integrity monitoring. However, the surrogate runs in milliseconds and is suitable for rapid, qualitative screening of operating schedules.

### Q18: How do you plan to enforce mass conservation in future versions of the model?
* **Answer**: We plan to incorporate physics-informed loss terms (PINN-style residuals) that penalize mass balance violations, or integrate hard physical constraints directly into the GNN decoder, forcing the sum of predicted cell-wise mass changes to match the well injection/production rates.

### Q19: How does the model account for geochemical reactions, which are critical in UHS?
* **Answer**: The current surrogate does not model geochemical reactions. However, the GNN node feature vector contains hydrogen mole fractions in liquid and vapor phases. In future work, we can append concentration matrices of active chemical species to the node features and include reaction kinetics in the loss function.

### Q20: What is the computational speedup achieved by the ST-GNN compared to the ECLIPSE simulator?
* **Answer**: ECLIPSE requires several minutes to solve the coupled non-linear systems of equations for a single Case 5 run. The ST-GNN v2 performs the 142-step rollout in under 100 milliseconds, representing a computational speedup of approximately 4 to 5 orders of magnitude.
