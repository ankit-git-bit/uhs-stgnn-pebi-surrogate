# 📊 DATA INSPECTION SUMMARY REPORT

**Generated**: 2026-05-27  
**Location**: `c:\Users\user\Desktop\KTU intern\sciml models`  
**Status**: ✅ COMPLETE

---

## 🎯 EXECUTIVE SUMMARY

Your dataset contains **5 reservoir simulation cases** with **46 physical properties** tracked across a spatial grid over 730 days (146 time steps). This results in **~68 million data points per case**.

**Key Structure**: 
- **10,116 spatial cells** (indexed by X & Z coordinates)
- **146 time steps** (5-day intervals) 
- **Highly flexible X → Y mapping** (570 possible combinations)

---

## 📐 DIMENSION ANALYSIS

### Primary Grid Structure
```
Spatial Dimensions:  10,116 cells (2D grid with X, Z coordinates)
Temporal Dimension:  146 time steps (5-day intervals)
Observation Matrix:  10,116 × 146 = 1,476,936 spatial-temporal points per field
Feature Space:       46 distinct physical properties
Total Data Points:   46 × 1,476,936 ≈ 68,000,000 per case
```

### Coordinates
- **X_coords**: 12.5 to 4987.5 (mean: 2500.0)
- **Z_coords**: 0.25 to 99.75 (mean: 49.72)
- **Time Range**: 5 to 730 days (2-year simulation)

---

## 📋 FIELD CLASSIFICATION

### TYPE A: Spatial-Temporal Matrices (10,116 × 146) — 15 fields
Physical state at **each grid cell over all time steps**:
- `Bact_matrix` - Bacterial concentration
- `P_matrix` - Pressure distribution
- `Perm_matrix` - Permeability evolution
- `Poro_matrix` - Porosity evolution
- `Sg_matrix` - Gas saturation
- `Sw_matrix` - Water saturation
- `xH2_matrix`, `xCO2_matrix` - Liquid-phase compositions
- `yH2_matrix`, `yCO2_matrix`, `yC1_matrix` - Vapor-phase compositions

### TYPE B: Time Series Scalars (1 × 146) — 28 fields
Well parameters and global metrics **varying over time**:
- Injection: `inj_BHP`, `inj_H2`, `inj_PI`, `inj_TotalGas`, `inj_status`
- Production: `prod_BHP`, `prod_H2`, `prod_C1`, `prod_CO2`, `prod_TotalGas`, `prod_Water`, `prod_status`
- Reservoir: `avg_pressure`, `max_pressure`, `min_pressure`
- Mass tracking: `totMassH2_res`, `totMassCO2_res`, `totMassC1_res`, `totMassComp`
- Other: `Z_center_gas`, `time_days`, `dt_days`, `FractionMass*`, etc.

### TYPE C: Scalar Values (1 × 1) — 3 fields
Case summary metrics (constant per case):
- `H2_Efficiency` = 65.8% (average)
- `Total_H2_Injected` = 2,557 units
- `Total_H2_Produced` = 1,683 units

### TYPE D: Geometry (10,116 × 1) — 7 fields
Static spatial properties (one value per cell):
- `X_coords`, `Z_coords` - Grid positions
- `Perm_initial`, `Poro_initial` - Initial rock properties

---

## 🔗 X → Y MAPPING ANALYSIS

### All Possible Mappings: 570

```
Mapping Type                          | Count | Description
--------------------------------------|-------|------------------------------------------
(10,116 × 146) → (10,116 × 146)     |  55   | Spatial-temporal to spatial-temporal
(1 × 146) → (1 × 146)               | 378   | Well parameter to well parameter
(10,116 × 146) → (10,116 × 1)       |  17   | Spatial-temporal to spatial constant
(10,116 × 1) → (10,116 × 146)       |  27   | Spatial constant to spatial-temporal
(1 × 146) → (1 × 1)                 |   9   | Well series to scalar
(1 × 1) → (1 × 146)                 |  75   | Scalar to well series
(10,116 × 1) → (10,116 × 1)         |   6   | Spatial constant to spatial constant
(1 × 1) → (1 × 1)                   |   3   | Scalar to scalar
```

### PRIMARY X → Y MAPPINGS: (10,116 × 146) → (10,116 × 146)

**55 possible combinations** linking spatial-temporal properties:

| Input | Output | Physical Meaning |
|-------|--------|------------------|
| Bact_matrix | P_matrix | Bacterial effect on pressure |
| Bact_matrix | Sg_matrix | Gas saturation evolution |
| Bact_matrix | Perm_matrix | Bacterial clogging/permeability change |
| P_matrix | xH2_matrix | Pressure effect on H₂ liquid composition |
| P_matrix | yH2_matrix | Pressure effect on H₂ vapor composition |
| Perm_matrix | Sw_matrix | Permeability-saturation relationship |
| Poro_matrix | P_matrix | Porosity-pressure coupling |

---

## 💡 DATA CHARACTERISTICS

### Value Ranges (from Case_0001)

| Field | Min | Max | Mean | Notes |
|-------|-----|-----|------|-------|
| Bact_matrix | 0.001 | 5.31 | 0.112 | Bacterial concentration |
| P_matrix | 50.05 | 212.16 | 121.09 | Pressure (bar) |
| Sg_matrix | 0.0 | 0.642 | 0.011 | Gas saturation fraction |
| Sw_matrix | 0.358 | 1.0 | 0.989 | Water saturation fraction |
| xH2_matrix | 1e-8 | 0.00255 | 4.8e-5 | H₂ liquid mole fraction |
| yH2_matrix | 1e-8 | 0.999 | 0.0353 | H₂ vapor mole fraction |
| inj_H2 | 0.0 | 8.53 | 3.50 | Injection rate (daily) |
| prod_H2 | 0.0 | 8.49 | 2.31 | Production rate (daily) |

---

## ⭐ RECOMMENDED X → Y PAIRS FOR MACHINE LEARNING

### Option 1: Fluid State Prediction
**Goal**: Predict phase distributions given geological properties  
**Input** (10,116 × 146):
- `Perm_matrix` (permeability evolution)
- `Poro_matrix` (porosity evolution)
- Well operations: `inj_BHP`, `prod_BHP` (broadcasted to spatial grid)

**Output** (10,116 × 146):
- `Sg_matrix` (gas saturation)
- `Sw_matrix` (water saturation)

### Option 2: Composition Prediction
**Goal**: Predict gas/liquid compositions at each location  
**Input** (10,116 × 146):
- `P_matrix` (pressure)
- `Sg_matrix` (gas saturation)
- `Bact_matrix` (bacterial presence)

**Output** (10,116 × 146):
- `xH2_matrix`, `xCO2_matrix` (liquid compositions)
- `yH2_matrix`, `yCO2_matrix`, `yC1_matrix` (vapor compositions)

### Option 3: Pressure Evolution
**Goal**: Predict pressure from geological properties and initial conditions  
**Input** (10,116 × 146):
- `Perm_initial` (broadcasted to 10,116 × 146)
- `Bact_matrix` (bacterial impact)
- Well operations as time series (1 × 146) broadcasted

**Output** (10,116 × 146):
- `P_matrix` (pressure field)

### Option 4: Well Performance (1D Analysis)
**Goal**: Predict production from injection history  
**Input** (1 × 146):
- `inj_BHP`, `inj_H2`, `inj_TotalGas`

**Output** (1 × 146):
- `prod_H2`, `prod_TotalGas`

---

## 📊 SPATIAL-TEMPORAL STRUCTURE

### For CNN/RNN Architectures:
```
Input Shape:  (batch_size, spatial_height=?, spatial_width=?, time_steps=146, channels=N)
OR
Input Shape:  (batch_size, time_steps=146, spatial_cells=10116, features=N)
```

Your data can be reshaped depending on spatial topology:
- If 10,116 represents a 1D line: `(batch, 146, 10116, N_features)`
- If 10,116 is 2D grid: Need X/Z coordinate mapping

### For Feed-Forward Architectures:
```
Input Shape:  (batch_size, 10116*146=1476936, features)
OR
Input Shape:  (batch_size, time_steps=146, features=10116)
```

---

## 📁 GENERATED ANALYSIS FILES

1. **`data_check.py`** - Quick inspection script
2. **`display_summary.py`** - Field-by-field summary
3. **`detailed_mapping_report.py`** - Complete X→Y mapping analysis
4. **`data_analysis.json`** - Structured metadata (complete statistics for all 5 cases)

### Running Analysis:
```bash
python data_check.py              # Generate data_analysis.json
python display_summary.py         # Show field types
python detailed_mapping_report.py # Show X→Y options
```

---

## 🎓 NEXT STEPS FOR YOUR MODEL

1. **Normalize/standardize** values (e.g., Sg_matrix [0-1], P_matrix varies widely)
2. **Select X→Y pair** based on your physics goal (saturation, composition, or pressure)
3. **Handle spatial topology**:
   - Are the 10,116 cells connected in sequence?
   - Do they form a 2D grid? (Use coordinate data to reconstruct)
4. **Choose architecture**:
   - Physics-informed neural networks (PINNs) for conservation laws
   - Convolutional layers if spatial correlation is important
   - Recurrent layers for temporal evolution
   - Transformer layers for long-range dependencies
5. **Data split strategy**:
   - Use 5 cases: 3 train, 1 validation, 1 test
   - OR: Time-split within cases (early time → train, later time → test)

---

**Report Generated**: 2026-05-27 | **Status**: ✅ All Analyses Complete
