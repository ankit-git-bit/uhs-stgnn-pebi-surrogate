import json
import numpy as np

# Load the analysis
with open('data_analysis.json') as f:
    data = json.load(f)

analysis = data['analysis']

print("\n" + "=" * 120)
print("DATA DIMENSION & MAPPING REPORT")
print("=" * 120)

# Section 1: Dimension Summary
print("\n📊 DIMENSION SUMMARY")
print("-" * 120)
print(f"  • Spatial Grid:        10,116 cells (indexed by X_coords, Z_coords)")
print(f"  • Time Points:         146 steps (5-day intervals over 730 days)")
print(f"  • Total Observations:  10,116 × 146 = 1,476,936 per field")
print(f"  • Distinct Fields:     46 physical properties tracked")
print(f"  • Total Data Points:   46 × 1,476,936 ≈ 68M values per case")

# Section 2: Field Types
print("\n📋 FIELD CLASSIFICATION")
print("-" * 120)

spatial_fields_2d = {k: v for k, v in analysis['2d_fields'].items() if v[0] == 10116}
param_fields_1d = {k: v for k, v in analysis['2d_fields'].items() if v[0] == 1 and v[1] == 146}
scalar_fields = {k: v for k, v in analysis['2d_fields'].items() if v == [1, 1]}
geom_fields = {k: v for k, v in analysis['2d_fields'].items() if v[1] == 1}

print(f"\n  [TYPE A] SPATIAL-TEMPORAL MATRICES (10,116 × 146):")
print(f"  ─ {len(spatial_fields_2d)} fields representing physical state at each grid cell over time")
spatial_list = [f"      • {k}" for k in sorted(spatial_fields_2d.keys())]
for item in spatial_list[:8]:
    print(item)
if len(spatial_list) > 8:
    print(f"      ...and {len(spatial_list) - 8} more")

print(f"\n  [TYPE B] TIME SERIES SCALARS (1 × 146):")
print(f"  ─ {len(param_fields_1d)} fields: well parameters & global properties evolving over time")
param_list = sorted(param_fields_1d.keys())
for k in param_list[:10]:
    print(f"      • {k}")
if len(param_list) > 10:
    print(f"      ...and {len(param_list) - 10} more")

print(f"\n  [TYPE C] SCALAR VALUES (1 × 1):")
print(f"  ─ {len(scalar_fields)} fields: case summary metrics")
for k in sorted(scalar_fields.keys()):
    print(f"      • {k}")

print(f"\n  [TYPE D] GEOMETRY (10,116 × 1):")
print(f"  ─ {len(geom_fields)} fields: static grid coordinates")
for k in sorted(geom_fields.keys()):
    print(f"      • {k}")

# Section 3: X → Y Mappings
print("\n🔗 X → Y MAPPING ANALYSIS")
print("-" * 120)

# Count mappings by dimension
mapping_types = {}
for m in analysis['x_to_y_mappings']:
    key = f"({m['input_shape'][0]}, {m['input_shape'][1]}) → ({m['output_shape'][0]}, {m['output_shape'][1]})"
    if key not in mapping_types:
        mapping_types[key] = 0
    mapping_types[key] += 1

print("\n  Mapping Types:")
for key in sorted(mapping_types.keys(), reverse=True):
    count = mapping_types[key]
    print(f"    {key:50s} : {count:4d} possible mappings")

# Section 4: Most Likely X → Y Pairs
print("\n⭐ PRIMARY X → Y MAPPING OPTIONS")
print("-" * 120)

# Find (10116, 146) → (10116, 146) mappings
matrix_to_matrix = [m for m in analysis['x_to_y_mappings'] 
                    if m['input_shape'] == [10116, 146] and m['output_shape'] == [10116, 146]]

# Group by output
output_to_inputs = {}
for m in matrix_to_matrix:
    output = m['output']
    if output not in output_to_inputs:
        output_to_inputs[output] = []
    output_to_inputs[output].append(m['input'])

print(f"\n  (10,116 × 146) → (10,116 × 146) Mappings: {len(matrix_to_matrix)} options\n")

# Show key output variables and their inputs
key_outputs = ['P_matrix', 'Sg_matrix', 'Sw_matrix', 'Perm_matrix', 'Poro_matrix', 
               'xH2_matrix', 'xCO2_matrix', 'yH2_matrix', 'yCO2_matrix', 'yC1_matrix']

for output in key_outputs:
    if output in output_to_inputs:
        inputs = output_to_inputs[output]
        print(f"  OUTPUT: {output}")
        # Show top 3 inputs
        for inp in inputs[:3]:
            print(f"    ← INPUT: {inp}")
        if len(inputs) > 3:
            print(f"    ...and {len(inputs) - 3} more inputs available")
        print()

print("\n" + "=" * 120)
print("RECOMMENDATIONS")
print("=" * 120)
print("""
  1. FLEXIBLE X → Y MAPPINGS:
     • All 46 fields share compatible dimensions → any field can serve as input or output
     • Strongest mappings: (10,116 × 146) → (10,116 × 146) for spatial-temporal relationships
  
  2. TYPICAL ML SETUP:
     • INPUTS (time t):  Geological properties (Perm, Poro, Bact) + Previous state (Sg, Sw)
     • OUTPUTS (time t): Fluid phase distributions (Sg, Sw) OR compositions (xH2, yH2, etc.)
  
  3. FOR YOUR MODEL:
     • 10,116 spatial cells = independent prediction locations
     • 146 time steps = sequence length (can be used for temporal models like LSTM)
     • 46 fields = potential input + output feature space
""")
print("=" * 120 + "\n")
