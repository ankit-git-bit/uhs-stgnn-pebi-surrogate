import json
import numpy as np

# Load the analysis
with open('data_analysis.json') as f:
    data = json.load(f)

analysis = data['analysis']

print("\n" + "=" * 100)
print("COMPREHENSIVE DATA INSPECTION SUMMARY")
print("=" * 100)

print(f"\n✓ CASE ANALYSIS: {analysis['first_case']}")
print(f"✓ Total 2D fields: {len(analysis['2d_fields'])}")
print(f"✓ Total 1D fields: {len(analysis['1d_fields'])}")
print(f"✓ Possible X→Y mappings: {len(analysis['x_to_y_mappings'])}")

print("\n" + "-" * 100)
print("1D FIELDS (vectors/parameters - shared across spatial grid):")
print("-" * 100)
for name, shape in sorted(analysis['1d_fields'].items()):
    if shape[0] == 1:
        print(f"  {name:30s} → {str(shape):15s} [Time series scalar or constant]")
    else:
        print(f"  {name:30s} → {str(shape):15s} [Parameter per spatial location]")

print("\n" + "-" * 100)
print("2D FIELDS (matrices - spatial x time grids):")
print("-" * 100)
for name, shape in sorted(analysis['2d_fields'].items()):
    if shape[0] == 1:
        print(f"  {name:30s} → {str(shape):15s} [Single spatial snapshot, 146 time steps]")
    else:
        print(f"  {name:30s} → {str(shape):15s} [10,116 spatial locations, 146 time steps]")

print("\n" + "-" * 100)
print("PRIMARY DIMENSION STRUCTURE:")
print("-" * 100)
print("  Spatial Grid:   10,116 cells (X, Z coordinates)")
print("  Time Steps:     146 time points (5-day intervals = 730 days total)")
print("  Features:       Multiple physical properties (P, Perm, Poro, Sg, Sw, gas compositions, etc.)")

print("\n" + "-" * 100)
print("SAMPLE X → Y MAPPINGS (showing 20 representative mappings):")
print("-" * 100)

# Group mappings by output feature
output_groups = {}
for mapping in analysis['x_to_y_mappings']:
    output = mapping['output']
    if output not in output_groups:
        output_groups[output] = []
    output_groups[output].append(mapping)

count = 0
for output in sorted(output_groups.keys()):
    mappings = output_groups[output]
    if count < 5:  # Show first 5 unique outputs with their mappings
        print(f"\n  OUTPUT: {output} {mappings[0]['output_shape']}")
        # Show 2-3 sample inputs for this output
        for mapping in mappings[:2]:
            print(f"    INPUT:  {mapping['input']:30s} {str(mapping['input_shape']):15s} → "
                  f"OUTPUT: {mapping['output']:30s} {str(mapping['output_shape']):15s}")
        if len(mappings) > 2:
            print(f"    ...and {len(mappings) - 2} more input options for {output}")
        count += 1

print("\n" + "=" * 100)
print("KEY INSIGHTS:")
print("=" * 100)
print("  • Structure: Reservoir simulation output with spatial-temporal evolution")
print("  • 10,116 grid cells × 146 time steps = 1,476,936 spatial-temporal points per field")
print("  • 46 distinct physical properties tracked over simulation period")
print("  • All 2D fields share same (10,116, 146) grid structure → flexible X→Y mapping")
print("  • Primary input candidates: Bact_matrix, Perm_matrix, Poro_matrix, Sg_matrix, Sw_matrix")
print("  • Primary output candidates: P_matrix, xH2_matrix, yH2_matrix, xCO2_matrix, yCO2_matrix, etc.")
print("\n" + "=" * 100)
