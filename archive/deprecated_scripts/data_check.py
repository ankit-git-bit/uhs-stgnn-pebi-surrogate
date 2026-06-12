import scipy.io as sio
import numpy as np
from pathlib import Path
import json

# Load and inspect all .mat files
data_dir = Path(__file__).parent
mat_files = sorted(data_dir.glob("Case_*.mat"))

info = {
    'cases': {},
    'analysis': {}
}

# Collect info from all cases
for mat_file in mat_files:
    data = sio.loadmat(str(mat_file))
    data = {k: v for k, v in data.items() if not k.startswith('__')}
    
    case_info = {}
    for field_name in sorted(data.keys()):
        field_data = data[field_name]
        shape = field_data.shape
        
        case_info[field_name] = {
            'shape': shape,
            'dtype': str(field_data.dtype),
            'n_elements': int(field_data.size),
        }
        
        if field_data.size > 0:
            case_info[field_name]['min'] = float(np.nanmin(field_data))
            case_info[field_name]['max'] = float(np.nanmax(field_data))
            case_info[field_name]['mean'] = float(np.nanmean(field_data))
    
    info['cases'][mat_file.name] = case_info

# Analyze first case for X->Y mapping
if mat_files:
    first_case = sio.loadmat(str(mat_files[0]))
    first_case = {k: v for k, v in first_case.items() if not k.startswith('__')}
    
    fields_2d = {k: v.shape for k, v in first_case.items() if len(v.shape) == 2}
    fields_1d = {k: v.shape for k, v in first_case.items() if len(v.shape) == 1}
    
    info['analysis']['first_case'] = mat_files[0].name
    info['analysis']['1d_fields'] = fields_1d
    info['analysis']['2d_fields'] = fields_2d
    
    # Find mappings
    mappings = []
    field_names_2d = sorted(fields_2d.keys())
    
    for i, field1 in enumerate(field_names_2d):
        shape1 = fields_2d[field1]
        for field2 in field_names_2d[i+1:]:
            shape2 = fields_2d[field2]
            if shape1[0] == shape2[0]:  # Same first dimension
                mappings.append({
                    'input': field1,
                    'input_shape': shape1,
                    'output': field2,
                    'output_shape': shape2,
                    'n_samples': shape1[0],
                    'input_features': shape1[1] if len(shape1) > 1 else 1,
                    'output_features': shape2[1] if len(shape2) > 1 else 1,
                })
    
    info['analysis']['x_to_y_mappings'] = mappings

# Save as JSON
with open(str(data_dir / 'data_analysis.json'), 'w') as f:
    json.dump(info, f, indent=2)

print("Data analysis saved to data_analysis.json")
print("\nSummary:")
print(f"Cases found: {len(info['cases'])}")
print(f"2D fields in first case: {len(info['analysis']['2d_fields'])}")
print(f"Possible X->Y mappings: {len(info['analysis']['x_to_y_mappings'])}")
print(f"\nTop 10 X->Y mappings (by output dimensions):")
for i, mapping in enumerate(sorted(info['analysis']['x_to_y_mappings'], 
                                   key=lambda x: x['output_features'], 
                                   reverse=True)[:10]):
    print(f"{i+1}. {mapping['input']} {mapping['input_shape']} -> {mapping['output']} {mapping['output_shape']}")

