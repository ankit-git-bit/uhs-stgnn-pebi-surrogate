import scipy.io as sio
import numpy as np
from scipy.spatial import Delaunay, Voronoi
from pathlib import Path

def main():
    print("=" * 80)
    print("PHASE 4.5: NATIVE GRID GRAPH CONSTRUCTION")
    print("=" * 80)
    
    project_root = Path(__file__).resolve().parents[2]
    data_raw_dir = project_root / "data" / "raw"
    data_processed_dir = project_root / "data" / "processed"
    data_processed_dir.mkdir(parents=True, exist_ok=True)
    
    case_path = data_raw_dir / "Case_0001_wj.mat"
    
    if not case_path.exists():
        raise FileNotFoundError(f"Case file not found at {case_path}")
        
    print(f"Loading grid coordinates from: {case_path.name}")
    data = sio.loadmat(str(case_path))
    
    # Correct coordinate keys in MATLAB file
    X = data['X_coords'].squeeze()
    Z = data['Z_coords'].squeeze()
    points = np.column_stack((X, Z))
    n_cells = points.shape[0]
    
    print(f"  Total grid cells: {n_cells}")
    print(f"  X-coordinates range: {X.min():.2f} to {X.max():.2f} meters")
    print(f"  Z-coordinates range: {Z.min():.2f} to {Z.max():.2f} meters")
    
    # 1. Delaunay Triangulation for Connectivity Graph
    print("\nComputing Delaunay Triangulation...")
    tri = Delaunay(points)
    
    # Collect unique undirected edges
    edges_set = set()
    for simplex in tri.simplices:
        for i in range(3):
            u = simplex[i]
            v = simplex[(i+1)%3]
            if u > v:
                u, v = v, u
            edges_set.add((u, v))
    print(f"  Extracted {len(edges_set)} unique edges.")
    
    # 2. Voronoi Tessellation for Contact Face Areas (Ridge Lengths)
    print("\nComputing Voronoi Diagram for ridge lengths...")
    vor = Voronoi(points)
    
    # Map (u, v) pairs to Voronoi ridge length (face area)
    ridge_to_area = {}
    finite_lengths = []
    
    for idx, (u, v) in enumerate(vor.ridge_points):
        vertices = vor.ridge_vertices[idx]
        if -1 not in vertices and len(vertices) == 2:
            # Distance between Voronoi vertices is the shared cell boundary length
            v0 = vor.vertices[vertices[0]]
            v1 = vor.vertices[vertices[1]]
            length = float(np.sqrt(np.sum((v0 - v1)**2)))
            ridge_to_area[(u, v)] = length
            finite_lengths.append(length)
        else:
            # Boundary ridge
            ridge_to_area[(u, v)] = 0.0
            
    # Fill boundary ridges with the average finite ridge length
    avg_area = np.mean(finite_lengths) if finite_lengths else 18.5
    print(f"  Voronoi complete. Average interior shared face area (length): {avg_area:.4f} m")
    
    for k, v in ridge_to_area.items():
        if v == 0.0:
            ridge_to_area[k] = avg_area
            
    # 2.5 Compute Voronoi cell areas (Shoelace formula) for PINN physics loss
    print("\nComputing Voronoi cell areas...")
    cell_areas = np.zeros(n_cells, dtype=np.float32)
    finite_cell_areas = []
    
    for idx in range(n_cells):
        region_idx = vor.point_region[idx]
        region_vertices = vor.regions[region_idx]
        
        # Finite Voronoi regions have no -1
        if -1 not in region_vertices and len(region_vertices) > 2:
            vertices = vor.vertices[region_vertices]
            x = vertices[:, 0]
            y = vertices[:, 1]
            area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            cell_areas[idx] = area
            finite_cell_areas.append(area)
            
    avg_cell_area = np.mean(finite_cell_areas) if finite_cell_areas else 24.0
    print(f"  Average Voronoi cell area: {avg_cell_area:.4f} m^2")
    
    for idx in range(n_cells):
        if cell_areas[idx] == 0.0:
            cell_areas[idx] = avg_cell_area
            
    # 3. Identify well cell locations (verified from flow dynamics)
    # Injector is active in Case 1 cells 4981-4990 (vertical well at X=2450)
    # Producer is active in Case 1 cells 5134-5138 (vertical well at X=2500)
    inj_cells = np.arange(4981, 4991)
    prod_cells = np.arange(5134, 5139)
    
    # Calculate well centers for Euclidean distance features
    inj_center_X = X[inj_cells].mean()
    inj_center_Z = Z[inj_cells].mean()
    prod_center_X = X[prod_cells].mean()
    prod_center_Z = Z[prod_cells].mean()
    
    print(f"\nWell locations identified:")
    print(f"  Injector cells: {inj_cells[0]}-{inj_cells[-1]} | Center: ({inj_center_X:.1f}, {inj_center_Z:.1f}) m")
    print(f"  Producer cells: {prod_cells[0]}-{prod_cells[-1]} | Center: ({prod_center_X:.1f}, {prod_center_Z:.1f}) m")
    
    # Calculate cell-wise well distances
    dist_to_inj = np.sqrt((X - inj_center_X)**2 + (Z - inj_center_Z)**2)
    dist_to_prod = np.sqrt((X - prod_center_X)**2 + (Z - prod_center_Z)**2)
    
    # 4. Construct edge lists and compute transmissibilities
    perm = data['Perm_initial'].squeeze()
    
    # Build bidirectional edge arrays for PyTorch Geometric-style layout (2, 2*n_edges)
    edge_index_u = []
    edge_index_v = []
    edge_dist = []
    edge_area = []
    edge_trans = []
    
    for (u, v) in edges_set:
        # Distance between cell centers
        dist = float(np.sqrt(np.sum((points[u] - points[v])**2)))
        
        # Shared face length
        area = ridge_to_area.get((u, v), avg_area)
        if (v, u) in ridge_to_area:
            area = ridge_to_area[(v, u)]
            
        # Harmonic mean permeability
        ku = perm[u]
        kv = perm[v]
        k_harmonic = 2.0 * ku * kv / (ku + kv + 1e-12)
        
        # Transmissibility: T = k * A / d
        trans = k_harmonic * area / (dist + 1e-12)
        
        # Bidirectional graph edges
        edge_index_u.extend([u, v])
        edge_index_v.extend([v, u])
        
        edge_dist.extend([dist, dist])
        edge_area.extend([area, area])
        edge_trans.extend([trans, trans])
        
    edge_index = np.array([edge_index_u, edge_index_v], dtype=np.int64)
    edge_dist = np.array(edge_dist, dtype=np.float32)
    edge_area = np.array(edge_area, dtype=np.float32)
    edge_trans = np.array(edge_trans, dtype=np.float32)
    
    print(f"\nFinal Graph Structure:")
    print(f"  Number of nodes (cells): {n_cells}")
    print(f"  Number of directed edges: {edge_index.shape[1]}")
    
    # Save graph dataset
    output_path = data_processed_dir / "mesh_graph.npz"
    np.savez(
        output_path,
        edge_index=edge_index,
        edge_dist=edge_dist,
        edge_area=edge_area,
        edge_trans=edge_trans,
        cell_area=cell_areas,
        dist_to_inj=dist_to_inj,
        dist_to_prod=dist_to_prod,
        inj_cells=inj_cells,
        prod_cells=prod_cells,
        X_coords=X,
        Z_coords=Z
    )
    print(f"\n[OK] Saved graph data successfully to: data/processed/{output_path.name}")
    print("=" * 80)

if __name__ == "__main__":
    main()
