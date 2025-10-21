import os
import numpy as np
import open3d as o3d
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

"""
Scan Context Point Cloud Reconstructor

This script reconstructs point clouds from scan context (.npy) files and visualizes them
with covered cells from JSON files highlighted in red.

Height Processing Options:
- Set use_full_height_range = 1 to capture all heights including negative values
- Set use_full_height_range = 0 to use custom height offset (default: +2.0m)

JSON Processing Options:
- Set load_json_covered_cells = 1 to load and display covered cells from JSON files (red points)
- Set load_json_covered_cells = 0 to disable JSON loading and only show reconstructed scan context

These settings must match the original scaner_version2.py configuration used to generate the .npy files.
"""

class ScanContextReconstructor:
    """
    Reconstructs point clouds from scan context (.npy) files and visualizes them
    with covered cells from JSON files highlighted in red.
    """
    
    # Height processing options (must match scaner_version2.py)
    use_full_height_range = 1  # Set to 1 to capture all heights including negative values
    custom_height_offset = 2.0    # Set custom offset if use_full_height_range is False
    
    # JSON processing options (must match scaner_version2.py)
    load_json_covered_cells = 1  # Set to 1 to load and display covered cells from JSON, 0 to disable
    
    def __init__(self, scan_context_dir='test_data'):
        self.scan_context_dir = scan_context_dir
        self.pedestrian_json_dir = './test_data/'
        
        # Scan context parameters (must match the original generation)
        self.sector_res = 720
        self.ring_res = 160
        self.max_length = 20
            
    def scan_context_to_pointcloud(self, sc_data):
        R, S = self.ring_res, self.sector_res
        ring_idx = np.arange(R)[:, None]
        sector_idx = np.arange(S)[None, :]

        gap_ring   = self.max_length / R
        gap_sector = 360.0 / S

        r = (ring_idx + 0.5) * gap_ring
        theta = np.deg2rad((sector_idx + 0.5) * gap_sector)

        X = r * np.cos(theta)
        Y = r * np.sin(theta)
        Z = sc_data.copy()

        if self.use_full_height_range:
            mask = (Z != 0)
        else:
            mask = (Z > 0)
            Z[mask] = Z[mask] - self.custom_height_offset

        points = np.column_stack((X[mask], Y[mask], Z[mask]))
        colors = np.full((points.shape[0], 3), 0.7, dtype=float)  # gray

        # Keep the ring/sector indices for each emitted point:
        bins_ring = np.repeat(np.arange(R)[:, None], S, axis=1)[mask]
        bins_sector = np.repeat(np.arange(S)[None, :], R, axis=0)[mask]
        # Pack as a 1D “linear bin id” to quickly lookup later
        bin_ids = (bins_ring * S + bins_sector).astype(np.int32)

        return points, colors, bin_ids


    
    def load_covered_cells_from_json(self, json_path):
        """
        Load covered cells from pedestrian JSON file.
        Returns a list of [ring_idx, sector_idx] pairs.
        """
        try:
            with open(json_path, 'r') as f:
                peds = json.load(f)
            
            covered_cells = []
            if isinstance(peds, list):
                for ped in peds:
                    cells = ped.get('covered_cells', [])
                    covered_cells.extend(cells)
            
            return covered_cells
        except FileNotFoundError:
            print(f"[WARN] Pedestrian JSON not found: {json_path}")
            return []
        except Exception as e:
            print(f"[WARN] Failed to load JSON {json_path}: {e}")
            return []
    
    def add_covered_cells_to_pointcloud(self, sc_data, covered_cells):
        if not covered_cells:
            return np.empty((0,3)), np.empty((0,3))

        idx = np.array(covered_cells, dtype=int)
        idx[:,0] = np.clip(idx[:,0], 0, self.ring_res-1)
        idx[:,1] = np.clip(idx[:,1], 0, self.sector_res-1)

        gap_ring   = self.max_length / self.ring_res
        gap_sector = 360.0 / self.sector_res

        r     = (idx[:,0] + 0.5) * gap_ring
        theta = np.deg2rad((idx[:,1] + 0.5) * gap_sector)

        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = sc_data[idx[:,0], idx[:,1]]

        if not self.use_full_height_range:
            valid = z > 0
            x, y, z = x[valid], y[valid], z[valid]
            z = z - self.custom_height_offset

        points = np.column_stack((x, y, z))
        colors = np.tile(np.array([[1.0, 0.0, 0.0]]), (points.shape[0], 1))
        print(f"Added {points.shape[0]} covered cell points (red)")
        return points, colors

    def visualize_pointcloud_with_covered_cells(self, sc_file_path, json_file_path=None):
        """
        Visualize reconstructed point cloud with covered cells highlighted in red.
        Paints existing base points that belong to covered bins instead of creating new points.
        """
        # Load scan context data
        sc_data = np.load(sc_file_path)
        print(f"Loaded scan context shape: {sc_data.shape}")

        # Reconstruct base point cloud (must return base_points, base_colors, base_bin_ids)
        base_points, base_colors, base_bin_ids = self.scan_context_to_pointcloud(sc_data)
        print(f"Reconstructed {len(base_points)} base points")

        # Optionally recolor covered cells
        red_count = 0
        if bool(ScanContextReconstructor.load_json_covered_cells) and json_file_path and os.path.exists(json_file_path):
            covered_cells = self.load_covered_cells_from_json(json_file_path)
            print(f"Found {len(covered_cells)} covered cells in JSON")

            if covered_cells:
                # Convert to ndarray and clip to valid ranges
                covered_ids = np.array(covered_cells, dtype=int).reshape(-1, 2)
                covered_ids[:, 0] = np.clip(covered_ids[:, 0], 0, self.ring_res - 1)   # ring_idx
                covered_ids[:, 1] = np.clip(covered_ids[:, 1], 0, self.sector_res - 1) # sector_idx

                # Map (ring, sector) -> linear bin id to compare against base_bin_ids
                covered_bin_ids = (covered_ids[:, 0] * self.sector_res + covered_ids[:, 1]).astype(np.int32)

                # Mask base points that belong to covered bins and paint them red
                covered_mask = np.isin(base_bin_ids, covered_bin_ids)
                red_count = int(covered_mask.sum())
                if red_count > 0:
                    base_colors[covered_mask] = np.array([1.0, 0.0, 0.0])
                    print(f"Painted {red_count} base points red (covered cells)")
                else:
                    print("No base points matched covered cells (check shape/indexing/transposes).")
        elif not bool(ScanContextReconstructor.load_json_covered_cells):
            print("JSON covered cells loading is disabled")
        else:
            print(f"[WARN] JSON path missing or not found: {json_file_path}")

        # Build Open3D geometry
        base_pcd = o3d.geometry.PointCloud()
        base_pcd.points = o3d.utility.Vector3dVector(base_points)
        base_pcd.colors = o3d.utility.Vector3dVector(base_colors)

        # Coordinate axes scaled to scene
        axis_size = max(0.1, float(self.max_length) * 0.15)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=axis_size)

        # Visualize
        file_name = os.path.basename(sc_file_path)
        window_name = f"Reconstructed Point Cloud: {file_name}"

        total_points = len(base_points)
        print(f"Total points being displayed: {total_points} "
            f"({total_points - red_count} gray + {red_count} red)")
        print(f"Displaying reconstructed point cloud for {file_name}")
        print("Gray points: Reconstructed scan context")
        if bool(ScanContextReconstructor.load_json_covered_cells):
            print("Red points: Covered cells from JSON (painted onto base points)")
        else:
            print("JSON covered cells display is disabled")
        print("Close the window to continue...")

        o3d.visualization.draw_geometries(
            [base_pcd, coordinate_frame],
            window_name=window_name,
            width=1200,
            height=800
        )

    
    def process_all_files(self):
        """
        Process all .npy files in the scan context directory.
        """
        if not os.path.exists(self.scan_context_dir):
            print(f"Scan context directory not found: {self.scan_context_dir}")
            return
        
        # Find all .npy files
        npy_files = [f for f in os.listdir(self.scan_context_dir) if f.endswith('.npy')]
        npy_files.sort()
        
        print(f"Found {len(npy_files)} scan context files to process")
        
        for npy_file in npy_files:
            # Extract the base name (e.g., 'sc_000840.npy' -> '000840')
            base_name = npy_file.replace('sc_', '').replace('.npy', '')
            
            sc_file_path = os.path.join(self.scan_context_dir, npy_file)
            json_file_path = os.path.join(self.pedestrian_json_dir, f"pedestrians_{base_name}.json")
            
            print(f"\nProcessing: {npy_file}")
            print(f"Looking for JSON: pedestrians_{base_name}.json")
            
            self.visualize_pointcloud_with_covered_cells(sc_file_path, json_file_path)


def main():
    """
    Main function to run the point cloud reconstruction and visualization.
    """
    print("=== Scan Context Point Cloud Reconstructor ===")
    print("This script reconstructs point clouds from scan context (.npy) files")
    print("and highlights covered cells from JSON files in red.\n")
    
    print(f"Height processing: {'Full range (all heights)' if ScanContextReconstructor.use_full_height_range else f'Custom offset (+{ScanContextReconstructor.custom_height_offset}m)'}")
    print(f"JSON covered cells: {'Enabled' if ScanContextReconstructor.load_json_covered_cells else 'Disabled'}")
    print(f"Data directory: test_data/")
    
    # Initialize reconstructor
    reconstructor = ScanContextReconstructor()
    
    # Process all files
    reconstructor.process_all_files()
    
    print("\n=== Reconstruction Complete ===")


if __name__ == "__main__":
    main()
