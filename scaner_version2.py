import os
from pickle import TRUE 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import open3d as o3d
import json


class kitti_vlp_database:
    def __init__(self, bin_dir):
        self.bin_dir = bin_dir
        # Only get .bin files, filter out other file types like .json
        all_files = os.listdir(bin_dir)
        self.bin_files = [f for f in all_files if f.endswith('.bin')]
        self.bin_files.sort()    
  
        self.num_bins = len(self.bin_files)

# #####################
# global descriptor for LiDAR point cloud
# #####################

class ScanContext:
 
    # static variables 
    viz = 1  # Set to 1 if you want to visualize point clouds
    store_pointcloud = 0  # Set to 1 if you want to store point clouds
    
    # Downsampling options
    use_downsampling = 0  # Set to 1 to enable downsampling, 0 to disable
    downcell_size = 0.2
    
    # Height processing options
    use_full_height_range = 1  # Set to True to capture all heights including negative values
    custom_height_offset = 2.0    # Set custom offset if use_full_height_range is False
    
    # JSON processing options
    load_json_covered_cells = 0  # Set to 1 to load and display covered cells from JSON, 0 to disable
    
    # You can uncomment these for multiple resolutions
    # sector_res = np.array([45, 90, 180, 360, 720])
    # ring_res = np.array([10, 20, 40, 80, 160])
    sector_res = np.array([720])
    ring_res = np.array([160])
    max_length = 20
    
    def __init__(self, bin_dir, bin_file_name):
        self.bin_dir = bin_dir
        self.bin_file_name = bin_file_name
        self.bin_path = bin_dir + bin_file_name

        self.scancontexts = self.genSCs()
        
        
    def load_velo_scan(self):
        scan = np.fromfile(self.bin_path, dtype=np.float32)
        scan = scan.reshape((-1, 4))
        ptcloud_xyz = scan[:, :-1]
        
        return ptcloud_xyz
    
    def ptcloud2sc_vectorized(self, ptcloud, num_sector, num_ring, max_length):
        """
        Convert point cloud to scan context using vectorized operations
        Flexible height processing: can capture all heights or use custom offset
        """
        # Calculate parameters
        gap_ring = max_length / num_ring
        gap_sector = 360 / num_sector
        
        # Get coordinates with flexible height processing
        x = ptcloud[:, 0]
        y = ptcloud[:, 1]
        
        if bool(ScanContext.use_full_height_range):
            # Use original heights to capture all points including ground
            z = ptcloud[:, 2]
        else:
            # Apply custom height offset
            z = ptcloud[:, 2] + ScanContext.custom_height_offset
        
        # Replace zeros to avoid division by zero
        x = np.where(x == 0, 0.001, x)
        y = np.where(y == 0, 0.001, y)
        
        # Calculate polar coordinates
        theta = np.rad2deg(np.arctan2(y, x)) % 360
        r = np.sqrt(x*x + y*y)
        
        # Calculate indices
        ring_idx = np.minimum(r // gap_ring, num_ring-1).astype(np.int32)
        sector_idx = (theta // gap_sector).astype(np.int32)
        
        # Initialize scan context matrix based on height processing mode
        if bool(ScanContext.use_full_height_range):
            # Use negative infinity to capture all heights including negative values
            sc = np.full((num_ring, num_sector), -np.inf)
        else:
            # Use zeros for custom offset mode
            sc = np.zeros((num_ring, num_sector))
        
        # Use numpy's advanced indexing to find max height per cell
        for i in range(len(r)):
            # Update with the maximum height value
            if z[i] > sc[ring_idx[i], sector_idx[i]]:
                sc[ring_idx[i], sector_idx[i]] = z[i]
        
        # Clean up empty cells
        if bool(ScanContext.use_full_height_range):
            # Replace -inf with 0 for empty cells
            sc = np.where(sc == -np.inf, 0, sc)
        
        return sc
    
    def genSCs(self):
        ptcloud_xyz = self.load_velo_scan()
        print("The number of original points: " + str(ptcloud_xyz.shape)) 
    
        # Create PointCloud object using o3d namespace
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ptcloud_xyz)
        
        # Downsample point cloud (if enabled)
        if bool(ScanContext.use_downsampling):
            downpcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=ScanContext.downcell_size)
            ptcloud_xyz_downed = np.asarray(downpcd.points)
            print("The number of downsampled points: " + str(ptcloud_xyz_downed.shape))
        else:
            downpcd = pcd  # Use original point cloud without downsampling
            ptcloud_xyz_downed = ptcloud_xyz
            print("Downsampling disabled - using original point cloud") 
    
        # Visualize raw point cloud before scan context processing (if enabled)
        if(ScanContext.viz):
            # Print raw point cloud info
            print(f"\n=== RAW POINT CLOUD VISUALIZATION ===")
            print(f"File: {self.bin_file_name}")
            print(f"Points: {len(ptcloud_xyz_downed)}")
            print(f"Downsampling: {'Enabled' if ScanContext.use_downsampling else 'Disabled'}")
            print(f"Height processing: {'Full range' if ScanContext.use_full_height_range else 'Custom offset'}")
            print("=====================================\n")
            
            # Create visualization of raw point cloud
            raw_viz_name = f"Raw Point Cloud: {self.bin_file_name}"
            raw_viz_name += f" | Points: {len(ptcloud_xyz_downed)}"
            if ScanContext.use_downsampling:
                raw_viz_name += f" | Downsampled: {ScanContext.downcell_size}m"
            else:
                raw_viz_name += " | No Downsampling"
            
            self.visualize_pointcloud(downpcd, raw_viz_name)
    
        SCs = []
        for res in range(len(ScanContext.sector_res)):
            num_sector = ScanContext.sector_res[res]
            num_ring = ScanContext.ring_res[res]
            
            # Use the vectorized version for better performance
            sc = self.ptcloud2sc_vectorized(ptcloud_xyz_downed, num_sector, num_ring, ScanContext.max_length)
            SCs.append(sc)
        
        # Print scan context processing summary (without visualization)
        if(ScanContext.viz):
            print(f"\n=== SCAN CONTEXT PROCESSING SUMMARY ===")
            print(f"File: {self.bin_file_name}")
            print(f"Original points: {len(ptcloud_xyz)}")
            print(f"Processed points: {len(ptcloud_xyz_downed)}")
            print(f"Scan context resolution: {SCs[0].shape[0]} rings x {SCs[0].shape[1]} sectors")
            print(f"Downsampling: {'Enabled' if ScanContext.use_downsampling else 'Disabled'}")
            print(f"Height processing: {'Full range' if ScanContext.use_full_height_range else 'Custom offset'}")
            print(f"Max range: {ScanContext.max_length}m")
            print(f"Ring spacing: {ScanContext.max_length/SCs[0].shape[0]:.3f}m")
            print(f"Sector spacing: {360/SCs[0].shape[1]:.3f}°")
            print("==========================================\n")
        
        return SCs

    def visualize_pointcloud(self, pcd, window_name):
        """Visualization of point cloud that stays open until manually closed"""
        try:
            # Create a new visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name, width=800, height=600)
            
            # Create coordinate axes
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3.0)
            
            # Add the geometries
            vis.add_geometry(pcd)
            vis.add_geometry(coordinate_frame)
            
            # Set view control
            view_control = vis.get_view_control()
            view_control.set_zoom(0.8)
            
            # Capture screenshot if enabled
            if ScanContext.store_pointcloud:
                image_path = f"pointcloud_{self.bin_file_name.split('.')[0]}.png"
                vis.capture_screen_image(image_path)
                print(f"Saved point cloud image to {image_path}")
            
            # Run the visualizer until the window is closed by the user
            print(f"Displaying point cloud for {self.bin_file_name}. Close the window to continue...")
            vis.run()  # This blocks until the user closes the window
            
            # Once the user closes the window, clean up
            vis.destroy_window()
            del vis
            time.sleep(1.0)
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def scan_context_to_pointcloud(self, sc_data):
        """
        Convert scan context back to point cloud representation.
        This creates a sparse point cloud where each non-zero cell becomes a point.
        """
        points = []
        colors = []
        
        # Calculate parameters
        gap_ring = ScanContext.max_length / ScanContext.ring_res[0]
        gap_sector = 360 / ScanContext.sector_res[0]
        
        # Convert scan context indices back to Cartesian coordinates
        for ring_idx in range(ScanContext.ring_res[0]):
            for sector_idx in range(ScanContext.sector_res[0]):
                height = sc_data[ring_idx, sector_idx]
                
                # Process cells based on height processing mode
                should_process = False
                if bool(ScanContext.use_full_height_range):
                    # In full height range mode, process all non-zero cells (including negative heights)
                    should_process = (height != 0)
                else:
                    # In custom offset mode, only process positive heights
                    should_process = (height > 0)
                
                if should_process:
                    # Convert back to polar coordinates
                    r = (ring_idx + 0.5) * gap_ring  # Use center of ring
                    theta = np.deg2rad((sector_idx + 0.5) * gap_sector)  # Use center of sector
                    
                    # Convert to Cartesian coordinates
                    x = r * np.cos(theta)
                    y = r * np.sin(theta)
                    
                    # Apply height processing based on mode
                    if bool(ScanContext.use_full_height_range):
                        # Use original heights (no offset removal)
                        z = height
                    else:
                        # Remove the custom offset
                        z = height - ScanContext.custom_height_offset
                    
                    points.append([x, y, z])
                    colors.append([0.7, 0.7, 0.7])  # Default gray color
        
        return np.array(points), np.array(colors)
    
    def visualize_scan_context_reconstruction(self):
        """
        Reconstruct scan context back to 3D points and visualize them.
        """
        if not ScanContext.viz:
            return
            
        sc_data = self.scancontexts[0]
        print(f"\n=== SCAN CONTEXT RECONSTRUCTION ===")
        print(f"File: {self.bin_file_name}")
        print(f"Scan context shape: {sc_data.shape}")
        
        # Reconstruct point cloud from scan context
        points, colors = self.scan_context_to_pointcloud(sc_data)
        print(f"Reconstructed {len(points)} points from scan context")
        
        # Create point cloud for visualization
        reconstructed_pcd = o3d.geometry.PointCloud()
        reconstructed_pcd.points = o3d.utility.Vector3dVector(points)
        reconstructed_pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Create visualization
        recon_viz_name = f"Scan Context Reconstruction: {self.bin_file_name}"
        recon_viz_name += f" | Points: {len(points)}"
        recon_viz_name += f" | SC: {sc_data.shape[0]}x{sc_data.shape[1]}"
        
        print("=====================================\n")
        self.visualize_pointcloud(reconstructed_pcd, recon_viz_name)

    def plot_sc(self, save_path=None):
        """Plot scan context image and optionally save to file"""
        plt.figure(figsize=(10, 8))
        plt.imshow(self.scancontexts[0], cmap='viridis')
        # plt.colorbar(label='Height (m)')
        plt.title(f'Scan Context: {self.bin_file_name}')
        plt.xlabel('Sector Index')
        plt.ylabel('Ring Index')
        
        if save_path:
            plt.savefig(save_path)
            print(f"Saved scan context image to {save_path}")

        plt.show()


    def plot_sc_from_json(self, pedestrians_json_path, save_path=None, draw_sc_background=True, cmap='viridis'):
        """
        Plot scan context and overlay ONLY covered_cells from a pedestrian JSON.
        pedestrians_json_path: JSON with a list of dicts, each having 'covered_cells': [[ring, sector], ...]
        """
        # Load SC data
        sc_data = self.scancontexts[0]
        H, W = sc_data.shape  # (num_ring, num_sector)

        # Load JSON
        try:
            with open(pedestrians_json_path, 'r') as f:
                peds = json.load(f)
        except FileNotFoundError:
            print(f"[WARN] Pedestrian JSON not found: {pedestrians_json_path}")
            peds = []
        except Exception as e:
            print(f"[WARN] Failed to load JSON {pedestrians_json_path}: {e}")
            peds = []

        # Prepare figure sized to the SC
        fig = plt.figure(figsize=(W/100.0, H/100.0), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        if draw_sc_background:
            # Normalize for display
            min_val = float(np.min(sc_data))
            max_val = float(np.max(sc_data))
            sc_norm = (sc_data - min_val) / (max_val - min_val) if max_val > min_val else sc_data
            ax.imshow(sc_norm, cmap=cmap, aspect='auto', origin='lower')
        else:
            ax.imshow(np.zeros_like(sc_data), cmap='gray', aspect='auto', origin='lower', vmin=0, vmax=1)

        # Draw ONLY covered_cells
        total_cells = 0
        if isinstance(peds, list):
            for ped in peds:
                cells = ped.get('covered_cells', [])
                if not cells:
                    continue
                arr = np.array(cells, dtype=np.int32)  # shape (K,2) -> [ring, sector]
                if arr.size:
                    rr = np.clip(arr[:, 0], 0, H-1)
                    ss = np.clip(arr[:, 1], 0, W-1)
                    ax.scatter(ss, rr, s=3, c='r', alpha=0.85)
                    total_cells += arr.shape[0]
        else:
            print(f"[WARN] JSON has unexpected format (expected list, got {type(peds)})")

        if save_path:
            plt.savefig(save_path, dpi=100, bbox_inches=None, pad_inches=0)
            print(f"Saved SC with covered_cells overlay to {save_path}")

        plt.close(fig)
        print(f"[INFO] Drawn {total_cells} covered cells from JSON.")

# #######################
# Data preparation
# #######################

# At the end of the for loop, process the dataset for deep learning
def process_dataset_to_images(metadata_list, output_dir='scan_images', image_size=(128, 512), cmap='viridis'):
    """
    Process the scan contexts to create properly formatted images for deep learning
    
    Parameters:
    - metadata_list: List of metadata dictionaries that include scan context data
    - output_dir: Directory to save output images
    - image_size: Size of output images (height, width)
    - cmap: Colormap to use for images
    """
    # Create directory for deep learning images
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each file based on metadata
    for item in metadata_list:
        bin_file_name = item['file']
        
        # Get the scan context data directly from metadata
        sc_data = item['data']
        
        # Save as an image optimized for deep learning input
        image_file = f"{output_dir}/sc_{bin_file_name.split('.')[0]}.png"
        
        # Create a clean figure (no axes, borders, etc.)
        fig = plt.figure(figsize=(image_size[1]/100, image_size[0]/100), dpi=100)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        # Normalize data for better visualization
        min_val = np.min(sc_data)
        max_val = np.max(sc_data)
        if max_val > min_val:
            sc_data_normalized = (sc_data - min_val) / (max_val - min_val)
        else:
            sc_data_normalized = sc_data
        
        # Plot the image with the selected colormap
        ax.imshow(sc_data_normalized, cmap=cmap, aspect='auto')
        
        # Save the image with high quality
        plt.savefig(image_file, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
        # Save scan context data as .npy file in the same directory
        npy_file = f"{output_dir}/sc_{bin_file_name.split('.')[0]}.npy"
        np.save(npy_file, sc_data.astype(np.float32))
        
        # Load and print the length of the saved npy file
        loaded_data = np.load(npy_file)
        print(f"Saved deep learning image to {image_file}")
        print(f"Saved scan context data to {npy_file}")
        print(f"Length of npy file data: {len(loaded_data)}")
        print(f"Shape of npy file data: {loaded_data.shape}")
    
    print(f"\nProcessed {len(metadata_list)} images for deep learning in {output_dir}")


if __name__ == "__main__":

    os.makedirs("scan_results", exist_ok=True)

    bin_dir = './sample_data/'
    bin_db = kitti_vlp_database(bin_dir)
    ped_json_dir = './sample_data/'

    metadata = []

    print("\nNumber of bins: ", bin_db.num_bins)
        
    print(f"Height processing: {'Full range (all heights)' if ScanContext.use_full_height_range else f'Custom offset (+{ScanContext.custom_height_offset}m)'}")
    print(f"Downsampling: {'Enabled (voxel size: ' + str(ScanContext.downcell_size) + 'm)' if ScanContext.use_downsampling else 'Disabled'}")
    print(f"JSON covered cells: {'Enabled' if ScanContext.load_json_covered_cells else 'Disabled'}")

    for bin_idx in range(bin_db.num_bins):
        bin_file_name = bin_db.bin_files[bin_idx]
        stem = os.path.splitext(bin_file_name)[0]
        print(f"\nProcessing file {bin_idx+1}/{bin_db.num_bins}: {bin_file_name}")
        
        # Create a ScanContext object for this bin file
        sc = ScanContext(bin_dir, bin_file_name)
        
        # Print scan context info
        print(f"Number of scan contexts: {len(sc.scancontexts)}")
        print(f"Scan context shape: {sc.scancontexts[0].shape}")

        # print the data of array
        print(f"Scan context data: {sc.scancontexts[0]}")
        # print the length of the scan context data
        print(f"Length of scan context data: {len(sc.scancontexts[0])}")

        # Visualize scan context reconstruction (if enabled)
        sc.visualize_scan_context_reconstruction()

        output_file = f"scan_results/scan_context_{bin_file_name.split('.')[0]}.png"
        sc.plot_sc(save_path=output_file)

        # Try to overlay ONLY covered_cells from pedestrians JSON (if enabled)
        if bool(ScanContext.load_json_covered_cells):
            ped_json_path = os.path.join(ped_json_dir, f"pedestrians_{stem}.json")
            overlay_file = f"scan_results/scan_context_{stem}_covered_cells.png"
            sc.plot_sc_from_json(ped_json_path, save_path=overlay_file, draw_sc_background=True, cmap='viridis')
        else:
            print("JSON covered cells processing is disabled")

        # Add to metadata
        metadata.append({
            'file': bin_file_name,
            'index': bin_idx,
            'shape': sc.scancontexts[0].shape,
            'data' : sc.scancontexts[0]
        })
    

    # Process the dataset to create images for deep learning training
    process_dataset_to_images(metadata, output_dir='scan_dl_images_second', image_size=(160, 720))
