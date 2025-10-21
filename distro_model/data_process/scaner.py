import os
from pickle import TRUE 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import open3d as o3d

class kitti_vlp_database:
    def __init__(self, bin_dir):
        self.bin_dir = bin_dir
        self.bin_files = os.listdir(bin_dir); self.bin_files.sort()    
  
        self.num_bins = len(self.bin_files)

# #####################
# global descriptor for LiDAR point cloud
# #####################

class ScanContext:
 
    # static variables 
    viz = 0  # Set to 1 if you want to visualize point clouds
    store_pointcloud = 0  # Set to 1 if you want to store point clouds
    
    downcell_size = 0.2
    
    # Height processing options
    use_full_height_range = 0  # Set to True to capture all heights including negative values
    custom_height_offset = 2.0    # Set custom offset if use_full_height_range is False
    
    # You can uncomment these for multiple resolutions
    # sector_res = np.array([45, 90, 180, 360, 720])
    # ring_res = np.array([10, 20, 40, 80, 160])
    sector_res = np.array([720])
    ring_res = np.array([160])
    max_length = 80
    
     
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
        
        # Downsample point cloud
        downpcd = o3d.geometry.PointCloud.voxel_down_sample(pcd, voxel_size=ScanContext.downcell_size)
        ptcloud_xyz_downed = np.asarray(downpcd.points)
        print("The number of downsampled points: " + str(ptcloud_xyz_downed.shape)) 
    
        if(ScanContext.viz):
            self.visualize_pointcloud(downpcd, f"Point Cloud: {self.bin_file_name}")
            # o3d.visualization.draw_geometries([downpcd])
            
        SCs = []
        for res in range(len(ScanContext.sector_res)):
            num_sector = ScanContext.sector_res[res]
            num_ring = ScanContext.ring_res[res]
            
            # Use the vectorized version for better performance
            sc = self.ptcloud2sc_vectorized(ptcloud_xyz_downed, num_sector, num_ring, ScanContext.max_length)
            SCs.append(sc)
        
        return SCs

    def visualize_pointcloud(self, pcd, window_name):
        """Visualization of point cloud that stays open until manually closed"""
        try:
            # Create a new visualizer
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=window_name, width=800, height=600)
            
            # Add the geometry
            vis.add_geometry(pcd)
            
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
        
        print(f"Saved deep learning image to {image_file}")
        print(f"Saved scan context data to {npy_file}")
    
    print(f"\nProcessed {len(metadata_list)} images for deep learning in {output_dir}")

if __name__ == "__main__":

    os.makedirs("scan_results", exist_ok=True)

    bin_dir = './sample_data/KITTI/00/velodyne/'
    bin_db = kitti_vlp_database(bin_dir)

    metadata = []

    print("\nNumber of bins: ", bin_db.num_bins)
        
    print(f"Height processing: {'Full range (all heights)' if ScanContext.use_full_height_range else f'Custom offset (+{ScanContext.custom_height_offset}m)'}")

    for bin_idx in range(bin_db.num_bins):
        bin_file_name = bin_db.bin_files[bin_idx]
        print(f"\nProcessing file {bin_idx+1}/{bin_db.num_bins}: {bin_file_name}")
        
        # Create a ScanContext object for this bin file
        sc = ScanContext(bin_dir, bin_file_name)
        
        # Print scan context info
        print(f"Number of scan contexts: {len(sc.scancontexts)}")
        print(f"Scan context shape: {sc.scancontexts[0].shape}")

        # print the data of array
        print(f"Scan context data: {sc.scancontexts[0]}")

        output_file = f"scan_results/scan_context_{bin_file_name.split('.')[0]}.png"
        sc.plot_sc(save_path=output_file)

        # Add to metadata
        metadata.append({
            'file': bin_file_name,
            'index': bin_idx,
            'shape': sc.scancontexts[0].shape,
            'data' : sc.scancontexts[0]
        })
    

    # Process the dataset to create images for deep learning training
    process_dataset_to_images(metadata, output_dir='scan_dl_images_second', image_size=(160, 720))
