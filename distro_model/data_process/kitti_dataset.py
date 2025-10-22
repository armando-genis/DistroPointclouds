import sys
import os
import random

# Set environment variables to avoid Qt issues in headless environments
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

import numpy as np
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import open3d as o3d

sys.path.append('../')

from data_process import transformation, kitti_bev_utils, kitti_data_utils
import config.kitti_config as cnf


class ScanContext:
    """Scan Context generation for LiDAR point clouds"""
    
    # Static variables - disabled for headless environment
    viz = 0  # Disabled to avoid GUI issues in headless environments
    store_pointcloud = 0  # Disabled to avoid GUI issues in headless environments
    
    use_downsampling = 0
    downcell_size = 0.2
    
    # Height processing options
    use_full_height_range = 1  # Set to True to capture all heights including negative values
    custom_height_offset = 2.0    # Set custom offset if use_full_height_range is False
    
    # Resolution settings
    sector_res = np.array([720])
    ring_res = np.array([160])
    max_length = 20
    
    def __init__(self, ptcloud_xyz):
        self.ptcloud_xyz = ptcloud_xyz
        self.scancontexts = self.genSCs()
        
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
        ptcloud_xyz = self.ptcloud_xyz
        
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
    

        SCs = []
        for res in range(len(ScanContext.sector_res)):
            num_sector = ScanContext.sector_res[res]
            num_ring = ScanContext.ring_res[res]
            
            # Use the vectorized version for better performance
            sc = self.ptcloud2sc_vectorized(ptcloud_xyz_downed, num_sector, num_ring, ScanContext.max_length)
            SCs.append(sc)
        
        return SCs
    
    @staticmethod
    def pedestrian_to_scan_context_coords(pedestrian_x, pedestrian_y):
        """
        Convert pedestrian LiDAR coordinates to scan context indices using ScanContext parameters
        
        Args:
            pedestrian_x: X coordinate in LiDAR frame (meters)
            pedestrian_y: Y coordinate in LiDAR frame (meters)
        
        Returns:
            ring_idx: Ring index (0-159)
            sector_idx: Sector index (0-719)
        """
        # Use ScanContext parameters
        num_ring = ScanContext.ring_res[0]  # 160
        num_sector = ScanContext.sector_res[0]  # 720
        max_length = ScanContext.max_length  # 80
        
        # Calculate gaps
        gap_ring = max_length / num_ring  # 0.5 meters per ring
        gap_sector = 360 / num_sector     # 0.5 degrees per sector
        
        # Calculate polar coordinates
        theta = np.rad2deg(np.arctan2(pedestrian_y, pedestrian_x)) % 360
        r = np.sqrt(pedestrian_x**2 + pedestrian_y**2)
        
        # Calculate indices
        ring_idx = int(np.minimum(r / gap_ring, num_ring - 1))
        sector_idx = int(theta / gap_sector)
        
        return ring_idx, sector_idx
    
    @staticmethod
    def find_pedestrians_in_scan_context(labels):
        """
        Args:
            labels: ndarray (N,8): [class, x, y, z, h, w, l, yaw] in LiDAR coords
        Returns:
            list of dicts with center indices and full covered_cells
        """
        pedestrian_positions = []
        PEDESTRIAN_ID = cnf.CLASS_NAME_TO_ID.get('Pedestrian', 1)

        print(f"DEBUG: Processing {len(labels)} labels")
        for i, lab in enumerate(labels):
            class_id, x, y, z, h, w, l, yaw = lab
            print(f"DEBUG: Label {i}: class={class_id} (type: {type(class_id)}), "
                f"x={x:.2f}, y={y:.2f}, z={z:.2f}, hwl=({h:.2f},{w:.2f},{l:.2f}), yaw={yaw:.3f}")

            if int(class_id) != PEDESTRIAN_ID:
                continue

            # Center cell for reference
            ring_idx, sector_idx = ScanContext.pedestrian_to_scan_context_coords(x, y)

            # All cells whose centers lie inside the rotated footprint
            covered = ScanContext.cells_covered_by_box(
                x=float(x), y=float(y), l=float(l), w=float(w), yaw_lidar=float(yaw)
            )

            pedestrian_positions.append({
                'class': int(class_id),
                'x': float(x), 'y': float(y), 'z': float(z),
                'h': float(h), 'w': float(w), 'l': float(l), 'yaw': float(yaw),
                'center_ring_idx': int(ring_idx),
                'center_sector_idx': int(sector_idx),
                'covered_cells': covered.tolist(),
                'distance': float(np.hypot(x, y)),
                'angle': float((np.degrees(np.arctan2(y, x)) % 360.0))
            })

        print(f"DEBUG: Total pedestrians found: {len(pedestrian_positions)}")
        return pedestrian_positions


    @staticmethod
    def cells_covered_by_box(x, y, l, w, yaw_lidar,
                             num_ring=None, num_sector=None, max_length=None):
        """
        Return all (ring_idx, sector_idx) cells whose *cell centers* lie inside a
        rotated rectangle (pedestrian footprint) centered at (x,y), size (l,w), yaw in LiDAR frame.

        Args:
            x, y: pedestrian center in LiDAR (meters)
            l, w: box length (along box local X) and width (along box local Y) in meters
            yaw_lidar: rotation (radians) in LiDAR frame (counterclockwise about +Z)
            num_ring, num_sector, max_length: if None, take from ScanContext.* static vars
        Returns:
            np.ndarray of shape (K, 2) with rows [ring_idx, sector_idx]
        """
        if num_ring   is None: num_ring   = int(ScanContext.ring_res[0])
        if num_sector is None: num_sector = int(ScanContext.sector_res[0])
        if max_length is None: max_length = float(ScanContext.max_length)

        gap_ring   = max_length / num_ring
        gap_sector = 360.0 / num_sector

        # centers of SC cells in polar coords
        r_centers     = (np.arange(num_ring) + 0.5) * gap_ring                          # (R,)
        theta_centers = np.deg2rad((np.arange(num_sector) + 0.5) * gap_sector)          # (S,)

        # grid of cell-center coordinates in Cartesian
        R, TH = np.meshgrid(r_centers, theta_centers, indexing='ij')  # shapes (R,S)
        Xc = R * np.cos(TH)
        Yc = R * np.sin(TH)

        # translate to box center
        X = Xc - x
        Y = Yc - y

        # rotate points into box local frame (inverse rotation by yaw)
        c = np.cos(yaw_lidar)
        s = np.sin(yaw_lidar)
        # inverse rotation = R(-yaw): [ [ c, s], [-s, c] ]
        Xl =  c * X + s * Y
        Yl = -s * X + c * Y

        half_l = 0.5 * l
        half_w = 0.5 * w

        inside = (np.abs(Xl) <= half_l) & (np.abs(Yl) <= half_w)  # (R,S) boolean mask

        # Return indices where True
        rr, ss = np.nonzero(inside)  # ring indices, sector indices
        return np.stack([rr, ss], axis=1)  # (K,2)

class KittiDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', lidar_transforms=None, aug_transforms=None, multiscale=False,
                 num_samples=None, mosaic=False, random_padding=False):
        self.dataset_dir = dataset_dir
        assert mode in ['train', 'val', 'test'], 'Invalid mode: {}'.format(mode)
        self.mode = mode
        self.is_test = (self.mode == 'test')
        sub_folder = 'testing' if self.is_test else 'training'

        self.multiscale = multiscale
        self.lidar_transforms = lidar_transforms
        self.aug_transforms = aug_transforms
        self.img_size = cnf.BEV_WIDTH
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.mosaic = mosaic
        self.random_padding = random_padding
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]

        self.lidar_dir = os.path.join(self.dataset_dir, sub_folder, "velodyne")
        self.image_dir = os.path.join(self.dataset_dir, sub_folder, "image_2")
        self.calib_dir = os.path.join(self.dataset_dir, sub_folder, "calib")
        self.label_dir = os.path.join(self.dataset_dir, sub_folder, "label_2")
        
        # Create scan context output directories
        self.scan_context_dir = os.path.join(self.dataset_dir, sub_folder, "scan_context")
        os.makedirs(self.scan_context_dir, exist_ok=True)
        
        split_txt_path = os.path.join(self.dataset_dir, 'ImageSets', '{}.txt'.format(mode))
        self.image_idx_list = [x.strip() for x in open(split_txt_path).readlines()]

        if self.is_test:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        else:
            self.sample_id_list = self.remove_invalid_idx(self.image_idx_list)

        if num_samples is not None:
            self.sample_id_list = self.sample_id_list[:num_samples]
        self.num_samples = len(self.sample_id_list)

    def __getitem__(self, index):
        if self.is_test:
            return self.load_img_only(index)
        else:
            if self.mosaic:
                img_files, rgb_map, targets = self.load_mosaic(index)

                return img_files[0], rgb_map, targets
            else:
                return self.load_img_with_targets(index)

    def load_img_only(self, index):
        """Load only image for the testing phase"""

        sample_id = int(self.sample_id_list[index])
        lidarData = self.get_lidar(sample_id)
        
        # Generate scan context for this sample
        self.generate_scan_context(sample_id, lidarData)
        
        b = kitti_bev_utils.removePoints(lidarData, cnf.boundary)
        rgb_map = kitti_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))

        return img_file, rgb_map

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""

        sample_id = int(self.sample_id_list[index])

        lidarData = self.get_lidar(sample_id)
        objects = self.get_label(sample_id)
        calib = self.get_calib(sample_id)

        labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects)

        if not noObjectLabels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                               calib.P)  # convert rect cam to velo cord

        # Generate scan context for this sample with pedestrian locations
        self.generate_scan_context(sample_id, lidarData, labels, objects, calib)

        if self.lidar_transforms is not None:
            lidarData, labels[:, 1:] = self.lidar_transforms(lidarData, labels[:, 1:])

        b = kitti_bev_utils.removePoints(lidarData, cnf.boundary)
        rgb_map = kitti_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
        target = kitti_bev_utils.build_yolo_target(labels)
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(sample_id))

        # on image space: targets are formatted as (box_idx, class, x, y, w, l, im, re)
        n_target = len(target)
        targets = torch.zeros((n_target, 8))
        if n_target > 0:
            targets[:, 1:] = torch.from_numpy(target)

        rgb_map = torch.from_numpy(rgb_map).float()

        if self.aug_transforms is not None:
            rgb_map, targets = self.aug_transforms(rgb_map, targets)

        return img_file, rgb_map, targets

    def load_mosaic(self, index):
        """loads images in a mosaic
        Refer: https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        """

        targets_s4 = []
        img_file_s4 = []
        if self.random_padding:
            yc, xc = [int(random.uniform(-x, 2 * self.img_size + x)) for x in self.mosaic_border]  # mosaic center
        else:
            yc, xc = [self.img_size, self.img_size]  # mosaic center

        indices = [index] + [random.randint(0, self.num_samples - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            img_file, img, targets = self.load_img_with_targets(index)
            img_file_s4.append(img_file)

            c, h, w = img.size()  # (3, 608, 608), torch tensor

            # place img in img4
            if i == 0:  # top left
                img_s4 = torch.full((c, self.img_size * 2, self.img_size * 2), 0.5, dtype=torch.float)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img_s4[:, y1a:y2a, x1a:x2a] = img[:, y1b:y2b, x1b:x2b]  # img_s4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # on image space: targets are formatted as (box_idx, class, x, y, w, l, sin(yaw), cos(yaw))
            if targets.size(0) > 0:
                targets[:, 2] = (targets[:, 2] * w + padw) / (2 * self.img_size)
                targets[:, 3] = (targets[:, 3] * h + padh) / (2 * self.img_size)
                targets[:, 4] = targets[:, 4] * w / (2 * self.img_size)
                targets[:, 5] = targets[:, 5] * h / (2 * self.img_size)

            targets_s4.append(targets)
        if len(targets_s4) > 0:
            targets_s4 = torch.cat(targets_s4, 0)
            torch.clamp(targets_s4[:, 2:4], min=0., max=(1. - 0.5 / self.img_size), out=targets_s4[:, 2:4])

        return img_file_s4, img_s4, targets_s4

    def __len__(self):
        return len(self.sample_id_list)

    def remove_invalid_idx(self, image_idx_list):
        """Discard samples which don't have current training class objects, which will not be used for training."""

        sample_id_list = []
        filtered_samples = []
        
        print(f"\nFiltering {len(image_idx_list)} samples for valid training objects...")
        
        for sample_id in image_idx_list:
            sample_id = int(sample_id)
            objects = self.get_label(sample_id)
            calib = self.get_calib(sample_id)
            labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects)
            if not noObjectLabels:
                labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                                   calib.P)  # convert rect cam to velo cord
            valid_list = []
            invalid_reasons = []
            
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in cnf.CLASS_NAME_TO_ID.values():
                    if self.check_point_cloud_range(labels[i, 1:4]):
                        valid_list.append(labels[i, 0])
                    else:
                        invalid_reasons.append(f"object_{i}_out_of_range")
                else:
                    invalid_reasons.append(f"object_{i}_invalid_class")

            if len(valid_list) > 0:
                sample_id_list.append(sample_id)
            else:
                filtered_samples.append({
                    'sample_id': sample_id,
                    'reasons': invalid_reasons if invalid_reasons else ['no_valid_objects']
                })

        # Print filtered samples
        if filtered_samples:
            print(f"\nFILTERED OUT {len(filtered_samples)} samples:")
            for item in filtered_samples[:10]:  # Show first 10 filtered samples
                print(f"  Sample {item['sample_id']:06d}: {', '.join(item['reasons'])}")
            if len(filtered_samples) > 10:
                print(f"  ... and {len(filtered_samples) - 10} more samples")
        else:
            print("No samples were filtered out.")
            
        print(f"KEPT {len(sample_id_list)} valid samples for training.")
        
        return sample_id_list

    def generate_scan_context(self, sample_id, lidarData, labels=None, objects=None, calib=None):
        """Generate scan context for a given sample and save as image and npy file"""
        try:
            ptcloud_xyz = lidarData[:, :3]
            sc = ScanContext(ptcloud_xyz)
            sc_data = sc.scancontexts[0]

            # Save .npy
            npy_file = os.path.join(self.scan_context_dir, f'sc_{sample_id:06d}.npy')
            np.save(npy_file, sc_data.astype(np.float32))

            # ---- collect pedestrians from both sources ----
            pedestrian_info = []

            if labels is not None:
                print(f"Sample {sample_id}: Processing {len(labels)} labels")
                if len(labels) > 0:
                    print(f"Sample {sample_id}: Label classes: {labels[:, 0]}")
                pedestrian_info.extend(ScanContext.find_pedestrians_in_scan_context(labels))

            if objects is not None and calib is not None:
                pedestrian_info.extend(self.objects_to_pedestrians_in_scan_context(objects, calib))

            print(f"Sample {sample_id}: Found {len(pedestrian_info)} pedestrians (merged)")

            # Always save JSON
            import json
            pedestrian_file = os.path.join(self.scan_context_dir, f'pedestrians_{sample_id:06d}.json')
            with open(pedestrian_file, 'w') as f:
                json.dump(pedestrian_info, f, indent=2)

            # ---- save image ----
            image_file = os.path.join(self.scan_context_dir, f'sc_{sample_id:06d}.png')

            fig = plt.figure(figsize=(7.2, 1.6), dpi=100)  # 720x160
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)

            # Normalize for display
            min_val, max_val = float(np.min(sc_data)), float(np.max(sc_data))
            sc_data_normalized = (sc_data - min_val) / (max_val - min_val) if max_val > min_val else sc_data

            ax.imshow(sc_data_normalized, cmap='viridis', aspect='auto', origin='lower')

            # Draw: all covered cells (red), plus a tiny white dot at center for reference
            # for ped in pedestrian_info:
            #     cells = np.array(ped.get('covered_cells', []), dtype=np.int32)
            #     if cells.size:
            #         # cells[:,0] = ring indices (rows); cells[:,1] = sector indices (cols)
            #         ax.scatter(cells[:, 1], cells[:, 0], s=3, c='r', alpha=0.85)
            #     # center
            #     if 'center_ring_idx' in ped and 'center_sector_idx' in ped:
            #         ax.plot(ped['center_sector_idx'], ped['center_ring_idx'], 'wo', markersize=2, alpha=0.9)

            plt.savefig(image_file, dpi=100, bbox_inches=None, pad_inches=0)
            plt.close(fig)
            return True

        except Exception as e:
            print(f"Error generating scan context for sample {sample_id}: {e}")
            return False


    def objects_to_pedestrians_in_scan_context(self, objects, calib):
        ped_list = []
        PEDESTRIAN_ID = cnf.CLASS_NAME_TO_ID.get('Pedestrian', 1)

        for obj in objects:
            if obj.type not in ("Pedestrian", "Person_sitting"):
                continue

            # Use mid-height center in rect coords (KITTI convention: box bottom at y, so subtract h/2)
            center_rect = np.array([obj.t[0], obj.t[1] - obj.h / 2.0, obj.t[2]], dtype=np.float32)

            # Convert center to LiDAR
            center_velo = calib.project_rect_to_velo(center_rect.reshape(1, 3)).reshape(-1)
            x, y, z = float(center_velo[0]), float(center_velo[1]), float(center_velo[2])

            # Convert yaw to LiDAR yaw
            yaw_lidar = self._rect_yaw_to_velo_yaw(float(obj.ry), center_rect, calib)

            # Compute indices for the center
            ring_idx, sector_idx = ScanContext.pedestrian_to_scan_context_coords(x, y)

            # Covered cells in SC using (l,w) from label, yaw in LiDAR
            covered = ScanContext.cells_covered_by_box(
                x=x, y=y, l=float(obj.l), w=float(obj.w), yaw_lidar=float(yaw_lidar)
            )

            ped_list.append({
                'class': PEDESTRIAN_ID,
                'x': x, 'y': y, 'z': z,
                'h': float(obj.h), 'w': float(obj.w), 'l': float(obj.l), 'yaw': float(yaw_lidar),
                'center_ring_idx': int(ring_idx),
                'center_sector_idx': int(sector_idx),
                'covered_cells': covered.tolist(),
                'distance': float(np.hypot(x, y)),
                'angle': float((np.degrees(np.arctan2(y, x)) % 360.0))
            })

        return ped_list

    def _rect_yaw_to_velo_yaw(self, ry_rect, center_rect, calib):
        """
        Convert KITTI rect-camera yaw (ry) at a given center point to LiDAR yaw (about +Z).
        We rotate a unit forward vector by ry in rect frame, transform both points to LiDAR,
        then compute atan2(dy, dx).
        """
        # center and a small step along the local +X in rect coords
        step = 1.0  # 1 meter forward
        # Rotation about camera Y axis
        c, s = np.cos(ry_rect), np.sin(ry_rect)
        R = np.array([[ c, 0, s],
                    [ 0, 1, 0],
                    [-s, 0, c]], dtype=np.float32)

        fwd_rect = (R @ np.array([step, 0.0, 0.0], dtype=np.float32)).reshape(1, 3)
        p0_rect = center_rect.reshape(1, 3).astype(np.float32)
        p1_rect = (center_rect + fwd_rect.reshape(3)).reshape(1, 3).astype(np.float32)

        p0_velo = calib.project_rect_to_velo(p0_rect).reshape(-1)
        p1_velo = calib.project_rect_to_velo(p1_rect).reshape(-1)
        dx, dy = (p1_velo[0] - p0_velo[0]), (p1_velo[1] - p0_velo[1])
        yaw_velo = float(np.arctan2(dy, dx))
        return yaw_velo



    def check_point_cloud_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range = [cnf.boundary["minX"], cnf.boundary["maxX"]]
        y_range = [cnf.boundary["minY"], cnf.boundary["maxY"]]
        z_range = [cnf.boundary["minZ"], cnf.boundary["maxZ"]]

        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if (self.batch_count % 10 == 0) and self.multiscale and (not self.mosaic):
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack(imgs)
        if self.img_size != cnf.BEV_WIDTH:
            imgs = F.interpolate(imgs, size=self.img_size, mode="bilinear", align_corners=True)
        self.batch_count += 1

        return paths, imgs, targets

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, '{:06d}.png'.format(idx))
        # assert os.path.isfile(img_file)
        return cv2.imread(img_file)  # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, '{:06d}.bin'.format(idx))
        # assert os.path.isfile(lidar_file)
        return np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(calib_file)
        return kitti_data_utils.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, '{:06d}.txt'.format(idx))
        # assert os.path.isfile(label_file)
        return kitti_data_utils.read_label(label_file)
