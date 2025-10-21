#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    QoSDurabilityPolicy,
)

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

import open3d as o3d
import matplotlib.pyplot as plt


class ScanContextComputer:
    """
    Builds Scan Context images from XYZ point clouds, matching your updated options:
      - use_full_height_range: z used as-is, cells init with -inf then replace with 0
      - custom_height_offset: z += offset, cells init with 0
    """

    def __init__(
        self,
        voxel_size: float = 0.20,
        max_length: float = 80.0,
        sector_res: int = 720,
        ring_res: int = 160,
        use_full_height_range: bool = False,
        custom_height_offset: float = 2.0,
        visualize: bool = False,
        store_pointcloud_png: bool = False,
        output_dir: str = "scan_results",
        dl_images_dir: str = "scan_dl_images_live",
        cmap: str = "viridis",
    ):
        self.voxel_size = voxel_size
        self.max_length = max_length
        self.sector_res = sector_res
        self.ring_res = ring_res
        self.use_full_height_range = bool(use_full_height_range)
        self.custom_height_offset = float(custom_height_offset)
        self.visualize = visualize
        self.store_pointcloud_png = store_pointcloud_png
        self.output_dir = output_dir
        self.dl_images_dir = dl_images_dir
        self.cmap = cmap

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.dl_images_dir, exist_ok=True)

        self._o3d_vis = None  # persistent window if visualize=True

    # ---------- Open3D helpers ----------
    def _o3d_visualize(self, pts_xyz: np.ndarray, title: str):
        if not self.visualize:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_xyz)
        if self._o3d_vis is None:
            self._o3d_vis = o3d.visualization.Visualizer()
            self._o3d_vis.create_window(window_name=title, width=960, height=720)
            self._o3d_vis.add_geometry(pcd)
        else:
            self._o3d_vis.clear_geometries()
            self._o3d_vis.add_geometry(pcd)
        self._o3d_vis.poll_events()
        self._o3d_vis.update_renderer()

    def _o3d_capture_png(self, pts_xyz: np.ndarray, file_stem: str):
        if not self.store_pointcloud_png:
            return
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_xyz)
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        vis.add_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        out = os.path.join(self.output_dir, f"pointcloud_{file_stem}.png")
        vis.capture_screen_image(out)
        vis.destroy_window()

    # ---------- core processing ----------
    @staticmethod
    def _downsample_voxel(pts_xyz: np.ndarray, voxel: float) -> np.ndarray:
        if pts_xyz.size == 0:
            return pts_xyz
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(pts_xyz)
        d = o3d.geometry.PointCloud.voxel_down_sample(p, voxel_size=voxel)
        return np.asarray(d.points)

    def _scan_context(self, pts_xyz: np.ndarray) -> np.ndarray:
        """
        Vectorized Scan Context via np.maximum.at for fast cell-wise maxima.
        Ring bins are radial distances (0..max_length), sector bins are 0..360°.
        """
        if pts_xyz.size == 0:
            return np.zeros((self.ring_res, self.sector_res), dtype=np.float32)

        x = pts_xyz[:, 0]
        y = pts_xyz[:, 1]
        z = pts_xyz[:, 2]

        # sanity: finite only
        m = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        if not np.all(m):
            x, y, z = x[m], y[m], z[m]

        # height mode
        if self.use_full_height_range:
            z_used = z  # keep negatives / positives
            init_val = -np.inf
            fill_empty_with = 0.0
        else:
            z_used = z + self.custom_height_offset
            init_val = 0.0    # in "offset" mode your original code started at 0
            fill_empty_with = 0.0

        # polar
        r = np.sqrt(x * x + y * y)
        theta = (np.rad2deg(np.arctan2(y, x)) + 360.0) % 360.0

        # only within max range
        m = r <= self.max_length
        if not np.any(m):
            return np.zeros((self.ring_res, self.sector_res), dtype=np.float32)
        r, theta, z_used = r[m], theta[m], z_used[m]

        gap_ring = self.max_length / self.ring_res
        gap_sector = 360.0 / self.sector_res

        ring_idx = np.floor(r / gap_ring).astype(np.int32)
        ring_idx = np.clip(ring_idx, 0, self.ring_res - 1)
        sector_idx = np.floor(theta / gap_sector).astype(np.int32)
        sector_idx = np.clip(sector_idx, 0, self.sector_res - 1)

        # flattened index
        flat_idx = ring_idx * self.sector_res + sector_idx

        sc_flat = np.full(self.ring_res * self.sector_res, init_val, dtype=np.float32)
        np.maximum.at(sc_flat, flat_idx, z_used.astype(np.float32))

        # empty cells → 0 if we were using -inf init
        if self.use_full_height_range:
            sc_flat[~np.isfinite(sc_flat)] = fill_empty_with

        return sc_flat.reshape((self.ring_res, self.sector_res))

    def compute_and_save(self, pts_xyz: np.ndarray, stamp_stem: str):
        # downsample
        pts_ds = self._downsample_voxel(pts_xyz, self.voxel_size)

        # optional viz
        self._o3d_visualize(pts_ds, "Live Velodyne / ScanContext")
        self._o3d_capture_png(pts_ds, stamp_stem)

        # SC
        sc = self._scan_context(pts_ds)

        # normalized PNG (0..1)
        mn = float(np.min(sc))
        mx = float(np.max(sc))
        sc_norm = (sc - mn) / (mx - mn) if mx > mn else sc

        fig = plt.figure(figsize=(self.sector_res / 100.0, self.ring_res / 100.0), dpi=100)
        ax = plt.Axes(fig, [0, 0, 1, 1])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(sc_norm, cmap=self.cmap, aspect="auto")

        out_png = os.path.join(self.dl_images_dir, f"sc_{stamp_stem}.png")
        plt.savefig(out_png, dpi=100, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        out_npy = os.path.join(self.output_dir, f"sc_{stamp_stem}.npy")
        np.save(out_npy, sc.astype(np.float32))

        return sc, out_png, out_npy


class ScanContextNode(Node):
    def __init__(self):
        super().__init__("scan_context_node")

        # -------- Parameters (match your class flags) --------
        self.declare_parameter("topic", "/velodyne_points")
        self.declare_parameter("voxel_size", 0.20)
        self.declare_parameter("max_length", 80.0)
        self.declare_parameter("sector_res", 720)
        self.declare_parameter("ring_res", 160)
        self.declare_parameter("use_full_height_range", 1)
        self.declare_parameter("custom_height_offset", 2.0)
        self.declare_parameter("visualize", 1)
        self.declare_parameter("store_pointcloud_png", 0)
        self.declare_parameter("output_dir", "scan_results")
        self.declare_parameter("dl_images_dir", "scan_dl_images_live")
        self.declare_parameter("cmap", "viridis")

        topic = self.get_parameter("topic").get_parameter_value().string_value
        voxel_size = float(self.get_parameter("voxel_size").value)
        max_length = float(self.get_parameter("max_length").value)
        sector_res = int(self.get_parameter("sector_res").value)
        ring_res = int(self.get_parameter("ring_res").value)
        use_full_height_range = bool(self.get_parameter("use_full_height_range").value)
        custom_height_offset = float(self.get_parameter("custom_height_offset").value)
        visualize = bool(self.get_parameter("visualize").value)
        store_pointcloud_png = bool(self.get_parameter("store_pointcloud_png").value)
        output_dir = self.get_parameter("output_dir").get_parameter_value().string_value
        dl_images_dir = self.get_parameter("dl_images_dir").get_parameter_value().string_value
        cmap = self.get_parameter("cmap").get_parameter_value().string_value

        self.sc = ScanContextComputer(
            voxel_size=voxel_size,
            max_length=max_length,
            sector_res=sector_res,
            ring_res=ring_res,
            use_full_height_range=use_full_height_range,
            custom_height_offset=custom_height_offset,
            visualize=visualize,
            store_pointcloud_png=store_pointcloud_png,
            output_dir=output_dir,
            dl_images_dir=dl_images_dir,
            cmap=cmap,
        )

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.sub = self.create_subscription(PointCloud2, topic, self._cb, qos)
        mode = "Full range" if use_full_height_range else f"Custom offset (+{custom_height_offset} m)"
        self.get_logger().info(
            f"Subscribed to {topic} | rings={ring_res}, sectors={sector_res}, Rmax={max_length}, voxel={voxel_size}, height mode: {mode}"
        )

    @staticmethod
    def _extract_xyz(msg: PointCloud2) -> np.ndarray:
        """
        Robust XYZ extraction that works whether sensor_msgs_py returns a structured
        array or tuples. Prefers read_points_numpy (fast), falls back to read_points.
        Returns an (N,3) float32 array.
        """
        # Fast path: use read_points_numpy if available (ROS 2 Humble+)
        try:
            # Newer API returns a structured ndarray with named fields
            arr = pc2.read_points_numpy(msg, field_names=("x", "y", "z"))
            # Some implementations may give a plain ndarray already
            if isinstance(arr, np.ndarray) and arr.dtype.fields:
                # Structured -> stack named fields into plain Nx3
                xyz = np.stack([arr["x"], arr["y"], arr["z"]], axis=-1).astype(np.float32, copy=False)
            else:
                # Already plain
                xyz = arr.astype(np.float32, copy=False)
            return xyz
        except Exception:
            # Fallback: read_points returns a generator of tuples
            pts_iter = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            # Build a structured array then view as plain float32
            rec = np.fromiter(pts_iter, dtype=[("x", np.float32), ("y", np.float32), ("z", np.float32)])
            if rec.size == 0:
                return np.empty((0, 3), dtype=np.float32)
            xyz = np.vstack((rec["x"], rec["y"], rec["z"])).T
            return xyz.astype(np.float32, copy=False)

    def _cb(self, msg: PointCloud2):
        stamp = msg.header.stamp
        stem = f"{msg.header.frame_id}_{stamp.sec}.{str(stamp.nanosec).zfill(9)}"
        try:
            xyz = self._extract_xyz(msg)
            if xyz.size == 0:
                self.get_logger().warn("Empty point cloud; skipping.")
                return
            sc, png, npy = self.sc.compute_and_save(xyz, stem)
            self.get_logger().info(
                f"SC {sc.shape} saved → {os.path.basename(png)}, {os.path.basename(npy)}"
            )
        except Exception as e:
            self.get_logger().error(f"Processing error: {e}")

    def destroy_node(self):
        if self.sc._o3d_vis is not None:
            self.sc._o3d_vis.destroy_window()
        super().destroy_node()


def main():
    rclpy.init()
    node = ScanContextNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
