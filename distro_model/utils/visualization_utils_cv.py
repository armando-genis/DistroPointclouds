import sys
import math
import numpy as np
import cv2

sys.path.append('../')

from data_process import kitti_data_utils, kitti_bev_utils, transformation
import config.kitti_config as cnf


def create_bev_image(pc_velo, boundary=None, resolution=0.1):
    """
    Create bird's eye view (BEV) image from lidar point cloud
    """
    if boundary is None:
        boundary = {
            'minX': 0, 'maxX': 40,
            'minY': -20, 'maxY': 20,
            'minZ': -2.0, 'maxZ': 0.4
        }
    
    # Define boundary and resolution
    x_min, x_max = boundary['minX'], boundary['maxX']
    y_min, y_max = boundary['minY'], boundary['maxY']
    z_min, z_max = boundary['minZ'], boundary['maxZ']
    
    # Calculate dimensions
    width = int((x_max - x_min) / resolution)
    height = int((y_max - y_min) / resolution)
    
    # Create empty BEV image
    bev_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Filter points within boundary
    mask = (pc_velo[:, 0] >= x_min) & (pc_velo[:, 0] < x_max) & \
           (pc_velo[:, 1] >= y_min) & (pc_velo[:, 1] < y_max) & \
           (pc_velo[:, 2] >= z_min) & (pc_velo[:, 2] < z_max)
    
    pc_filtered = pc_velo[mask]
    
    if len(pc_filtered) == 0:
        return bev_img
    
    # Convert to pixel coordinates
    x_img = ((pc_filtered[:, 0] - x_min) / resolution).astype(np.int32)
    y_img = ((pc_filtered[:, 1] - y_min) / resolution).astype(np.int32)
    
    # Normalize height (z) for coloring
    z_normalized = np.clip((pc_filtered[:, 2] - z_min) / (z_max - z_min), 0, 1)
    
    # Draw points on BEV image
    for i in range(len(x_img)):
        x, y, c = x_img[i], y_img[i], z_normalized[i]
        # Ensure coordinates are within image bounds
        if 0 <= x < width and 0 <= y < height:
            # Use height for coloring (blue-green-red gradient)
            color = (int((1-c)*255), int(c*255), 0)  # (B, G, R)
            cv2.circle(bev_img, (x, height - y - 1), 1, color, -1)
    
    # Draw reference grid
    grid_size = 10  # meters
    grid_steps = int(grid_size / resolution)
    
    for i in range(0, width, grid_steps):
        cv2.line(bev_img, (i, 0), (i, height), (50, 50, 50), 1)
    
    for i in range(0, height, grid_steps):
        cv2.line(bev_img, (0, i), (width, i), (50, 50, 50), 1)
    
    # Draw axes and origin
    origin_x = int((0 - x_min) / resolution)
    origin_y = height - int((0 - y_min) / resolution) - 1
    
    # X-axis (forward) - Red
    cv2.line(bev_img, (origin_x, origin_y), (origin_x + 50, origin_y), (0, 0, 255), 2)
    # Y-axis (left) - Green
    cv2.line(bev_img, (origin_x, origin_y), (origin_x, origin_y - 50), (0, 255, 0), 2)
    
    # Draw origin
    cv2.circle(bev_img, (origin_x, origin_y), 5, (255, 255, 255), -1)
    
    # Draw FOV lines
    fov_angle = 45  # degrees
    fov_length = 20  # meters
    angle_rad = np.radians(fov_angle)
    
    p1_x = origin_x + int((fov_length * np.cos(angle_rad)) / resolution)
    p1_y = origin_y - int((fov_length * np.sin(angle_rad)) / resolution)
    p2_x = origin_x + int((fov_length * np.cos(-angle_rad)) / resolution)
    p2_y = origin_y - int((fov_length * np.sin(-angle_rad)) / resolution)
    
    cv2.line(bev_img, (origin_x, origin_y), (p1_x, p1_y), (255, 255, 255), 1)
    cv2.line(bev_img, (origin_x, origin_y), (p2_x, p2_y), (255, 255, 255), 1)
    
    # Draw square region (same as in the original code)
    x1 = int((boundary['minX'] - x_min) / resolution)
    x2 = int((boundary['maxX'] - x_min) / resolution)
    y1 = height - int((boundary['maxY'] - y_min) / resolution) - 1
    y2 = height - int((boundary['minY'] - y_min) / resolution) - 1
    
    cv2.line(bev_img, (x1, y1), (x1, y2), (128, 128, 128), 1)
    cv2.line(bev_img, (x2, y1), (x2, y2), (128, 128, 128), 1)
    cv2.line(bev_img, (x1, y1), (x2, y1), (128, 128, 128), 1)
    cv2.line(bev_img, (x1, y2), (x2, y2), (128, 128, 128), 1)
    
    return bev_img


def draw_3d_boxes_on_bev(bev_img, objects, calib, boundary=None, resolution=0.1):
    """
    Draw 3D bounding boxes on BEV image
    """
    if boundary is None:
        boundary = {
            'minX': 0, 'maxX': 40,
            'minY': -20, 'maxY': 20,
            'minZ': -2.0, 'maxZ': 0.4
        }
    
    x_min, y_min = boundary['minX'], boundary['minY']
    height, width = bev_img.shape[:2]
    
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        
        # Get box corners in camera coordinate
        _, box3d_pts_3d = kitti_data_utils.compute_box_3d(obj, calib.P)
        
        # Convert to lidar coordinate
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        
        # Convert to BEV image coordinate
        x_img = ((box3d_pts_3d_velo[:, 0] - x_min) / resolution).astype(np.int32)
        y_img = height - ((box3d_pts_3d_velo[:, 1] - y_min) / resolution).astype(np.int32) - 1
        
        # Get color based on object class (similar to original code)
        if hasattr(obj, 'cls_id') and obj.cls_id in cnf.colors:
            color = cnf.colors[obj.cls_id]
        else:
            color = (0, 255, 255)  # Default: Yellow
        
        # Draw box in BEV
        # Bottom face (first 4 corners)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            if 0 <= x_img[i] < width and 0 <= y_img[i] < height and 0 <= x_img[j] < width and 0 <= y_img[j] < height:
                cv2.line(bev_img, (x_img[i], y_img[i]), (x_img[j], y_img[j]), color, 2)
        
        # Calculate heading (front of the box)
        front_mid_x = int((x_img[0] + x_img[1]) / 2)
        front_mid_y = int((y_img[0] + y_img[1]) / 2)
        back_mid_x = int((x_img[2] + x_img[3]) / 2)
        back_mid_y = int((y_img[2] + y_img[3]) / 2)
        
        # Draw heading arrow if within image bounds
        if (0 <= front_mid_x < width and 0 <= front_mid_y < height and 
            0 <= back_mid_x < width and 0 <= back_mid_y < height):
            cv2.arrowedLine(bev_img, (back_mid_x, back_mid_y), (front_mid_x, front_mid_y), 
                          (0, 0, 255), 2, tipLength=0.3)
    
    return bev_img


def draw_lidar_points_on_image(pc_velo, img, calib, point_size=2, color_by_distance=True):
    """
    Project lidar points to image plane
    """
    img_copy = img.copy()
    pts_2d = calib.project_velo_to_image(pc_velo)
    
    # Filter points that are in the image
    img_h, img_w = img.shape[:2]
    valid_pts = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_w) & \
                (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_h)
    
    # Get valid points
    pts_valid = pts_2d[valid_pts]
    pc_valid = pc_velo[valid_pts]
    
    # Color by distance
    if color_by_distance and len(pc_valid) > 0:
        depths = pc_valid[:, 0]  # Distance in x direction (forward)
        depth_min, depth_max = np.min(depths), np.max(depths)
        depth_range = max(1.0, depth_max - depth_min)
        
        for i in range(len(pts_valid)):
            pt = pts_valid[i]
            depth = depths[i]
            
            # Normalize depth for coloring (blue to red gradient)
            norm_depth = (depth - depth_min) / depth_range
            color = (
                int(255 * (1 - norm_depth)),  # Blue
                0,                           # Green
                int(255 * norm_depth)         # Red
            )
            
            cv2.circle(img_copy, (int(pt[0]), int(pt[1])), point_size, color, -1)
    else:
        # Simple uniform color
        for pt in pts_valid:
            cv2.circle(img_copy, (int(pt[0]), int(pt[1])), point_size, (0, 255, 0), -1)
    
    return img_copy


def show_lidar_with_boxes_cv(pc_velo, objects, calib, boundary=None, img_size=(800, 800)):
    """
    Visualize lidar point cloud with 3D boxes using OpenCV
    """
    # Create BEV image
    bev_img = create_bev_image(pc_velo, boundary)
    
    # Draw 3D boxes on BEV
    bev_img = draw_3d_boxes_on_bev(bev_img, objects, calib, boundary)
    
    # Resize to desired shape
    bev_img = cv2.resize(bev_img, img_size)
    
    return bev_img


def show_image_with_boxes(img, objects, calib, show3d=False):
    """
    Show image with 2D bounding boxes
    """
    img2 = np.copy(img)  # for 3d bbox
    for obj in objects:
        if obj.type == 'DontCare':
            continue
        
        box3d_pts_2d, box3d_pts_3d = kitti_data_utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is not None:
            img2 = kitti_data_utils.draw_projected_box3d(img2, box3d_pts_2d, cnf.colors[obj.cls_id])
    
    if show3d:
        cv2.imshow("img", img2)
        
    return img2


def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=0.0):
    """
    Filter lidar points, keep those in image FOV
    """
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:, 0] < xmax) & (pts_2d[:, 0] >= xmin) & \
               (pts_2d[:, 1] < ymax) & (pts_2d[:, 1] >= ymin)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds, :]
    
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo


def merge_rgb_to_bev(img_rgb, img_bev, output_width):
    """
    Merge RGB image and BEV image vertically
    """
    img_rgb_h, img_rgb_w = img_rgb.shape[:2]
    ratio_rgb = output_width / img_rgb_w
    output_rgb_h = int(ratio_rgb * img_rgb_h)
    ret_img_rgb = cv2.resize(img_rgb, (output_width, output_rgb_h))

    img_bev_h, img_bev_w = img_bev.shape[:2]
    ratio_bev = output_width / img_bev_w
    output_bev_h = int(ratio_bev * img_bev_h)
    ret_img_bev = cv2.resize(img_bev, (output_width, output_bev_h))

    out_img = np.zeros((output_rgb_h + output_bev_h, output_width, 3), dtype=np.uint8)
    # Upper: RGB --> BEV
    out_img[:output_rgb_h, ...] = ret_img_rgb
    out_img[output_rgb_h:, ...] = ret_img_bev

    return out_img


def visualize_lidar_with_depth(pc_velo, img, calib, img_width, img_height):
    """
    Visualize lidar points with depth color on image
    """
    # Project lidar points to image plane
    img_with_points = draw_lidar_points_on_image(pc_velo, img, calib)
    
    # Create BEV representation
    bev_img = create_bev_image(pc_velo)
    
    # Merge images
    combined_img = merge_rgb_to_bev(img_with_points, bev_img, img_width)
    
    return combined_img


def invert_target(targets, calib, img_shape_2d, RGB_Map=None):
    """
    Convert targets from network output format to KITTI format
    """
    predictions = targets
    predictions = kitti_bev_utils.inverse_yolo_target(predictions, cnf.boundary)
    if predictions.shape[0]:
        predictions[:, 1:] = transformation.lidar_to_camera_box(predictions[:, 1:], calib.V2C, calib.R0, calib.P)

    objects_new = []
    corners3d = []
    for index, l in enumerate(predictions):
        if l[0] == 0:
            str = "Car"
        elif l[0] == 1:
            str = "Pedestrian"
        elif l[0] == 2:
            str = "Cyclist"
        else:
            str = "Ignore"
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        obj = kitti_data_utils.Object3d(line)
        obj.t = l[1:4]
        obj.h, obj.w, obj.l = l[4:7]
        obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))

        _, corners_3d = kitti_data_utils.compute_box_3d(obj, calib.P)
        corners3d.append(corners_3d)
        objects_new.append(obj)

    if len(corners3d) > 0:
        corners3d = np.array(corners3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape_2d[1] * 0.8, img_boxes_h < img_shape_2d[0] * 0.8)

    for i, obj in enumerate(objects_new):
        x, z, ry = obj.t[0], obj.t[2], obj.ry
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        obj.alpha = alpha
        obj.box2d = img_boxes[i, :]

    if RGB_Map is not None:
        labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects_new)
        if not noObjectLabels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                               calib.P)  # convert rect cam to velo cord

        target = kitti_bev_utils.build_yolo_target(labels)
        kitti_bev_utils.draw_box_in_bev(RGB_Map, target)

    return objects_new


def predictions_to_kitti_format(img_detections, calib, img_shape_2d, img_size, RGB_Map=None):
    """
    Convert network predictions to KITTI format
    """
    predictions = []
    for detections in img_detections:
        if detections is None:
            continue
        # Rescale boxes to original image
        for x, y, w, l, im, re, *_, cls_pred in detections:
            predictions.append([cls_pred, x / img_size, y / img_size, w / img_size, l / img_size, im, re])

    predictions = kitti_bev_utils.inverse_yolo_target(np.array(predictions), cnf.boundary)
    if predictions.shape[0]:
        predictions[:, 1:] = transformation.lidar_to_camera_box(predictions[:, 1:], calib.V2C, calib.R0, calib.P)

    objects_new = []
    corners3d = []
    for index, l in enumerate(predictions):
        if l[0] == 0:
            str = "Car"
        elif l[0] == 1:
            str = "Pedestrian"
        elif l[0] == 2:
            str = "Cyclist"
        else:
            str = "Ignore"
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        obj = kitti_data_utils.Object3d(line)
        obj.t = l[1:4]
        obj.h, obj.w, obj.l = l[4:7]
        obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))

        _, corners_3d = kitti_data_utils.compute_box_3d(obj, calib.P)
        corners3d.append(corners_3d)
        objects_new.append(obj)

    if len(corners3d) > 0:
        corners3d = np.array(corners3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape_2d[1] * 0.8, img_boxes_h < img_shape_2d[0] * 0.8)

    for i, obj in enumerate(objects_new):
        x, z, ry = obj.t[0], obj.t[2], obj.ry
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        obj.alpha = alpha
        obj.box2d = img_boxes[i, :]

    if RGB_Map is not None:
        labels, noObjectLabels = kitti_bev_utils.read_labels_for_bevbox(objects_new)
        if not noObjectLabels:
            labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
                                                               calib.P)  # convert rect cam to velo cord

        target = kitti_bev_utils.build_yolo_target(labels)
        kitti_bev_utils.draw_box_in_bev(RGB_Map, target)

    return objects_new


def display_lidar_and_camera_visualization(pc_velo, img, objects, calib, boundary=None):
    """
    Create a comprehensive visualization with camera image, lidar points, and 3D boxes
    """
    # Project lidar to the image
    img_with_lidar = draw_lidar_points_on_image(pc_velo, img, calib)
    
    # Draw 3D boxes on the image
    img_with_boxes = show_image_with_boxes(img_with_lidar, objects, calib)
    
    # Create BEV visualization
    bev_img = show_lidar_with_boxes_cv(pc_velo, objects, calib, boundary)
    
    # Merge both visualizations
    combined_img = merge_rgb_to_bev(img_with_boxes, bev_img, img_with_boxes.shape[1])
    
    return combined_img