"""
Face Animation Module for animating Mona Lisa's mouth movements
"""
import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import threading
import time

try:
    from scipy.spatial import Delaunay
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

class FaceAnimator:
    def __init__(self, image_path, max_dimension=1500):
        """
        Initialize face animator
        max_dimension: Maximum dimension for processing (MediaPipe works better with smaller images)
        """
        # Load and resize image for processing
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Resize if too large (MediaPipe works better with reasonable sizes)
        h, w = original.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            self.original_image = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self.scale_factor = scale
            print(f"Resized image from {w}x{h} to {new_w}x{new_h} for processing")
        else:
            self.original_image = original
            self.scale_factor = 1.0
        
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # MediaPipe face mesh has 468 landmarks
        # Mouth landmarks: outer lips (61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321)
        # Inner mouth (78, 95, 88, 178, 87, 14, 317, 402, 318, 324)
        self.mouth_outer = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321]
        self.mouth_inner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        
        # Detect face landmarks in original image
        self.face_landmarks = self._detect_landmarks()
        if not self.face_landmarks:
            print("Warning: Could not detect face landmarks. Animation may not work correctly.")
            print("Try using a clearer image or ensure the face is clearly visible.")
        else:
            print(f"Successfully detected {len(self.face_landmarks)} facial landmarks")
        
        self.current_image = self.original_image.copy()
        self.is_animating = False
        self.mouth_center = None
        self.mouth_height = None
        
        # Calculate mouth region if landmarks detected
        if self.face_landmarks:
            self._calculate_mouth_region()
    
    def _detect_landmarks(self):
        """Detect facial landmarks in the image"""
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = self.original_image.shape[:2]
            
            # Convert normalized landmarks to pixel coordinates
            landmark_points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append((x, y))
            
            return landmark_points
        else:
            print("MediaPipe could not detect a face in the image")
        return None
    
    def _calculate_mouth_region(self):
        """Calculate mouth center and dimensions from landmarks"""
        if not self.face_landmarks or len(self.face_landmarks) < 468:
            return
        
        # Get mouth landmarks
        mouth_points = []
        for idx in self.mouth_outer + self.mouth_inner:
            if idx < len(self.face_landmarks):
                mouth_points.append(self.face_landmarks[idx])
        
        if mouth_points:
            # Calculate mouth bounds (lips-only, from landmarks)
            x_coords = [p[0] for p in mouth_points]
            y_coords = [p[1] for p in mouth_points]
            
            self.mouth_center = (int(np.mean(x_coords)), int(np.mean(y_coords)))
            mouth_width = max(x_coords) - min(x_coords)
            mouth_height = max(y_coords) - min(y_coords)
            self.mouth_height = mouth_height
            
            # Lips-only bounding box with small padding (so we don't move whole face)
            pad_x = max(4, int(mouth_width * 0.25))
            pad_y = max(3, int(mouth_height * 0.4))
            h_img, w_img = self.original_image.shape[:2]
            self.mouth_x1 = max(0, min(x_coords) - pad_x)
            self.mouth_x2 = min(w_img, max(x_coords) + pad_x)
            self.mouth_y1 = max(0, min(y_coords) - pad_y)   # top of lips
            self.mouth_y2 = min(h_img, max(y_coords) + pad_y)  # bottom of lips
            
            # Build mesh for CPU mesh warp (lip-contour deformation)
            self._build_mouth_mesh()
            
            print(f"Mouth region: center={self.mouth_center}, lips box ({self.mouth_x1},{self.mouth_y1})-({self.mouth_x2},{self.mouth_y2})")
    
    def _build_mouth_mesh(self):
        """Build Delaunay triangulation over mouth landmarks + ROI corners for mesh warp."""
        if not _HAS_SCIPY or not getattr(self, 'mouth_x1', None):
            self.mouth_tri_simplices = None
            return
        mouth_indices = self.mouth_outer + self.mouth_inner
        points = []
        for idx in mouth_indices:
            if idx < len(self.face_landmarks):
                points.append(self.face_landmarks[idx])
        if len(points) < 6:
            self.mouth_tri_simplices = None
            return
        # Add ROI corners so mesh covers full mouth rectangle
        x1, x2, y1, y2 = self.mouth_x1, self.mouth_x2, self.mouth_y1, self.mouth_y2
        points.extend([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
        pts = np.array(points, dtype=np.float32)
        try:
            tri = Delaunay(pts)
            self.mouth_tri_points = pts
            self.mouth_tri_simplices = tri.simplices
            cy = self.mouth_center[1]
            self.mouth_lower_point_mask = np.array(
                [self.face_landmarks[idx][1] >= cy for idx in mouth_indices if idx < len(self.face_landmarks)],
                dtype=bool
            )
        except Exception:
            self.mouth_tri_simplices = None
    
    def _apply_mouth_movement_mesh(self, mouth_open_factor):
        """Warp mouth region by deforming landmark-based mesh (lip contour follows landmarks)."""
        if not _HAS_SCIPY or getattr(self, 'mouth_tri_simplices', None) is None:
            return False
        h, w = self.current_image.shape[:2]
        x1, x2, y1, y2 = self.mouth_x1, self.mouth_x2, self.mouth_y1, self.mouth_y2
        roi_w, roi_h = x2 - x1, y2 - y1
        if roi_w <= 0 or roi_h <= 0:
            return False
        
        open_amount = (mouth_open_factor * self.mouth_height * 1.5) if self.mouth_height else (mouth_open_factor * 18)
        if open_amount <= 0:
            return True
        
        src_pts = self.mouth_tri_points.copy()
        n_mouth = min(len(self.mouth_outer) + len(self.mouth_inner), len(src_pts) - 4)
        tgt_pts = src_pts.copy()
        mask = getattr(self, 'mouth_lower_point_mask', None)
        if mask is not None and len(mask) >= n_mouth:
            for i in range(n_mouth):
                if mask[i]:
                    tgt_pts[i, 1] = min(h - 1, src_pts[i, 1] + open_amount)
        
        # ROI-local coordinates; start with identity remap
        map_x = np.zeros((roi_h, roi_w), dtype=np.float32)
        map_y = np.zeros((roi_h, roi_w), dtype=np.float32)
        for rj in range(roi_w):
            map_x[:, rj] = rj
        for ri in range(roi_h):
            map_y[ri, :] = ri
        gx = np.arange(roi_w, dtype=np.float32) + x1 + 0.5
        gy = np.arange(roi_h, dtype=np.float32) + y1 + 0.5
        gx = np.broadcast_to(gx, (roi_h, roi_w))
        gy = np.broadcast_to(gy[:, np.newaxis], (roi_h, roi_w))
        
        for tri in self.mouth_tri_simplices:
            t_tri = tgt_pts[tri]
            s_tri = src_pts[tri]
            # Affine: target pixel -> source pixel (in global coords)
            M = cv2.getAffineTransform(t_tri.astype(np.float32), s_tri.astype(np.float32))
            # Barycentric in target triangle (vectorized)
            v0 = t_tri[1] - t_tri[0]
            v1 = t_tri[2] - t_tri[0]
            d00 = np.dot(v0, v0)
            d01 = np.dot(v0, v1)
            d11 = np.dot(v1, v1)
            denom = d00 * d11 - d01 * d01 + 1e-10
            v2_x = gx - t_tri[0, 0]
            v2_y = gy - t_tri[0, 1]
            d20 = v2_x * v0[0] + v2_y * v0[1]
            d21 = v2_x * v1[0] + v2_y * v1[1]
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1 - v - w
            inside = (u >= -1e-5) & (v >= -1e-5) & (w >= -1e-5)
            if not np.any(inside):
                continue
            # Source coords for pixels inside this triangle
            src_xy = M @ np.stack([gx[inside], gy[inside], np.ones(np.sum(inside))], axis=0)
            sx = (src_xy[0] - x1).astype(np.float32)
            sy = (src_xy[1] - y1).astype(np.float32)
            map_x[inside] = sx
            map_y[inside] = sy
        
        roi_src = self.original_image[y1:y2, x1:x2]
        warped_roi = cv2.remap(roi_src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        blend_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        cx_roi, cy_roi = roi_w // 2, roi_h // 2
        cv2.ellipse(blend_mask, (cx_roi, cy_roi), (roi_w // 2, roi_h // 2), 0, 0, 360, 255, -1)
        k = min(15, roi_h | 1, roi_w | 1)
        if k % 2 == 0:
            k -= 1
        if k >= 3:
            blend_mask = cv2.GaussianBlur(blend_mask, (k, k), 0)
        blend_mask = blend_mask.astype(np.float32) / 255.0
        roi_dst = self.current_image[y1:y2, x1:x2]
        for c in range(3):
            roi_dst[:, :, c] = (roi_dst[:, :, c] * (1 - blend_mask) + warped_roi[:, :, c] * blend_mask).astype(np.uint8)
        self.current_image[y1:y2, x1:x2] = roi_dst
        return True
    
    def animate_mouth(self, duration=3.0, intensity=0.7, update_callback=None):
        """
        Animate mouth opening/closing
        duration: how long to animate in seconds
        intensity: how much the mouth opens (0.0 to 1.0)
        update_callback: function to call after each frame update
        """
        if not self.face_landmarks:
            print("Cannot animate: No face landmarks detected")
            return
        
        if not self.mouth_center:
            print("Cannot animate: Mouth region not calculated")
            return
        
        print(f"Starting mouth animation for {duration} seconds with intensity {intensity}")
        self.is_animating = True
        start_time = time.time()
        fps = 20  # 20 FPS for smoother GUI updates
        frame_time = 1.0 / fps
        frame_count = 0
        
        while time.time() - start_time < duration and self.is_animating:
            elapsed = time.time() - start_time
            # Create sine wave for smooth mouth movement (faster cycle for visible animation)
            progress = (elapsed % 0.15) / 0.15  # Cycle every 0.15 seconds for more visible movement
            mouth_open = abs(np.sin(progress * 2 * np.pi)) * intensity  # Use abs for always positive
            
            self._apply_mouth_movement(mouth_open, frame_count)
            
            # Call update callback every frame to ensure smooth GUI updates
            if update_callback:
                update_callback()
            
            frame_count += 1
            time.sleep(frame_time)
        
        # Reset to original
        self.current_image = self.original_image.copy()
        self.is_animating = False
        if update_callback:
            update_callback()  # Final update
        print("Animation finished")
    
    def _apply_mouth_movement(self, mouth_open_factor, frame_count=0):
        """Apply mouth movement: try mesh warp (lip contour) first, else rectangle stretch."""
        if not self.face_landmarks or not self.mouth_center:
            return
        if not getattr(self, 'mouth_y1', None):
            return
        
        self.current_image = self.original_image.copy()
        h, w = self.current_image.shape[:2]
        
        if mouth_open_factor <= 0:
            return
        
        # Prefer mesh warp so lip contour deforms with landmarks
        if self._apply_mouth_movement_mesh(mouth_open_factor):
            return
        
        # Fallback: lips-only rectangle stretch (upper lip fixed)
        if self.mouth_height:
            open_amount = int(mouth_open_factor * self.mouth_height * 1.8)
        else:
            open_amount = int(mouth_open_factor * 20)
        if open_amount <= 0:
            return
        
        x1, x2 = self.mouth_x1, self.mouth_x2
        y1, y2 = self.mouth_y1, self.mouth_y2
        mouth_roi = self.original_image[y1:y2, x1:x2].copy()
        if mouth_roi.size == 0:
            return
        
        roi_h, roi_w = mouth_roi.shape[:2]
        new_height = roi_h + open_amount
        stretched = cv2.resize(mouth_roi, (roi_w, new_height), interpolation=cv2.INTER_LINEAR)
        stretch_y1, stretch_y2 = y1, min(h, y1 + new_height)
        stretch_x1, stretch_x2 = x1, x2
        actual_h = stretch_y2 - stretch_y1
        actual_w = stretch_x2 - stretch_x1
        if actual_h <= 0 or actual_w <= 0:
            return
        if stretched.shape[0] > actual_h or stretched.shape[1] != actual_w:
            stretched = cv2.resize(stretched, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)
        if stretched.shape[0] != actual_h or stretched.shape[1] != actual_w:
            return
        mask = np.zeros((actual_h, actual_w), dtype=np.uint8)
        cv2.ellipse(mask, (actual_w // 2, actual_h // 2),
                    (actual_w // 2, actual_h // 2), 0, 0, 360, 255, -1)
        k = min(15, actual_h | 1, actual_w | 1)
        if k % 2 == 0:
            k -= 1
        if k >= 3:
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        mask = mask.astype(np.float32) / 255.0
        roi = self.current_image[stretch_y1:stretch_y2, stretch_x1:stretch_x2]
        if roi.shape == stretched.shape:
            for c in range(3):
                roi[:, :, c] = (roi[:, :, c] * (1 - mask) + stretched[:, :, c] * mask).astype(np.uint8)
            self.current_image[stretch_y1:stretch_y2, stretch_x1:stretch_x2] = roi
    
    def get_current_frame(self):
        """Get current animated frame"""
        return self.current_image.copy()
    
    def stop_animation(self):
        """Stop animation and reset to original"""
        self.is_animating = False
        self.current_image = self.original_image.copy()
