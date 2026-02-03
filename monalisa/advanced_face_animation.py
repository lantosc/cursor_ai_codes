"""
Advanced Face Animation with audio-driven visemes and sophisticated morphing
"""
import cv2
import numpy as np
import mediapipe as mp
import time
import threading
import os

try:
    from audio_analyzer import AudioAnalyzer
    AUDIO_ANALYZER_AVAILABLE = True
except ImportError as e:
    print(f"Audio analyzer not available: {e}")
    print("Falling back to basic animation")
    AUDIO_ANALYZER_AVAILABLE = False
    AudioAnalyzer = None

class AdvancedFaceAnimator:
    """Advanced face animator with viseme-based lip sync and smooth morphing"""
    
    def __init__(self, image_path, max_dimension=1500):
        """Initialize advanced face animator"""
        # Load and resize image
        original = cv2.imread(image_path)
        if original is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        h, w = original.shape[:2]
        if max(h, w) > max_dimension:
            scale = max_dimension / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            self.original_image = cv2.resize(original, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"Resized image from {w}x{h} to {new_w}x{new_h}")
        else:
            self.original_image = original
        
        # Initialize MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Detect landmarks
        self.face_landmarks = self._detect_landmarks()
        if not self.face_landmarks:
            raise ValueError("Could not detect face landmarks")
        
        print(f"Detected {len(self.face_landmarks)} facial landmarks")
        
        # Extract key facial regions
        self._extract_facial_regions()
        
        # Initialize audio analyzer (if available)
        if AUDIO_ANALYZER_AVAILABLE and AudioAnalyzer:
            try:
                self.audio_analyzer = AudioAnalyzer()
            except Exception as e:
                print(f"Could not initialize audio analyzer: {e}")
                self.audio_analyzer = None
        else:
            self.audio_analyzer = None
        
        self.viseme_sequence = []
        self.current_audio_path = None
        
        # Animation state
        self.current_image = self.original_image.copy()
        self.is_animating = False
        
        # Viseme shapes (mouth landmark offsets for different visemes)
        self._define_viseme_shapes()
    
    def _detect_landmarks(self):
        """Detect facial landmarks"""
        rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = self.original_image.shape[:2]
            
            landmark_points = []
            for landmark in landmarks.landmark:
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                landmark_points.append((x, y))
            
            return landmark_points
        return None
    
    def _extract_facial_regions(self):
        """Extract key facial regions for animation"""
        # Mouth landmarks (MediaPipe indices)
        self.mouth_outer_indices = [61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321]
        self.mouth_inner_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        
        # Get mouth points
        self.mouth_outer_points = [self.face_landmarks[i] for i in self.mouth_outer_indices if i < len(self.face_landmarks)]
        self.mouth_inner_points = [self.face_landmarks[i] for i in self.mouth_inner_indices if i < len(self.face_landmarks)]
        
        # Calculate mouth center and lips-only bounds
        all_mouth_points = self.mouth_outer_points + self.mouth_inner_points
        if all_mouth_points:
            x_coords = [p[0] for p in all_mouth_points]
            y_coords = [p[1] for p in all_mouth_points]
            self.mouth_center = (int(np.mean(x_coords)), int(np.mean(y_coords)))
            self.mouth_width = max(x_coords) - min(x_coords)
            self.mouth_height = max(y_coords) - min(y_coords)
            
            h_img, w_img = self.original_image.shape[:2]
            pad_x = max(4, int(self.mouth_width * 0.25))
            pad_y = max(3, int(self.mouth_height * 0.4))
            self.mouth_x1 = max(0, min(x_coords) - pad_x)
            self.mouth_x2 = min(w_img, max(x_coords) + pad_x)
            self.mouth_y1 = max(0, min(y_coords) - pad_y)
            self.mouth_y2 = min(h_img, max(y_coords) + pad_y)
            
            print(f"Mouth: center={self.mouth_center}, lips box ({self.mouth_x1},{self.mouth_y1})-({self.mouth_x2},{self.mouth_y2})")
        
        # Eye landmarks (for future eye animation)
        self.left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    
    def _define_viseme_shapes(self):
        """Define mouth shapes for different visemes"""
        # Base mouth shape (rest position)
        self.viseme_shapes = {
            0: {'open': 0.0, 'round': 0.0, 'stretch': 0.0},  # Rest/closed
            1: {'open': 1.0, 'round': 0.0, 'stretch': 0.8},  # Wide open (A, E, I)
            2: {'open': 0.9, 'round': 1.0, 'stretch': 0.0},  # Round open (O, U)
            3: {'open': 0.0, 'round': 0.0, 'stretch': 0.0},  # Closed (M, B, P)
            4: {'open': 0.3, 'round': 0.0, 'stretch': 0.5},  # Fricatives (F, V)
            5: {'open': 0.2, 'round': 0.0, 'stretch': 0.3},   # TH
            6: {'open': 0.4, 'round': 0.0, 'stretch': 0.2},  # D, T, N, L
            7: {'open': 0.3, 'round': 0.0, 'stretch': 0.4},  # S, Z, CH, SH
            8: {'open': 0.6, 'round': 0.0, 'stretch': 0.1}, # G, K, R
        }
    
    def _apply_viseme_morph(self, viseme_id, intensity):
        """Apply viseme-based mouth morphing using simple stretch warp (reliable and visible)"""
        if viseme_id not in self.viseme_shapes:
            viseme_id = 0
        
        viseme = self.viseme_shapes[viseme_id]
        
        # Start from original each frame so animation doesn't compound
        self.current_image = self.original_image.copy()
        
        # Calculate amounts in pixels for simple warp
        open_amount = viseme['open'] * intensity * self.mouth_height * 2.0
        round_amount = viseme['round'] * intensity * self.mouth_width * 0.3
        stretch_amount = viseme['stretch'] * intensity * self.mouth_width * 0.2
        
        self._apply_simple_mouth_warp(open_amount, round_amount, stretch_amount)
    
    def _apply_simple_mouth_warp(self, open_amount, round_amount, stretch_amount):
        """Lips-only warp: stretch downward so upper lip stays fixed, only lower lip moves."""
        h, w = self.current_image.shape[:2]
        if not getattr(self, 'mouth_y1', None):
            return
        
        open_px = int(open_amount)
        if open_px <= 0:
            return
        
        # Lips-only bounding box (from landmarks)
        x1, x2 = self.mouth_x1, self.mouth_x2
        y1, y2 = self.mouth_y1, self.mouth_y2
        if x2 <= x1 or y2 <= y1:
            return
        
        mouth_roi = self.original_image[y1:y2, x1:x2].copy()
        if mouth_roi.size == 0:
            return
        
        roi_h, roi_w = mouth_roi.shape[:2]
        new_height = roi_h + open_px
        stretched = cv2.resize(mouth_roi, (roi_w, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Anchor top: place so upper lip stays at same y (stretch goes downward)
        stretch_y1 = y1
        stretch_y2 = min(h, y1 + new_height)
        stretch_x1, stretch_x2 = x1, x2
        
        actual_h = stretch_y2 - stretch_y1
        actual_w = stretch_x2 - stretch_x1
        if actual_h <= 0 or actual_w <= 0:
            return
        
        if stretched.shape[0] > actual_h or stretched.shape[1] != actual_w:
            stretched = cv2.resize(stretched, (actual_w, actual_h), interpolation=cv2.INTER_LINEAR)
        if stretched.shape[0] != actual_h or stretched.shape[1] != actual_w:
            return
        
        rx, ry = max(1, actual_w // 2), max(1, actual_h // 2)
        center_x, center_y = actual_w // 2, actual_h // 2
        y_coords, x_coords = np.ogrid[:actual_h, :actual_w]
        ellipse_mask = ((x_coords - center_x)**2 / (rx**2 + 1e-6) +
                       (y_coords - center_y)**2 / (ry**2 + 1e-6)) <= 1.0
        mask = ellipse_mask.astype(np.float32)
        k = min(15, actual_h, actual_w)
        if k % 2 == 0:
            k -= 1
        if k >= 3:
            mask = cv2.GaussianBlur(mask, (k, k), 0)
        
        roi = self.current_image[stretch_y1:stretch_y2, stretch_x1:stretch_x2]
        if roi.shape == stretched.shape:
            for c in range(3):
                roi[:, :, c] = (roi[:, :, c] * (1 - mask) + stretched[:, :, c] * mask).astype(np.uint8)
            self.current_image[stretch_y1:stretch_y2, stretch_x1:stretch_x2] = roi
    
    def _apply_tps_warp(self, source_points, target_points):
        """Apply Thin-Plate Spline warping to mouth region"""
        if len(source_points) < 4:
            return
        
        h, w = self.original_image.shape[:2]
        
        # Create a mask for the mouth region
        mouth_region_size = int(max(self.mouth_width, self.mouth_height) * 2.5)
        cx, cy = self.mouth_center
        
        x1 = max(0, cx - mouth_region_size)
        x2 = min(w, cx + mouth_region_size)
        y1 = max(0, cy - mouth_region_size)
        y2 = min(h, cy + mouth_region_size)
        
        # Create grid for warping
        y_coords, x_coords = np.mgrid[y1:y2, x1:x2]
        grid_points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        
        # Calculate displacement using radial basis function (simplified TPS)
        displacements = []
        for gp in grid_points:
            dx, dy = 0.0, 0.0
            total_weight = 0.0
            
            for i, (sp, tp) in enumerate(zip(source_points, target_points)):
                # Distance from grid point to source point
                dist = np.sqrt((gp[0] - sp[0])**2 + (gp[1] - sp[1])**2) + 1e-6
                # Weight decreases with distance
                weight = 1.0 / (dist**2 + 1.0)
                
                # Displacement vector
                disp_x = (tp[0] - sp[0]) * weight
                disp_y = (tp[1] - sp[1]) * weight
                
                dx += disp_x
                dy += disp_y
                total_weight += weight
            
            if total_weight > 0:
                dx /= total_weight
                dy /= total_weight
            
            displacements.append((dx, dy))
        
        # Apply warping
        self.current_image = self.original_image.copy()
        warped_region = self.original_image[y1:y2, x1:x2].copy()
        
        # Create remap arrays
        map_x = np.zeros((y2-y1, x2-x1), dtype=np.float32)
        map_y = np.zeros((y2-y1, x2-x1), dtype=np.float32)
        
        for idx, (gp, (dx, dy)) in enumerate(zip(grid_points, displacements)):
            local_x = idx % (x2-x1)
            local_y = idx // (x2-x1)
            
            new_x = gp[0] + dx - x1
            new_y = gp[1] + dy - y1
            
            map_x[local_y, local_x] = new_x
            map_y[local_y, local_x] = new_y
        
        # Clamp values
        map_x = np.clip(map_x, 0, x2-x1-1)
        map_y = np.clip(map_y, 0, y2-y1-1)
        
        # Remap
        try:
            remapped = cv2.remap(warped_region, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            
            # Blend with original
            mask = np.ones((y2-y1, x2-x1), dtype=np.float32)
            # Create elliptical mask
            center_x, center_y = (x2-x1)//2, (y2-y1)//2
            y_coords, x_coords = np.ogrid[:y2-y1, :x2-x1]
            ellipse_mask = ((x_coords - center_x)**2 / (mouth_region_size**2) + 
                           (y_coords - center_y)**2 / (mouth_region_size**2)) <= 1.0
            mask = mask * ellipse_mask.astype(np.float32)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)
            
            # Blend
            for c in range(3):
                self.current_image[y1:y2, x1:x2, c] = (
                    self.original_image[y1:y2, x1:x2, c] * (1 - mask) + 
                    remapped[:, :, c] * mask
                ).astype(np.uint8)
        except Exception as e:
            print(f"Warping error: {e}")
    
    def animate_with_audio(self, audio_path, duration=None, update_callback=None):
        """Animate face synchronized with audio"""
        import os  # Ensure os is available
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            # Fallback to basic animation
            self._fallback_animation(duration, update_callback)
            return
        
        if not self.audio_analyzer:
            print("Audio analyzer not available, using fallback animation")
            self._fallback_animation(duration, update_callback)
            return
        
        try:
            # Analyze audio
            print("Analyzing audio for visemes...")
            self.viseme_sequence = self.audio_analyzer.analyze_audio(audio_path)
            self.current_audio_path = audio_path
            
            if not self.viseme_sequence:
                print("Could not extract visemes from audio, using fallback")
                self._fallback_animation(duration, update_callback)
                return
            
            # Get duration from audio or use provided
            if duration is None:
                duration = self.viseme_sequence[-1][0] if self.viseme_sequence else 3.0
            
            print(f"Starting audio-synchronized animation for {duration:.2f} seconds")
            print(f"Found {len(self.viseme_sequence)} viseme frames")
            self.is_animating = True
            start_time = time.time()
            fps = 25
            frame_time = 1.0 / fps
            frame_count = 0
            
            while time.time() - start_time < duration and self.is_animating:
                elapsed = time.time() - start_time
                
                # Get viseme at current time
                viseme_id, intensity = self.audio_analyzer.get_viseme_at_time(
                    self.viseme_sequence, elapsed
                )
                
                # Apply viseme morph
                self._apply_viseme_morph(viseme_id, intensity)
                
                # Update display every few frames
                if update_callback and frame_count % 2 == 0:
                    update_callback()
                
                frame_count += 1
                time.sleep(frame_time)
            
            # Reset
            self.current_image = self.original_image.copy()
            self.is_animating = False
            if update_callback:
                update_callback()
            print("Animation finished")
        except Exception as e:
            print(f"Error in advanced animation: {e}")
            import traceback
            traceback.print_exc()
            self._fallback_animation(duration, update_callback)
    
    def _fallback_animation(self, duration=3.0, update_callback=None):
        """Fallback to simple animation if advanced fails"""
        print("Using fallback animation")
        self.is_animating = True
        start_time = time.time()
        fps = 20
        frame_time = 1.0 / fps
        
        while time.time() - start_time < duration and self.is_animating:
            elapsed = time.time() - start_time
            progress = (elapsed % 0.2) / 0.2
            mouth_open = abs(np.sin(progress * 2 * np.pi)) * 0.7
            
            # Simple viseme application
            viseme_id = 1 if mouth_open > 0.3 else 0
            self._apply_viseme_morph(viseme_id, mouth_open)
            
            if update_callback:
                update_callback()
            
            time.sleep(frame_time)
        
        self.current_image = self.original_image.copy()
        self.is_animating = False
        if update_callback:
            update_callback()
    
    def animate_mouth(self, duration=3.0, intensity=0.7, update_callback=None):
        """Basic mouth animation (same interface as FaceAnimator for main.py fallback)"""
        if not getattr(self, 'mouth_center', None):
            self._fallback_animation(duration, update_callback)
            return
        self.is_animating = True
        start_time = time.time()
        fps = 20
        frame_time = 1.0 / fps
        while time.time() - start_time < duration and self.is_animating:
            elapsed = time.time() - start_time
            progress = (elapsed % 0.15) / 0.15
            mouth_open = abs(np.sin(progress * 2 * np.pi)) * intensity
            viseme_id = 1 if mouth_open > 0.2 else 0
            self._apply_viseme_morph(viseme_id, mouth_open)
            if update_callback:
                update_callback()
            time.sleep(frame_time)
        self.current_image = self.original_image.copy()
        self.is_animating = False
        if update_callback:
            update_callback()
    
    def get_current_frame(self):
        """Get current animated frame"""
        return self.current_image.copy()
    
    def stop_animation(self):
        """Stop animation"""
        self.is_animating = False
        self.current_image = self.original_image.copy()
