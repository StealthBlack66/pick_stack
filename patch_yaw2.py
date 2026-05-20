import re

with open('15_바둑판_정렬.py', 'r') as f:
    content = f.read()

old_yaw_logic = """        color_crop = color[y1:y2, x1:x2]
        gray = _cv2.cvtColor(color_crop, _cv2.COLOR_BGR2GRAY)
        gray = _cv2.GaussianBlur(gray, (3, 3), 0)
        edges = _cv2.Canny(gray, 50, 150)
        
        # 상단면 마스크를 약간 팽창시켜 실제 RGB 경계선을 포함하도록 함
        k3 = _cv2.getStructuringElement(_cv2.MORPH_RECT, (5, 5))
        mask_dilated = _cv2.dilate(mask_local, k3)
        edges = _cv2.bitwise_and(edges, edges, mask=mask_dilated)
        
        min_len = max(8, int(0.3 * min(x2 - x1, y2 - y1)))
        lines = _cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=15, minLineLength=min_len, maxLineGap=5)
        
        yaw_deg = None
        import math
        if lines is not None and len(lines) >= 1:
            f = 4.0
            sin_sum, cos_sum, n = 0.0, 0.0, 0
            for x_1, y_1, x_2, y_2 in lines.reshape(-1, 4):
                if x_1 == x_2 and y_1 == y_2: continue
                a = math.atan2(y_2 - y_1, x_2 - x_1)
                sin_sum += math.sin(a * f)
                cos_sum += math.cos(a * f)
                n += 1
            if n > 0:
                yaw_img = math.degrees(math.atan2(sin_sum, cos_sum)) / f
                while yaw_img > 45.0: yaw_img -= 90.0
                while yaw_img <= -45.0: yaw_img += 90.0
                
                # 이미지 평면 각도를 3D Base 각도로 변환 (원근 왜곡 보정)
                cu = (x1 + x2) / 2.0
                cv_ = (y1 + y2) / 2.0
                half_px = 0.4 * min(x2 - x1, y2 - y1)
                rad = math.radians(yaw_img)
                cs, sn = math.cos(rad), math.sin(rad)
                local = np.array([[-half_px, -half_px], [half_px, -half_px],
                                  [half_px, half_px], [-half_px, half_px]], dtype=np.float32)
                R = np.array([[cs, -sn], [sn, cs]], dtype=np.float32)
                box_px = (local @ R.T) + np.array([cu, cv_], dtype=np.float32)
                
                yaw_deg = self._polygon_to_base_yaw(depth_frame, box_px)
        
        # RGB 라인 추출 실패 시 fallback (기존 Depth 기반)
        if yaw_deg is None:
            xy_int = (base_pts[:, :2] * 10.0).astype(np.int32)
            rect = _cv2.minAreaRect(xy_int)
            _, _, angle_deg = rect
            yaw_deg = float(angle_deg)
            while yaw_deg > 45.0: yaw_deg -= 90.0
            while yaw_deg <= -45.0: yaw_deg += 90.0"""

new_yaw_logic = """        color_crop = color[y1:y2, x1:x2]
        gray = _cv2.cvtColor(color_crop, _cv2.COLOR_BGR2GRAY)
        gray = _cv2.GaussianBlur(gray, (3, 3), 0)
        edges = _cv2.Canny(gray, 30, 100)  # 나무 블록 대비를 잘 잡도록 임계값 낮춤
        
        # 상단면 마스크를 여유있게 팽창시켜 상단면의 진짜 RGB 경계선을 확실히 포함하도록 함
        # 측면 세로선은 팽창된 마스크 바깥에 위치하므로 자연스럽게 필터링됨
        k7 = _cv2.getStructuringElement(_cv2.MORPH_RECT, (7, 7))
        mask_dilated = _cv2.dilate(mask_local, k7)
        edges = _cv2.bitwise_and(edges, edges, mask=mask_dilated)
        
        lines = _cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=5, maxLineGap=5)
        
        yaw_deg = None
        import math
        import pyrealsense2 as _rs
        if lines is not None and len(lines) >= 1:
            f = 4.0
            sin_sum, cos_sum, n = 0.0, 0.0, 0
            for x_1, y_1, x_2, y_2 in lines.reshape(-1, 4):
                if x_1 == x_2 and y_1 == y_2: continue
                a = math.atan2(y_2 - y_1, x_2 - x_1)
                sin_sum += math.sin(a * f)
                cos_sum += math.cos(a * f)
                n += 1
            if n > 0:
                yaw_img = math.degrees(math.atan2(sin_sum, cos_sum)) / f
                while yaw_img > 45.0: yaw_img -= 90.0
                while yaw_img <= -45.0: yaw_img += 90.0
                
                # 원근 왜곡을 제거하기 위해 2D 회전 사각형을 3D 상단면 평면에 직접 투영
                # 기존 polygon_to_base_yaw는 모서리의 depth를 맵에서 다시 읽어오기 때문에 테이블 바닥 노이즈에 취약함.
                # 상단면의 평면 높이(z_top_m)를 고정하여 4개의 꼭짓점을 투영하면 완벽한 3D 평행사변형이 됨!
                cu = (x1 + x2) / 2.0
                cv_ = (y1 + y2) / 2.0
                half_px = 15.0
                rad = math.radians(yaw_img)
                cs, sn = math.cos(rad), math.sin(rad)
                local = np.array([[-half_px, -half_px], [half_px, -half_px],
                                  [half_px, half_px], [-half_px, half_px]], dtype=np.float32)
                R = np.array([[cs, -sn], [sn, cs]], dtype=np.float32)
                box_px = (local @ R.T) + np.array([cu, cv_], dtype=np.float32)
                
                base_pts_virtual = []
                for pt in box_px:
                    cam = _rs.rs2_deproject_pixel_to_point(self.intr, [float(pt[0]), float(pt[1])], z_top_m)
                    cam_h = np.array([cam[0], cam[1], cam[2], 1.0])
                    bp = (self.T_cam2base @ cam_h)[:3] * 1000.0
                    base_pts_virtual.append([bp[0], bp[1]])
                
                base_pts_virtual = np.array(base_pts_virtual, dtype=np.float32)
                rect_virtual = _cv2.minAreaRect(base_pts_virtual)
                yaw_deg = float(rect_virtual[2])
                while yaw_deg > 45.0: yaw_deg -= 90.0
                while yaw_deg <= -45.0: yaw_deg += 90.0
        
        # RGB 라인 추출 실패 시 fallback (기존 Depth 기반)
        if yaw_deg is None:
            xy_int = (base_pts[:, :2] * 10.0).astype(np.int32)
            rect = _cv2.minAreaRect(xy_int)
            _, _, angle_deg = rect
            yaw_deg = float(angle_deg)
            while yaw_deg > 45.0: yaw_deg -= 90.0
            while yaw_deg <= -45.0: yaw_deg += 90.0"""

if old_yaw_logic in content:
    content = content.replace(old_yaw_logic, new_yaw_logic)
    with open('15_바둑판_정렬.py', 'w') as f:
        f.write(content)
    print("Yaw logic patched successfully!")
else:
    print("Failed to find old yaw logic.")
