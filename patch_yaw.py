import re

with open('15_바둑판_정렬.py', 'r') as f:
    content = f.read()

# 1. Update _fit_cube_top_face_25mm definition
old_def = """    def _fit_cube_top_face_25mm(self, depth_frame, bbox):"""
new_def = """    def _fit_cube_top_face_25mm(self, color, depth_frame, bbox):"""
content = content.replace(old_def, new_def)

# 2. Update detect_all call
old_call = """            fit = self._fit_cube_top_face_25mm(df, bbox)"""
new_call = """            fit = self._fit_cube_top_face_25mm(color, df, bbox)"""
content = content.replace(old_call, new_call)

# 3. Update the yaw logic inside _fit_cube_top_face_25mm
old_yaw = """        # base XY 평면의 minAreaRect — 0.1mm 정밀도로 정수화 후 cv2 호출하여 회전각(yaw)만 가져옴
        xy_int = (base_pts[:, :2] * 10.0).astype(np.int32)
        rect = _cv2.minAreaRect(xy_int)
        _, _, angle_deg = rect

        # 4-fold 대칭 정규화 (-45 ~ +45)
        yaw_deg = float(angle_deg)"""

new_yaw = """        # --- 최상단면(Top face)의 정확한 회전각(yaw) 추출 ---
        # 1. Depth 포인트 클라우드의 minAreaRect는 노이즈에 취약하고,
        # 2. YOLO 전체 폴리곤은 측면(Side face)을 포함하여 엉뚱한 각도를 만듭니다.
        # 따라서, 앞서 구한 상단면 마스크(mask_local)를 RGB 이미지에 덧씌워
        # 상단면의 '진짜 모서리(선명한 RGB 엣지)'만 추출합니다.
        
        color_crop = color[y1:y2, x1:x2]
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

content = content.replace(old_yaw, new_yaw)

with open('15_바둑판_정렬.py', 'w') as f:
    f.write(content)
print("Yaw logic patched successfully!")
