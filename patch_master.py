import re

with open('15_바둑판_정렬.py', 'r') as f:
    content = f.read()

# 1. Insert the robust top face extraction logic before _yaw_from_depth_bbox
robust_logic = """
    def _compute_top_face_center_and_yaw(self, color, depth_frame, bbox):
        \"\"\"
        YOLO bounding box 안에서 깊이(Depth)를 이용해 큐브의 '순수 상단면'만 분리하고,
        해당 마스크를 RGB 이미지에 덮어씌워 선명한 엣지로 회전각(yaw)을 구합니다.
        \"\"\"
        x1, y1, x2, y2 = bbox
        arr = np.asanyarray(depth_frame.get_data())
        H, W = arr.shape
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W, int(x2)); y2 = min(H, int(y2))
        if x2 - x1 < 5 or y2 - y1 < 5: return None
        
        crop = arr[y1:y2, x1:x2].astype(np.float32) * 0.001
        valid = crop[crop > 0.05]
        if valid.size < 30: return None
        z_top_m = float(np.percentile(valid, 5))

        # 상단면 분리 (±8mm 여유)
        mask_local = ((crop >= z_top_m - 0.008) & (crop <= z_top_m + 0.008)).astype(np.uint8) * 255
        import cv2 as _cv2
        import math
        import pyrealsense2 as _rs
        
        k = _cv2.getStructuringElement(_cv2.MORPH_RECT, (3, 3))
        mask_local = _cv2.morphologyEx(mask_local, _cv2.MORPH_OPEN, k)
        mask_local = _cv2.morphologyEx(mask_local, _cv2.MORPH_CLOSE, k)
        
        ys_local, xs_local = np.where(mask_local > 0)
        if len(ys_local) < 20: return None

        # 중심점 구하기 (점들의 평균)
        base_pts = []
        for v_loc, u_loc in zip(ys_local, xs_local):
            u = int(u_loc + x1)
            v = int(v_loc + y1)
            d = float(arr[v, u]) * 0.001
            if d < 0.05: continue
            cam = _rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], d)
            cam_h = np.array([cam[0], cam[1], cam[2], 1.0])
            bp = (self.T_cam2base @ cam_h)[:3] * 1000.0
            base_pts.append(bp)
        if len(base_pts) < 20: return None
        base_pts = np.array(base_pts, dtype=np.float32)
        cx = float(base_pts[:, 0].mean())
        cy = float(base_pts[:, 1].mean())
        cz = float(base_pts[:, 2].mean())

        # RGB 엣지로 Yaw 구하기
        color_crop = color[y1:y2, x1:x2]
        gray = _cv2.cvtColor(color_crop, _cv2.COLOR_BGR2GRAY)
        gray = _cv2.GaussianBlur(gray, (3, 3), 0)
        edges = _cv2.Canny(gray, 30, 100)
        
        k7 = _cv2.getStructuringElement(_cv2.MORPH_RECT, (7, 7))
        mask_dilated = _cv2.dilate(mask_local, k7)
        edges = _cv2.bitwise_and(edges, edges, mask=mask_dilated)
        
        lines = _cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=10, minLineLength=5, maxLineGap=5)
        
        yaw_deg = None
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
                
                # 2D 각도를 3D 평면에 투영하여 원근 보정
                cu = (x1 + x2) / 2.0
                cv_ = (y1 + y2) / 2.0
                half_px = 15.0
                rad = math.radians(yaw_img)
                cs, sn = math.cos(rad), math.sin(rad)
                local = np.array([[-half_px, -half_px], [half_px, -half_px],
                                  [half_px, half_px], [-half_px, half_px]], dtype=np.float32)
                R = np.array([[cs, -sn], [sn, cs]], dtype=np.float32)
                box_px = (local @ R.T) + np.array([cu, cv_], dtype=np.float32)
                
                bp_virtual = []
                for pt in box_px:
                    c = _rs.rs2_deproject_pixel_to_point(self.intr, [float(pt[0]), float(pt[1])], z_top_m)
                    ch = np.array([c[0], c[1], c[2], 1.0])
                    b = (self.T_cam2base @ ch)[:3] * 1000.0
                    bp_virtual.append([b[0], b[1]])
                
                bp_virtual = np.array(bp_virtual, dtype=np.float32)
                rect_virtual = _cv2.minAreaRect(bp_virtual)
                yaw_deg = float(rect_virtual[2])
                while yaw_deg > 45.0: yaw_deg -= 90.0
                while yaw_deg <= -45.0: yaw_deg += 90.0
                
        if yaw_deg is None:
            xy_int = (base_pts[:, :2] * 10.0).astype(np.int32)
            rect = _cv2.minAreaRect(xy_int)
            yaw_deg = float(rect[2])
            while yaw_deg > 45.0: yaw_deg -= 90.0
            while yaw_deg <= -45.0: yaw_deg += 90.0
            
        return {'center_xyz_mm': (cx, cy, cz), 'cube_yaw_deg': yaw_deg, 'yaw_src': 'robust'}

    def _yaw_from_depth_bbox(self, depth_frame, bbox):
"""

content = re.sub(r'    def _yaw_from_depth_bbox\(self, depth_frame, bbox\):', robust_logic.lstrip('\n'), content)

# 2. Update detect_all
old_detect_logic = """            # 잡기 픽셀: mask centroid (cube top 무게중심). 없으면 bbox center.
            cen = self._polygon_centroid_px(poly)
            if cen is None:
                u, v = int(cx), int(cy)
            else:
                u, v = cen
            # base 좌표: mask 영역의 depth percentile (cube top 표면) 기반.
            base = self._polygon_to_base(df, poly, (u, v))
            if base is None:
                continue
            bbox = (int(cx - w / 2), int(cy - h / 2),
                    int(cx + w / 2), int(cy + h / 2))

            # secondary (yaw 전용) 모델 inference — 한 번에 처리 후 nearest bbox match.
            secondary_poly = find_yaw_polygon(u, v)
            poly_for_yaw = secondary_poly if secondary_poly is not None else poly

            yaw, yaw_src = self._refine_yaw_multi(color, df, bbox, poly_for_yaw)
            if secondary_poly is not None and yaw_src == 'poly':
                yaw_src = 'poly_yaw_model'

            rhombus_pts = self._get_projected_rhombus(base, yaw)
            out.append({"""

new_detect_logic = """            # 기본값 설정
            cen = self._polygon_centroid_px(poly)
            if cen is None:
                u, v = int(cx), int(cy)
            else:
                u, v = cen
            base = self._polygon_to_base(df, poly, (u, v))
            if base is None:
                continue
            bbox = (int(cx - w / 2), int(cy - h / 2),
                    int(cx + w / 2), int(cy + h / 2))

            # 정밀한 상단면 고립 및 회전각 추출
            fit = self._compute_top_face_center_and_yaw(color, df, bbox)
            if fit is not None:
                base = fit['center_xyz_mm']
                yaw = fit['cube_yaw_deg']
                yaw_src = fit['yaw_src']
                # 3D 상단면 중심점을 2D 화면으로 재투영하여 빨간 점 위치 보정
                T_base2cam = np.linalg.inv(self.T_cam2base)
                p_base_m = np.array([base[0]/1000.0, base[1]/1000.0, base[2]/1000.0, 1.0])
                cam_m = T_base2cam @ p_base_m
                import pyrealsense2 as _rs
                pixel = _rs.rs2_project_point_to_pixel(self.intr, cam_m[:3])
                u, v = int(round(pixel[0])), int(round(pixel[1]))
            else:
                secondary_poly = find_yaw_polygon(u, v)
                poly_for_yaw = secondary_poly if secondary_poly is not None else poly
                yaw, yaw_src = self._refine_yaw_multi(color, df, bbox, poly_for_yaw)
                if secondary_poly is not None and yaw_src == 'poly':
                    yaw_src = 'poly_yaw_model'

            rhombus_pts = self._get_projected_rhombus(base, yaw)
            out.append({"""

content = content.replace(old_detect_logic, new_detect_logic)

with open('15_바둑판_정렬.py', 'w') as f:
    f.write(content)
print("Master logic patched successfully!")
