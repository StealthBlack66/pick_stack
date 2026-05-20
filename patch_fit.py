import re

with open('15_바둑판_정렬.py', 'r') as f:
    content = f.read()

# 1. Update _fit_cube_top_face_25mm
old_fit = """    def _fit_cube_top_face_25mm(self, depth_frame, bbox):
        \"\"\"25mm cube prior + depth 로 cube top face 의 정밀 (center, yaw, 4 corners) 추정.

        YOLO model 의 bbox/polygon 부정확성을 우회. depth + 25mm prior 가 권위.

        원리:
          1. bbox 안 depth 의 5% percentile = cube top 표면 (가장 가까운 z)
          2. top ± 3mm 안의 픽셀만 → cube top face mask (옆면/배경 제외)
          3. 각 mask 픽셀을 카메라 → base frame XY 평면으로 변환
          4. base XY 점 cloud 의 minAreaRect → centroid (진짜 cube 중심) + 회전각
          5. 25mm 정사각형 prior 로 4 코너 강제 (mask noise 영향 X)

        반환: {'center_xyz_mm': (x,y,z), 'cube_yaw_deg': yaw,
               'corners_base_mm': [(x,y,z) x4], 'n_points': int} 또는 None (fail)
        \"\"\"
        x1, y1, x2, y2 = bbox
        arr = np.asanyarray(depth_frame.get_data())   # uint16 mm
        H, W = arr.shape
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W, int(x2)); y2 = min(H, int(y2))
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None

        crop = arr[y1:y2, x1:x2].astype(np.float32) * 0.001    # m
        valid = crop[crop > 0.05]
        if valid.size < 50:
            return None
        z_top_m = float(np.percentile(valid, 5))

        # cube top face mask — 표면 ± 3mm (옆면은 z 가 더 멀어서 제외됨)
        mask_local = ((crop >= z_top_m - 0.003) & (crop <= z_top_m + 0.003))
        if mask_local.sum() < 30:
            return None

        # 각 mask 픽셀을 base frame 좌표로 변환
        ys_local, xs_local = np.where(mask_local)
        base_pts = []
        for v_loc, u_loc in zip(ys_local, xs_local):
            u = int(u_loc + x1)
            v = int(v_loc + y1)
            d = float(arr[v, u]) * 0.001
            if d < 0.05:
                continue
            cam_xyz = rs.rs2_deproject_pixel_to_point(self.intr, [u, v], d)
            cam_h = np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0])
            base = self.T_cam2base @ cam_h
            base_pts.append(base[:3] * 1000.0)   # mm
        if len(base_pts) < 20:
            return None
        base_pts = np.array(base_pts, dtype=np.float32)

        # base XY 평면의 minAreaRect — 0.1mm 정밀도로 정수화 후 cv2 호출
        xy_int = (base_pts[:, :2] * 10.0).astype(np.int32)
        rect = cv2.minAreaRect(xy_int)
        (rcx_t, rcy_t), (rw_t, rh_t), angle_deg = rect
        centroid_x = rcx_t / 10.0
        centroid_y = rcy_t / 10.0
        z_mm = float(base_pts[:, 2].mean())

        # 4-fold 대칭 정규화 (-45 ~ +45)
        yaw_deg = float(angle_deg)"""

new_fit = """    def _fit_cube_top_face_25mm(self, depth_frame, bbox):
        \"\"\"25mm cube prior + depth 로 cube top face 의 정밀 (center, yaw, 4 corners) 추정.

        YOLO model 의 bbox/polygon 은 측면(side face)까지 포함하기 때문에,
        가장 가까운 깊이(z_top)를 기준으로 상단면만 분리합니다.

        반환: {'center_xyz_mm': (x,y,z), 'cube_yaw_deg': yaw,
               'corners_base_mm': [(x,y,z) x4], 'n_points': int} 또는 None (fail)
        \"\"\"
        x1, y1, x2, y2 = bbox
        arr = np.asanyarray(depth_frame.get_data())   # uint16 mm
        H, W = arr.shape
        x1 = max(0, int(x1)); y1 = max(0, int(y1))
        x2 = min(W, int(x2)); y2 = min(H, int(y2))
        if x2 - x1 < 5 or y2 - y1 < 5:
            return None

        crop = arr[y1:y2, x1:x2].astype(np.float32) * 0.001    # m
        valid = crop[crop > 0.05]
        if valid.size < 30:
            return None
        z_top_m = float(np.percentile(valid, 5))

        # cube top face mask — 표면 ± 8mm (측면은 최소 15~25mm 이상 멀리 있으므로 분리됨)
        mask_local = ((crop >= z_top_m - 0.008) & (crop <= z_top_m + 0.008)).astype(np.uint8) * 255
        
        # 노이즈 및 구멍 채우기
        import cv2 as _cv2
        k = _cv2.getStructuringElement(_cv2.MORPH_RECT, (3, 3))
        mask_local = _cv2.morphologyEx(mask_local, _cv2.MORPH_OPEN, k)
        mask_local = _cv2.morphologyEx(mask_local, _cv2.MORPH_CLOSE, k)

        ys_local, xs_local = np.where(mask_local > 0)
        if len(ys_local) < 20:
            return None

        # 각 mask 픽셀을 base frame 좌표로 변환
        base_pts = []
        for v_loc, u_loc in zip(ys_local, xs_local):
            u = int(u_loc + x1)
            v = int(v_loc + y1)
            d = float(arr[v, u]) * 0.001
            if d < 0.05:
                continue
            cam_xyz = rs.rs2_deproject_pixel_to_point(self.intr, [u, v], d)
            cam_h = np.array([cam_xyz[0], cam_xyz[1], cam_xyz[2], 1.0])
            base = self.T_cam2base @ cam_h
            base_pts.append(base[:3] * 1000.0)   # mm
        if len(base_pts) < 20:
            return None
        base_pts = np.array(base_pts, dtype=np.float32)

        # 중심점: 모서리가 깎인 포인트 클라우드에서 minAreaRect 중심을 쓰면 왜곡되므로, 점들의 무게중심(평균) 사용
        centroid_x = float(base_pts[:, 0].mean())
        centroid_y = float(base_pts[:, 1].mean())
        z_mm = float(base_pts[:, 2].mean())

        # base XY 평면의 minAreaRect — 0.1mm 정밀도로 정수화 후 cv2 호출하여 회전각(yaw)만 가져옴
        xy_int = (base_pts[:, :2] * 10.0).astype(np.int32)
        rect = _cv2.minAreaRect(xy_int)
        _, _, angle_deg = rect

        # 4-fold 대칭 정규화 (-45 ~ +45)
        yaw_deg = float(angle_deg)"""

content = content.replace(old_fit, new_fit)

# 2. Update detect_all to modify u, v
old_detect = """            # 25mm 기하학 fit 우선 — depth + cube 크기 prior 로 center+yaw 동시 정확.
            # 성공하면 모델 polygon / yaw 보다 더 권위 있는 결과 (depth 가 noise 만 없으면).
            fit = self._fit_cube_top_face_25mm(df, bbox)
            if fit is not None:
                base = fit['center_xyz_mm']   # YOLO bbox center 대신 25mm prior fit center
                yaw = fit['cube_yaw_deg']
                yaw_src = f'fit25(n={fit["n_points"]})'
            else:"""

new_detect = """            # 25mm 기하학 fit 우선 — depth + cube 크기 prior 로 center+yaw 동시 정확.
            # 성공하면 모델 polygon / yaw 보다 더 권위 있는 결과 (depth 가 noise 만 없으면).
            fit = self._fit_cube_top_face_25mm(df, bbox)
            if fit is not None:
                base = fit['center_xyz_mm']   # YOLO bbox center 대신 25mm prior fit center
                yaw = fit['cube_yaw_deg']
                yaw_src = f'fit25(n={fit["n_points"]})'
                
                # 화면상의 빨간 점(u, v)도 3D 상단면 중심점을 다시 2D 픽셀로 투영하여 갱신
                # (YOLO 마스크는 측면을 포함하므로 중심이 약간 아래로 쏠림)
                T_base2cam = np.linalg.inv(self.T_cam2base)
                p_base_m = np.array([base[0]/1000.0, base[1]/1000.0, base[2]/1000.0, 1.0])
                cam_m = T_base2cam @ p_base_m
                pixel = rs.rs2_project_point_to_pixel(self.intr, cam_m[:3])
                u, v = int(round(pixel[0])), int(round(pixel[1]))
            else:"""

content = content.replace(old_detect, new_detect)

with open('15_바둑판_정렬.py', 'w') as f:
    f.write(content)
print("Patched completely!")
