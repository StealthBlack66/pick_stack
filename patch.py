import re

with open('15_바둑판_정렬.py', 'r') as f:
    content = f.read()

# Modify _refine_yaw_multi to prioritize poly
old_refine = """    def _refine_yaw_multi(self, color, depth_frame, bbox, polygon_xy):
        \"\"\"cube yaw 를 3가지 방법 순서로 시도:
           1) depth plane fit — 회전 보존, 가장 robust
           2) RGB Canny+Hough — 모서리 또렷할 때
           3) seg polygon — 학습 라벨이 axis-aligned 일 때의 fallback
        반환: (yaw_deg, src_str). 모두 실패하면 (None, 'fail').\"\"\"
        yaw = self._yaw_from_depth_bbox(depth_frame, bbox)
        if yaw is not None:
            return yaw, 'depth'
        yaw = self._yaw_from_edges_bbox(color, depth_frame, bbox)
        if yaw is not None:
            return yaw, 'edges'
        yaw = self._polygon_to_base_yaw(depth_frame, polygon_xy)
        return (yaw, 'poly') if yaw is not None else (None, 'fail')"""

new_refine = """    def _refine_yaw_multi(self, color, depth_frame, bbox, polygon_xy):
        \"\"\"cube yaw 를 3가지 방법 순서로 시도:
           1) seg polygon — 현재 학습된 모델의 mask 가 가장 정확함 (우선 사용)
           2) depth plane fit — fallback
           3) RGB Canny+Hough — fallback
        반환: (yaw_deg, src_str). 모두 실패하면 (None, 'fail').\"\"\"
        yaw = self._polygon_to_base_yaw(depth_frame, polygon_xy)
        if yaw is not None:
            return yaw, 'poly'
        yaw = self._yaw_from_depth_bbox(depth_frame, bbox)
        if yaw is not None:
            return yaw, 'depth'
        yaw = self._yaw_from_edges_bbox(color, depth_frame, bbox)
        if yaw is not None:
            return yaw, 'edges'
        return None, 'fail'"""

content = content.replace(old_refine, new_refine)

# Disable fit25 in detect_all
old_detect = """            # 25mm 기하학 fit 우선 — depth + cube 크기 prior 로 center+yaw 동시 정확.
            # 성공하면 모델 polygon / yaw 보다 더 권위 있는 결과 (depth 가 noise 만 없으면).
            fit = self._fit_cube_top_face_25mm(df, bbox)
            if fit is not None:
                base = fit['center_xyz_mm']   # YOLO bbox center 대신 25mm prior fit center
                yaw = fit['cube_yaw_deg']
                yaw_src = f'fit25(n={fit["n_points"]})'
            else:
                # fit 실패 시 — 기존 path (yaw model polygon → depth → edges → poly fallback)
                secondary_poly = find_yaw_polygon(u, v)
                poly_for_yaw = secondary_poly if secondary_poly is not None else poly
                yaw, yaw_src = self._refine_yaw_multi(color, df, bbox, poly_for_yaw)
                if secondary_poly is not None and yaw_src == 'poly':
                    yaw_src = 'poly_yaw_model'"""

new_detect = """            # 25mm 기하학 fit 우선 — depth + cube 크기 prior 로 center+yaw 동시 정확.
            # 하지만 depth 노이즈로 인해 위치/회전이 빗나가는 문제가 발생하여 비활성화.
            # 항상 poly mask centroid 와 poly 기반 yaw 를 사용하도록 fallback 경로 직접 실행.
            fit = None # self._fit_cube_top_face_25mm(df, bbox)
            if fit is not None:
                base = fit['center_xyz_mm']   # YOLO bbox center 대신 25mm prior fit center
                yaw = fit['cube_yaw_deg']
                yaw_src = f'fit25(n={fit["n_points"]})'
            else:
                # fit 실패 시 — 기존 path (yaw model polygon → depth → edges → poly fallback)
                secondary_poly = find_yaw_polygon(u, v)
                poly_for_yaw = secondary_poly if secondary_poly is not None else poly
                yaw, yaw_src = self._refine_yaw_multi(color, df, bbox, poly_for_yaw)
                if secondary_poly is not None and yaw_src == 'poly':
                    yaw_src = 'poly_yaw_model'"""

content = content.replace(old_detect, new_detect)

with open('15_바둑판_정렬.py', 'w') as f:
    f.write(content)
print("Patched!")
