import re

with open('15_바둑판_정렬.py', 'r') as f:
    content = f.read()

# 1. Remove _compute_top_face_center_and_yaw completely
# It starts at `def _compute_top_face_center_and_yaw` and ends before `def _yaw_from_depth_bbox`
content = re.sub(r'    def _compute_top_face_center_and_yaw.*?def _yaw_from_depth_bbox', 
                 '    def _yaw_from_depth_bbox', content, flags=re.DOTALL)

# 2. Update detect_all logic
old_detect = """        xywh = r.boxes.xywh.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        polys = (r.masks.xy if r.masks is not None else [None] * len(xywh))

        out = []
        dbg = color.copy() if save_debug else None
        n_reject = 0
        for i in range(len(xywh)):
            cx, cy, w, h = xywh[i]
            poly = polys[i] if i < len(polys) else None
            # 품질 검증 — false positive (cube 아닌 mask) reject
            ok, reason = self._polygon_quality_ok(poly)
            if not ok:
                n_reject += 1
                if dbg is not None:
                    x1, y1 = int(cx - w / 2), int(cy - h / 2)
                    x2, y2 = int(cx + w / 2), int(cy + h / 2)
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 200), 1)  # 빨강
                    cv2.putText(dbg, f'rej:{reason}', (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
                continue
            # 기본값 설정
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
            out.append({
                'base_xyz_mm': (float(base[0]), float(base[1]), float(base[2])),
                'pixel': (u, v),
                'conf': float(confs[i]),
                'cube_yaw_deg': yaw,
                'yaw_src': yaw_src,
                'rhombus_pts': rhombus_pts,
            })
            if dbg is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if poly is not None and len(poly) >= 3:
                    cv2.polylines(dbg, [poly.astype(np.int32)], True, (255, 0, 255), 1)
                cv2.circle(dbg, (u, v), 4, (0, 0, 255), -1)

                # 계산된 3D 정사각형(마름모) 가시화 (노란색 굵은 선)
                if rhombus_pts is not None:
                    cv2.polylines(dbg, [rhombus_pts], True, (0, 255, 255), 2)

                label = f'{confs[i]:.2f}'
                if yaw is not None:
                    label += f' yaw={yaw:+.1f}d[{yaw_src[0].upper()}]'
                cv2.putText(dbg, label, (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if dbg is not None and save_debug:
            cv2.imwrite(save_debug, dbg)
        return out"""

new_detect = """        xywh = r.boxes.xywh.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        cls_ids = r.boxes.cls.cpu().numpy()
        polys = (r.masks.xy if r.masks is not None else [None] * len(xywh))

        out = []
        dbg = color.copy() if save_debug else None
        n_reject = 0
        for i in range(len(xywh)):
            cx, cy, w, h = xywh[i]
            conf = float(confs[i])
            cls_id = int(cls_ids[i])
            poly = polys[i] if i < len(polys) else None
            
            # Side face (클래스 1)는 그리기만 하고 타겟 처리에서 제외
            if cls_id != 0:
                if dbg is not None:
                    x1, y1 = int(cx - w / 2), int(cy - h / 2)
                    x2, y2 = int(cx + w / 2), int(cy + h / 2)
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (255, 0, 0), 1)  # 파란색
                    if poly is not None and len(poly) >= 3:
                        cv2.polylines(dbg, [poly.astype(np.int32)], True, (255, 100, 100), 1)
                    cv2.putText(dbg, f'side', (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                continue

            # Top face (클래스 0) 처리
            ok, reason = self._polygon_quality_ok(poly)
            if not ok:
                n_reject += 1
                if dbg is not None:
                    x1, y1 = int(cx - w / 2), int(cy - h / 2)
                    x2, y2 = int(cx + w / 2), int(cy + h / 2)
                    cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 0, 200), 1)  # 빨강
                    cv2.putText(dbg, f'rej:{reason}', (x1, max(15, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
                continue
                
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

            # 윗면(Top face) 폴리곤을 그대로 이용하여 3D 회전각 추출
            yaw = self._polygon_to_base_yaw(df, poly)
            yaw_src = 'top_poly'
            if yaw is None:
                yaw, yaw_src = 0.0, 'fail'

            rhombus_pts = self._get_projected_rhombus(base, yaw)
            out.append({
                'base_xyz_mm': (float(base[0]), float(base[1]), float(base[2])),
                'pixel': (u, v),
                'conf': conf,
                'cube_yaw_deg': yaw,
                'yaw_src': yaw_src,
                'rhombus_pts': rhombus_pts,
            })
            
            if dbg is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if poly is not None and len(poly) >= 3:
                    cv2.polylines(dbg, [poly.astype(np.int32)], True, (255, 0, 255), 1)
                cv2.circle(dbg, (u, v), 4, (0, 0, 255), -1)

                if rhombus_pts is not None:
                    cv2.polylines(dbg, [rhombus_pts], True, (0, 255, 255), 2)

                label = f'{conf:.2f}'
                if yaw is not None:
                    label += f' y={yaw:+.0f}d[{yaw_src[0].upper()}]'
                cv2.putText(dbg, label, (x1, max(15, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            
        if dbg is not None and save_debug:
            cv2.imwrite(save_debug, dbg)
        return out"""

content = content.replace(old_detect, new_detect)

# 3. Update preview_until_confirm logic
old_preview = """                    confs = r.boxes.conf.cpu().numpy()
                    polys = (r.masks.xy if r.masks is not None else [None] * len(xywh))
                    for i in range(len(xywh)):
                        cx, cy, w, h = xywh[i]
                        poly = polys[i] if i < len(polys) else None
                        # 품질 검증 — false positive reject (즉시 빨간 박스, 트래킹 대상 아님)
                        ok, reason = self._polygon_quality_ok(poly)
                        if not ok:
                            x1, y1 = int(cx - w / 2), int(cy - h / 2)
                            x2, y2 = int(cx + w / 2), int(cy + h / 2)
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 200), 1)
                            cv2.putText(vis, f'rej:{reason}',
                                        (x1, max(15, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
                            continue
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
                                
                        fit = self._compute_top_face_center_and_yaw(color, df, bbox)
                        if fit is not None:
                            base = fit['center_xyz_mm']
                            yaw = fit['cube_yaw_deg']
                            yaw_src = fit['yaw_src']
                            T_base2cam = np.linalg.inv(self.T_cam2base)
                            p_base_m = np.array([base[0]/1000.0, base[1]/1000.0, base[2]/1000.0, 1.0])
                            cam_m = T_base2cam @ p_base_m
                            import pyrealsense2 as _rs
                            pixel = _rs.rs2_project_point_to_pixel(self.intr, cam_m[:3])
                            u, v = int(round(pixel[0])), int(round(pixel[1]))
                        else:
                            yaw, yaw_src = self._refine_yaw_multi(color, df, bbox, poly)
                            
                        dets.append({
                            'base_xyz_mm': (float(base[0]), float(base[1]), float(base[2])),
                            'pixel': (u, v),
                            'conf': float(confs[i]),
                            'cube_yaw_deg': yaw,
                            'yaw_src': yaw_src,
                        })
                        draw_meta.append((bbox, float(confs[i]), poly))"""

new_preview = """                    confs = r.boxes.conf.cpu().numpy()
                    cls_ids = r.boxes.cls.cpu().numpy()
                    polys = (r.masks.xy if r.masks is not None else [None] * len(xywh))
                    for i in range(len(xywh)):
                        cx, cy, w, h = xywh[i]
                        poly = polys[i] if i < len(polys) else None
                        cls_id = int(cls_ids[i])
                        
                        # Side face 그리기
                        if cls_id != 0:
                            x1, y1 = int(cx - w / 2), int(cy - h / 2)
                            x2, y2 = int(cx + w / 2), int(cy + h / 2)
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 1)
                            if poly is not None and len(poly) >= 3:
                                cv2.polylines(vis, [poly.astype(np.int32)], True, (255, 100, 100), 1)
                            continue

                        # 품질 검증 — false positive reject
                        ok, reason = self._polygon_quality_ok(poly)
                        if not ok:
                            x1, y1 = int(cx - w / 2), int(cy - h / 2)
                            x2, y2 = int(cx + w / 2), int(cy + h / 2)
                            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 200), 1)
                            cv2.putText(vis, f'rej:{reason}',
                                        (x1, max(15, y1 - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
                            continue
                            
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
                                
                        yaw = self._polygon_to_base_yaw(df, poly)
                        yaw_src = 'top_poly'
                        if yaw is None:
                            yaw, yaw_src = 0.0, 'fail'
                            
                        dets.append({
                            'base_xyz_mm': (float(base[0]), float(base[1]), float(base[2])),
                            'pixel': (u, v),
                            'conf': float(confs[i]),
                            'cube_yaw_deg': yaw,
                            'yaw_src': yaw_src,
                        })
                        draw_meta.append((bbox, float(confs[i]), poly))"""

content = content.replace(old_preview, new_preview)

with open('15_바둑판_정렬.py', 'w') as f:
    f.write(content)

print("Multi-class architecture logic applied!")
