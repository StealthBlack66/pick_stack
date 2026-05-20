import re

with open('15_바둑판_정렬.py', 'r') as f:
    content = f.read()

old_logic = """                        # 잡기 픽셀: mask centroid (cube top 무게중심). 없으면 bbox center.
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
                        yaw, yaw_src = self._refine_yaw_multi(color, df, bbox, poly)
                        dets.append({
                            'base_xyz_mm': (float(base[0]), float(base[1]), float(base[2])),
                            'pixel': (u, v),
                            'conf': float(confs[i]),
                            'cube_yaw_deg': yaw,
                            'yaw_src': yaw_src,
                        })
                        draw_meta.append((bbox, float(confs[i]), poly))"""

new_logic = """                        cen = self._polygon_centroid_px(poly)
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

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    with open('15_바둑판_정렬.py', 'w') as f:
        f.write(content)
    print("Preview logic patched successfully!")
else:
    print("Failed to find old logic in preview_until_confirm.")
