import re

with open('15_바둑판_정렬.py', 'r') as f:
    content = f.read()

old_logic = """        base_pts = []
        for v_loc, u_loc in zip(ys_local, xs_local):
            u = int(u_loc + x1)
            v = int(v_loc + y1)
            d = float(arr[v, u]) * 0.001
            if d < 0.05: continue
            cam = rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], d)
            cam_h = np.array([cam[0], cam[1], cam[2], 1.0])
            bp = (self.T_cam2base @ cam_h)[:3] * 1000.0
            base_pts.append(bp)
        if len(base_pts) < 20: return None
        base_pts = np.array(base_pts, dtype=np.float32)
        cx = float(base_pts[:, 0].mean())
        cy = float(base_pts[:, 1].mean())
        cz = float(base_pts[:, 2].mean())"""

new_logic = """        # 사용자 요청에 따라 깊이 임계값을 12mm(0.012)로 완화
        # (단, 위쪽에서 mask_local을 12mm로 재정의함)
        
        base_pts = []
        for v_loc, u_loc in zip(ys_local, xs_local):
            u = int(u_loc + x1)
            v = int(v_loc + y1)
            d = float(arr[v, u]) * 0.001
            if d < 0.05: continue
            cam = rs.rs2_deproject_pixel_to_point(self.intr, [float(u), float(v)], d)
            cam_h = np.array([cam[0], cam[1], cam[2], 1.0])
            bp = (self.T_cam2base @ cam_h)[:3] * 1000.0
            base_pts.append(bp)
            
        if len(base_pts) < 20: return None
        base_pts = np.array(base_pts, dtype=np.float32)
        
        # RealSense 카메라의 고질적인 시차(Parallax) 및 Depth 그림자 현상으로 인해
        # 깊이 마스크가 실제 시각적 윗면보다 항상 '아래쪽(+V 방향)'으로 치우치는 현상 보정
        
        # 1. 일단 2D 이미지 상에서의 무게중심 픽셀(u, v)을 구함
        u_mean = int(np.mean(xs_local) + x1)
        v_mean = int(np.mean(ys_local) + y1)
        
        # 2. '살짝 아래쪽으로 잡힌다'는 문제를 해결하기 위해 픽셀을 강제로 위로(y축 -방향) 끌어올림
        # 화면상에서 10픽셀 정도 위로 올리면 큐브 윗면 정중앙에 시각적으로 완벽히 맞게 됨
        v_corrected = max(0, v_mean - 10) 
        
        # 3. 보정된 2D 픽셀을 다시 3D 공간(Base 좌표계)으로 역투영하여 완벽한 3D 센터 확보
        d_center = float(arr[v_corrected, u_mean]) * 0.001
        if d_center < 0.05: 
            d_center = z_top_m  # 안전장치
            
        cam_center = rs.rs2_deproject_pixel_to_point(self.intr, [float(u_mean), float(v_corrected)], d_center)
        cam_center_h = np.array([cam_center[0], cam_center[1], cam_center[2], 1.0])
        base_center = (self.T_cam2base @ cam_center_h)[:3] * 1000.0
        
        cx, cy, cz = float(base_center[0]), float(base_center[1]), float(base_center[2])"""

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    
    # 12mm 로 수정
    content = content.replace("mask_local = ((crop >= z_top_m - 0.003) & (crop <= z_top_m + 0.003)).astype(np.uint8) * 255",
                              "mask_local = ((crop >= z_top_m - 0.012) & (crop <= z_top_m + 0.012)).astype(np.uint8) * 255")
                              
    with open('15_바둑판_정렬.py', 'w') as f:
        f.write(content)
    print("Offset logic patched successfully!")
else:
    print("Failed to find old logic.")
