#!/usr/bin/env python3
"""
정확한 ArUco 마커 PNG 생성기.

캘리브레이션 정확도가 마커 품질에 매우 민감하므로, 손으로 그린/저품질 인쇄 마커
대신 이 스크립트로 생성한 PNG 를 사용 권장.

사용법:
  python3 generate_aruco_marker.py            # 기본: DICT_6X6_50, ID=0, 50mm
  python3 generate_aruco_marker.py --size 70  # 70mm 마커
  python3 generate_aruco_marker.py --id 5     # 다른 ID

출력 PNG 를 100% 스케일로 인쇄 (이미지 편집 X). 인쇄 후 자로 측정해 50.0mm 인지 확인.
딱딱한 평판 (아크릴/카드보드/PVC) 에 양면테이프로 부착해서 사용.
"""
import cv2
import numpy as np
import argparse
import os

DICT = cv2.aruco.DICT_6X6_50  # 캘리브레이션 코드와 일치
DPI_PRINT = 300                # 일반 잉크젯/레이저 프린터 표준 해상도
PADDING_MM = 10                # 마커 주변 흰 여백 (마커 검출에 필수, 1셀 이상 권장)


def generate(marker_id: int, marker_mm: float, out_path: str):
    aruco_dict = cv2.aruco.getPredefinedDictionary(DICT)

    # mm → pixel 환산
    marker_px = int(round(marker_mm / 25.4 * DPI_PRINT))    # 1 inch = 25.4 mm
    padding_px = int(round(PADDING_MM / 25.4 * DPI_PRINT))

    # 1) ArUco 마커 생성 (8x8 셀 = 6x6 inner + 1 border)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_px)

    # 2) 흰 여백 추가 (sub-pixel refinement 가 코너 주변 픽셀 봐야 하므로 필수)
    canvas_size = marker_px + 2 * padding_px
    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 255
    canvas[padding_px:padding_px+marker_px, padding_px:padding_px+marker_px] = marker_img

    # 3) ID + 사이즈 텍스트 추가 (인쇄 후 식별 용)
    text = f"DICT_6X6_50  ID={marker_id}  size={marker_mm:.0f}mm"
    cv2.putText(canvas, text, (padding_px, canvas_size - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)

    cv2.imwrite(out_path, canvas)
    print(f"✓ 생성됨: {out_path}")
    print(f"  마커 본체: {marker_px} x {marker_px} px = {marker_mm} x {marker_mm} mm")
    print(f"  전체 캔버스: {canvas_size} x {canvas_size} px")
    print(f"  인쇄 DPI: {DPI_PRINT}")
    print()
    print("인쇄 가이드:")
    print("  1. 이 PNG 를 인쇄 (Acrobat/뷰어에서 100% / Actual size 옵션)")
    print("  2. Fit-to-page 옵션 끄기 (반드시!)")
    print("  3. 인쇄 후 자로 마커 본체 길이 측정 → 정확히 {:.0f}mm 인지 확인".format(marker_mm))
    print("  4. 딱딱한 평판 (아크릴 5mm, PVC, 두꺼운 카드보드) 에 부착")
    print("  5. 보호 코팅 (laminate) 시 무광 권장 (광택은 반사로 검출 노이즈 ↑)")


def main():
    parser = argparse.ArgumentParser(description="ArUco 마커 PNG 생성")
    parser.add_argument('--id', type=int, default=0, help='마커 ID (기본 0)')
    parser.add_argument('--size', type=float, default=50.0,
                        help='마커 본체 크기 mm (기본 50)')
    parser.add_argument('--out', type=str, default=None,
                        help='출력 PNG 경로 (기본: aruco_DICT_6X6_50_ID{ID}_{SIZE}mm.png)')
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        d = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(d, f"aruco_DICT_6X6_50_ID{args.id}_{args.size:.0f}mm.png")
    generate(args.id, args.size, out_path)


if __name__ == "__main__":
    main()
