#!/usr/bin/env python3
import os
import cv2
import time
import subprocess
import numpy as np
import depthai as dai
from datetime import datetime

# ===== 설정 =====
RGB_SIZE = (640, 480)       # 미리보기 해상도
MAX_MM = 2000               # depth 시각화 범위 (0~2m)
OUT_DIR = "captures"        # 저장 폴더
RPICAM_JPG_QUALITY = 95     # rpicam-still JPEG 품질(선택)

os.makedirs(OUT_DIR, exist_ok=True)

def nowtag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def colorize_depth_mm(depth_mm, max_mm=5000):
    """uint16(mm) -> 컬러맵 BGR uint8"""
    vis = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    vis = (vis / max_mm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

def save_oak_frames(rgb_bgr, depth_mm, tag):
    """
    rgb_bgr: uint8 (H,W,3) BGR
    depth_mm: uint16 (H,W) in millimeters
    tag: timestamp tag
    """
    # 파일 경로
    rgb_path = os.path.join(OUT_DIR, f"{tag}_oak_rgb.png")
    depth_raw_path = os.path.join(OUT_DIR, f"{tag}_oak_depth_mm.png")          # 16-bit PNG
    depth_vis_path = os.path.join(OUT_DIR, f"{tag}_oak_depth_vis.png")         # colorized

    # 저장
    cv2.imwrite(rgb_path, rgb_bgr)
    cv2.imwrite(depth_raw_path, depth_mm)  # dtype=uint16로 16-bit PNG 저장
    cv2.imwrite(depth_vis_path, colorize_depth_mm(depth_mm, MAX_MM))

    return rgb_path, depth_raw_path, depth_vis_path

def capture_rpicam_raw(tag):
    """
    rpicam-still --raw로 촬영.
    -n: preview 없음(즉시 촬영)
    -o: JPEG 경로 지정 -> 같은 이름의 .dng 자동 저장
    """
    jpg_path = os.path.join(OUT_DIR, f"{tag}_rpicam.jpg")
    # --raw: JPEG와 같은 이름의 DNG를 함께 저장 (예: *_rpicam.dng)
    cmd = [
        "rpicam-still",
        "--raw",
        "-n",
        "-o", jpg_path,
        "-q", str(RPICAM_JPG_QUALITY)  # 선택
    ]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        # DNG 경로 추정
        dng_path = jpg_path.replace(".jpg", ".dng")
        return jpg_path, (dng_path if os.path.exists(dng_path) else None), None
    except subprocess.CalledProcessError as e:
        return None, None, e.stderr.decode("utf-8", errors="ignore")

def build_pipeline():
    pipeline = dai.Pipeline()

    # RGB 카메라(중앙)
    camRgb = pipeline.createColorCamera()
    camRgb.setPreviewSize(RGB_SIZE[0], RGB_SIZE[1])
    camRgb.setInterleaved(False)
    camRgb.setFps(30)

    # 좌/우 모노
    monoL = pipeline.createMonoCamera()
    monoR = pipeline.createMonoCamera()
    monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    # 스테레오
    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)

    # 핵심: depth를 중앙 RGB 시점으로 정렬
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    # 링크
    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.preview.link(xoutRgb.input)

    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)

    return pipeline

def main():
    pipeline = build_pipeline()

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        win_rgb = "OAK-D RGB"
        win_depth = "OAK-D Depth (aligned to RGB)"
        win_overlay = "Overlay (RGB + Depth)"

        print("[INFO] 창에서 스페이스바로 캡처, 'q'로 종료")
        while True:
            inRgb = qRgb.get()
            inDepth = qDepth.get()

            rgb = inRgb.getCvFrame()        # BGR uint8
            depth = inDepth.getFrame()      # uint16 (mm), RGB 시점에 align됨

            # 시각화
            depth_vis = colorize_depth_mm(depth, MAX_MM)

            # 오버레이(보기 편의)
            overlay = cv2.addWeighted(rgb, 0.55, depth_vis, 0.45, 0.0)

            # 표시
            cv2.imshow(win_rgb, rgb)
            cv2.imshow(win_depth, depth_vis)
            cv2.imshow(win_overlay, overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32:  # Space
                tag = nowtag()
                print(f"[CAPTURE] {tag} - 저장 중...")

                # 1) OAK 프레임 저장
                rgb_path, depth_raw_path, depth_vis_path = save_oak_frames(rgb, depth, tag)
                print(f"  - OAK RGB:        {rgb_path}")
                print(f"  - OAK Depth RAW:  {depth_raw_path}")
                print(f"  - OAK Depth VIS:  {depth_vis_path}")

                # 2) rpicam-still --raw 촬영 (JPEG + DNG)
                jpg_path, dng_path, err = capture_rpicam_raw(tag)
                if err is None:
                    print(f"  - RPICAM JPEG:    {jpg_path}")
                    if dng_path:
                        print(f"  - RPICAM DNG:     {dng_path}")
                    else:
                        print("  - RPICAM DNG:     (경로 확인 필요)")
                else:
                    print("  - RPICAM ERROR:")
                    print(err)

                # 화면에 잠깐 피드백
                fb = overlay.copy()
                cv2.putText(fb, f"Captured: {tag}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow(win_overlay, fb)
                cv2.waitKey(300)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
