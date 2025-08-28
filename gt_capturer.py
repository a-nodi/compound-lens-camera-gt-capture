#!/usr/bin/env python3
import os
import cv2
import time
import numpy as np
import subprocess
import depthai as dai
from datetime import datetime

MAX_MM = 5000
FPS = 30
OUT_DIR = "captures"
RPICAM_ENABLE = True
RPICAM_JPG_QUALITY = 95

os.makedirs(OUT_DIR, exist_ok=True)

def nowtag():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def colorize_depth_mm(depth_mm, max_mm=5000):
    vis = np.clip(depth_mm, 0, max_mm).astype(np.float32)
    vis = (vis / max_mm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(vis, cv2.COLORMAP_JET)

def save_oak_frames(rgb_bgr, depth_mm, tag):
    rgb_path = os.path.join(OUT_DIR, f"{tag}_oak_rgb.png")
    depth_raw_path = os.path.join(OUT_DIR, f"{tag}_oak_depth_mm.png")
    depth_vis_path = os.path.join(OUT_DIR, f"{tag}_oak_depth_vis.png")

    cv2.imwrite(rgb_path, rgb_bgr)
    cv2.imwrite(depth_raw_path, depth_mm)
    cv2.imwrite(depth_vis_path, colorize_depth_mm(depth_mm, MAX_MM))
    return rgb_path, depth_raw_path, depth_vis_path

def capture_rpicam_raw(tag):
    jpg_path = os.path.join(OUT_DIR, f"{tag}_rpicam.jpg")
    cmd = ["rpicam-still", "--raw", "-n", "-o", jpg_path, "-q", str(RPICAM_JPG_QUALITY)]
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        dng_path = jpg_path.replace(".jpg", ".dng")
        return jpg_path, (dng_path if os.path.exists(dng_path) else None), None
    except subprocess.CalledProcessError as e:
        return None, None, e.stderr.decode("utf-8", errors="ignore")

def build_pipeline():
    pipeline = dai.Pipeline()

    camRgb = pipeline.createColorCamera()
    camRgb.setBoardSocket(dai.CameraBoardSocket.RGB)
    camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_720_P)
    camRgb.setFps(FPS)
    camRgb.setInterleaved(False)

    monoL = pipeline.createMonoCamera()
    monoR = pipeline.createMonoCamera()
    monoL.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoR.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    monoL.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)
    monoR.setResolution(dai.MonoCameraProperties.SensorResolution.THE_720_P)

    stereo = pipeline.createStereoDepth()
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)

    monoL.out.link(stereo.left)
    monoR.out.link(stereo.right)

    xoutRgb = pipeline.createXLinkOut()
    xoutRgb.setStreamName("rgb")
    camRgb.video.link(xoutRgb.input)

    xoutDepth = pipeline.createXLinkOut()
    xoutDepth.setStreamName("depth")
    stereo.depth.link(xoutDepth.input)

    return pipeline

def main():
    pipeline = build_pipeline()

    with dai.Device(pipeline) as device:
        qRgb = device.getOutputQueue("rgb", maxSize=4, blocking=False)
        qDepth = device.getOutputQueue("depth", maxSize=4, blocking=False)

        print("[INFO] Press SPACE to capture, 'q' to quit")
        while True:
            inRgb = qRgb.get()
            inDepth = qDepth.get()

            rgb = inRgb.getCvFrame()
            depth = inDepth.getFrame()

            h, w = rgb.shape[:2]
            if depth.shape[:2] != (h, w):
                depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)

            depth_vis = colorize_depth_mm(depth, MAX_MM)
            overlay = cv2.addWeighted(rgb, 0.55, depth_vis, 0.45, 0.0)

            cv2.imshow("RGB (720p)", rgb)
            cv2.imshow("Depth aligned to RGB (720p)", depth_vis)
            cv2.imshow("Overlay", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32:
                tag = nowtag()
                print(f"[CAPTURE] {tag} saving...")
                rgb_path, depth_raw_path, depth_vis_path = save_oak_frames(rgb, depth, tag)
                print(f"  - OAK RGB:       {rgb_path}")
                print(f"  - OAK Depth RAW: {depth_raw_path}")
                print(f"  - OAK Depth VIS: {depth_vis_path}")
                if RPICAM_ENABLE:
                    jpg_path, dng_path, err = capture_rpicam_raw(tag)
                    if err is None:
                        print(f"  - RPICAM JPEG:   {jpg_path}")
                        print(f"  - RPICAM DNG:    {dng_path if dng_path else '(not found)'}")
                    else:
                        print("  - RPICAM ERROR:\n", err)
                fb = overlay.copy()
                cv2.putText(fb, f"Captured: {tag}", (10, 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2, cv2.LINE_AA)
                cv2.imshow("Overlay", fb)
                cv2.waitKey(250)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
