import time
import os
import cv2
import numpy as np
import subprocess 
from ultralytics import YOLO

# Picamera2 로드
try:
    from picamera2 import Picamera2
except ImportError:
    print("picamera2 라이브러리가 필요합니다.")
    exit()

# ==========================================
# [1] 설정: 상수 및 파라미터
# ==========================================
CONFIG_FILE = "camera_config.npy"
MODEL_PATH = "yolov8n-pose.pt"

# 캘리브레이션 실제 거리 (1m~4m)
REAL_POINTS_BASE = np.array([
    [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]
], dtype=np.float32)

# 현장 맞춤형 계수
ALPHA = 545.62 
BETA = 3.15

# 2차 보정 계수
CORRECT_A, CORRECT_B, CORRECT_C = 0, 1, 0 
REALITY_SCALE = 1.0 

# ==========================================
# [모듈 0] 초기화
# ==========================================
def check_calibration():
    if not os.path.exists(CONFIG_FILE):
        print("설정 파일이 없습니다. 캘리브레이션 모드를 실행합니다.")
        subprocess.run(["python", "calibration.py"])
        if not os.path.exists(CONFIG_FILE):
            print("캘리브레이션 실패. 종료합니다.")
            exit()
    else:
        print("설정 파일 로드됨.")

def init_camera():
    print("메인 시스템 카메라 초기화 (Wide View)...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (1640, 1232), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    return picam2

# ==========================================
# [모듈 1] 계산 로직
# ==========================================
def compute_homography(pixel_points):
    pixels = pixel_points.tolist()
    reals = REAL_POINTS_BASE.tolist()
    p1 = pixels[0]
    pixels.append([p1[0] + 100.0, p1[1]]) 
    reals.append([0.5, 1.0]) 
    src = np.array(pixels, dtype=np.float32)
    dst = np.array(reals, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H

def get_features(keypoints, box_h):
    kps = keypoints.data[0].cpu().numpy()
    # 9개 점 모델 기준 인덱스
    l_sh, r_sh = kps[1], kps[2]
    l_hip, r_hip = kps[3], kps[4]
    l_knee, r_knee = kps[5], kps[6]
    l_ankle, r_ankle = kps[7], kps[8]

    foot_pt = None
    if l_ankle[2] > 0.5 or r_ankle[2] > 0.5:
        pts = [p[:2] for p in [l_ankle, r_ankle] if p[2] > 0.5]
        foot_pt = np.mean(pts, axis=0)
    elif l_knee[2] > 0.5 or r_knee[2] > 0.5:
        pts = [p[:2] for p in [l_knee, r_knee] if p[2] > 0.5]
        avg = np.mean(pts, axis=0)
        foot_pt = [avg[0], avg[1] + (box_h * 0.25)]

    torso_len = None
    if l_sh[2] > 0.5 or r_sh[2] > 0.5 or l_hip[2] > 0.5 or r_hip[2] > 0.5:
        sh_ys = [p[1] for p in [l_sh, r_sh] if p[2] > 0.5]
        hip_ys = [p[1] for p in [l_hip, r_hip] if p[2] > 0.5]
        if sh_ys and hip_ys:
            torso_len = abs(np.mean(hip_ys) - np.mean(sh_ys))

    return foot_pt, torso_len

def apply_correction(d):
    return max(0, (CORRECT_A * d**2) + (CORRECT_B * d) + CORRECT_C)

def calculate_ensemble_distance(foot_pt, torso_len, img_h, H):
    dist_stat = 0
    if torso_len and torso_len > 0:
        raw_stat = (ALPHA / torso_len) + BETA
        dist_stat = apply_correction(raw_stat)

    dist_homo = 0
    is_clipped = True
    real_x = 0
    
    if foot_pt is not None:
        if foot_pt[1] < (img_h * 0.95):
            is_clipped = False
            pt_px = np.array([[[foot_pt[0], foot_pt[1]]]], dtype=np.float32)
            pt_real = cv2.perspectiveTransform(pt_px, H)
            dist_homo = apply_correction(pt_real[0][0][1])
            real_x = pt_real[0][0][0]
        else:
            real_x = (foot_pt[0] - 320) * dist_stat * 0.002

    final_dist = 0
    method = ""

    if is_clipped:
        final_dist = dist_stat
        method = "Stat"
    else:
        if dist_homo > 0 and dist_stat > 0:
            final_dist = (dist_homo + dist_stat) / 2
            method = "Mix"
        elif dist_homo > 0:
            final_dist = dist_homo
            method = "Homo"
        else:
            final_dist = dist_stat
            method = "Stat"

    return real_x, final_dist * REALITY_SCALE, method

# ==========================================
# [모듈 2] 상태 정보 및 시각화
# ==========================================
def get_status_info(dist):
    """ 거리별 위험 상태 및 색상 반환 """
    if dist < 1.5:
        return "DANGER", (0, 0, 255)   # Red
    elif dist < 2.5:
        return "WARNING", (0, 165, 255) # Orange
    else:
        return "Safe", (0, 255, 0)      # Green

def draw_bounding_box(frame, box, dist, status, color, method):
    """ 계산된 값으로 박스와 텍스트 그리기 """
    x1, y1, x2, y2 = box
    
    # 1. 박스 그리기
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # 2. 정보 표시
    label = f"{status} {dist:.1f}m ({method})"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(frame, label, (x1, y1 - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def draw_radar(objects, width=400, height=400, current_alert="Safe"):
    radar = np.zeros((height, width, 3), dtype=np.uint8)
    scale_z = height / 5.0
    cx = width // 2
    
    # 격자
    for i in range(1, 6):
        y = height - int(i * scale_z)
        col = (50, 50, 50)
        if i == 2: col = (0, 0, 150)
        cv2.line(radar, (0, y), (width, y), col, 1)
        cv2.putText(radar, f"{i}m", (10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150))
    
    cv2.circle(radar, (cx, height), 15, (255, 255, 255), -1) # 지게차
    
    # 객체 표시
    for (x, z, status) in objects:
        px = np.clip(int(cx + (x * (width / 4.0))), 0, width)
        py = np.clip(int(height - (z * scale_z)), 0, height)
        
        _, color = get_status_info(z)
        cv2.circle(radar, (px, py), 10, color, -1)

    # 경고 테두리
    if current_alert == "DANGER":
        cv2.rectangle(radar, (0,0), (width,height), (0,0,255), 10)
        cv2.putText(radar, "STOP!", (cx-40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    elif current_alert == "WARNING":
        cv2.rectangle(radar, (0,0), (width,height), (0,165,255), 5)
    
    return radar

# ==========================================
# [모듈 3] 메인 실행 루프
# ==========================================
def run_system(picam2, H):
    print("\n시스템 가동! (종료: q, 리셋: r)")
    model = YOLO(MODEL_PATH) 

    while True:
        frame = picam2.capture_array()
        h, w = frame.shape[:2]

        results = model(frame, verbose=False, conf=0.5)
        detected_objects = []
        max_alert = "Safe"

        # [Console Output] 화면 갱신 전 로그 출력용
        # print("-" * 30) 

        for result in results:
            if result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for i, box in enumerate(boxes):
                    # 1. 바운딩 박스 좌표 추출 (정수 변환)
                    x1, y1, x2, y2 = map(int, box)
                    box_h = y2 - y1
                    
                    # 2. 특징 추출
                    foot_pt, torso_len = get_features(result.keypoints[i], box_h)
                    
                    if foot_pt is not None:
                        # 3. 거리 계산
                        real_x, dist, method = calculate_ensemble_distance(foot_pt, torso_len, h, H)
                        
                        # 4. 상태 판단
                        status, color = get_status_info(dist)
                        
                        # 5. [핵심] 콘솔에 값 출력 (Output Values)
                        print(f"Object #{i}: Dist={dist:.2f}m | Status={status} | Box=[{x1}, {y1}, {x2}, {y2}] | Mode={method}")

                        # 6. 위험도 집계 (레이더용)
                        if status == "DANGER": max_alert = "DANGER"
                        elif status == "WARNING" and max_alert != "DANGER": max_alert = "WARNING"
                        
                        # 7. 데이터 저장
                        detected_objects.append((real_x, dist, status))

                        # 8. 메인 화면 시각화 (출력된 값 기반으로 그림)
                        draw_bounding_box(frame, (x1, y1, x2, y2), dist, status, color, method)
                        
                        # 발 위치 점 (옵션)
                        if 0 <= foot_pt[0] < w and 0 <= foot_pt[1] < h:
                            cv2.circle(frame, (int(foot_pt[0]), int(foot_pt[1])), 5, (0, 255, 255), -1)

        # 9. 레이더 시각화
        radar_img = draw_radar(detected_objects, current_alert=max_alert)

        cv2.imshow('Main Camera', frame)
        cv2.imshow('Radar View', radar_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): return "EXIT"
        elif key == ord('r'):
            print("설정 초기화...")
            picam2.stop()
            cv2.destroyAllWindows()
            os.remove(CONFIG_FILE)
            check_calibration()
            pixel_points = np.load(CONFIG_FILE)
            H = compute_homography(pixel_points)
            picam2 = init_camera()

# ==========================================
# 메인 진입점
# ==========================================
def main():
    check_calibration()
    pixel_points = np.load(CONFIG_FILE)
    H = compute_homography(pixel_points)
    picam2 = init_camera()
    
    try:
        status = run_system(picam2, H)
        if status == "EXIT": pass
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        print("프로그램 종료")

if __name__ == "__main__":
    main()
