import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

# ★ 작동 확인된 Picamera2 라이브러리 로드
try:
    from picamera2 import Picamera2
except ImportError:
    print("❌ 'picamera2' 라이브러리가 없습니다. 설치해주세요.")
    exit()

# ==========================================
# [1] 설정: 상수 및 파일 경로
# ==========================================
CONFIG_FILE = "camera_config.npy"
MODEL_PATH = "yolov8n-pose.pt" # .pt 파일 그대로 사용

# 캘리브레이션 할 실제 거리 (1m, 2m, 3m, 4m)
REAL_POINTS_BASE = np.array([
    [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]
], dtype=np.float32)

# 거리 공식 계수 (YOLO 맞춤형)
ALPHA = 1357.44
BETA = 4.29
REALITY_SCALE = 1.0

# ==========================================
# [모듈 0] 카메라 초기화 (사용자 코드 반영)
# ==========================================
def init_camera():
    print("📷 PiCamera2 초기화 중...")
    picam2 = Picamera2()
    
    # ★ 사용자님이 주신 코드 방식 적용 (해상도는 640x480 권장, 포맷은 BGR)
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    
    print("✅ 카메라 시작됨! (Warmup 2초)")
    time.sleep(2) # 카메라 안정화 대기
    return picam2

# ==========================================
# [모듈 1] 캘리브레이션
# ==========================================
def run_calibration(picam2):
    print("\n=== [캘리브레이션 모드] ===")
    print("바닥의 1m, 2m, 3m, 4m 지점을 순서대로 클릭하세요.")
    
    clicked_points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < 4:
                print(f"📍 포인트 {len(clicked_points)+1} 입력: [{x}, {y}]")
                clicked_points.append([x, y])

    window_name = "Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        # ★ Picamera2에서 프레임 가져오기 (작동 확인된 방식)
        frame = picam2.capture_array()

        cv2.putText(frame, f"Points: {len(clicked_points)}/4", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        for i, pt in enumerate(clicked_points):
            cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{i+1}m", (pt[0]+10, pt[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        
        if len(clicked_points) == 4:
            print("설정 완료! 저장 중...")
            cv2.waitKey(1000)
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("취소됨")
            picam2.stop()
            cv2.destroyAllWindows()
            exit()

    cv2.destroyWindow(window_name)
    
    pts_array = np.array(clicked_points, dtype=np.float32)
    np.save(CONFIG_FILE, pts_array)
    return pts_array

# ==========================================
# [모듈 2] 호모그래피 행렬 계산
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

# ==========================================
# [모듈 3] 거리 계산 및 레이더
# ==========================================
def get_foot_point(keypoints, box_h):
    kps = keypoints.data[0].cpu().numpy()
    l_ankle, r_ankle = kps[15], kps[16]
    l_knee, r_knee = kps[13], kps[14]

    if l_ankle[2] > 0.5 or r_ankle[2] > 0.5:
        pts = [p[:2] for p in [l_ankle, r_ankle] if p[2] > 0.5]
        avg = np.mean(pts, axis=0)
        return avg[0], avg[1], "Real"
    elif l_knee[2] > 0.5 or r_knee[2] > 0.5:
        pts = [p[:2] for p in [l_knee, r_knee] if p[2] > 0.5]
        avg = np.mean(pts, axis=0)
        return avg[0], avg[1] + (box_h * 0.25), "Virtual"
    return None, None, None

def calculate_distance_hybrid(u, v, box_h, img_h, H):
    is_clipped = v >= (img_h * 0.95)
    if is_clipped:
        dist = (ALPHA / box_h) + BETA
        method = "Stat"
        real_x = (u - (img_h * 1.33 / 2)) * dist * 0.002
    else:
        pt = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H)
        real_x = pt[0][0][0]
        dist = pt[0][0][1]
        method = "Homo"
    return real_x, dist * REALITY_SCALE, method

def draw_separate_radar(objects, width=400, height=400, current_alert="Safe"):
    radar = np.zeros((height, width, 3), dtype=np.uint8)
    scale_z = height / 5.0
    cx = width // 2
    
    for i in range(1, 6):
        y = height - int(i * scale_z)
        col = (50, 50, 50)
        if i == 2: col = (0, 0, 150)
        cv2.line(radar, (0, y), (width, y), col, 1)
        cv2.putText(radar, f"{i}m", (10, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150,150,150))

    cv2.circle(radar, (cx, height), 15, (255, 255, 255), -1)
    
    for (x, z, status) in objects:
        px = int(cx + (x * (width / 4.0)))
        py = int(height - (z * scale_z))
        px = np.clip(px, 0, width)
        py = np.clip(py, 0, height)
        
        color = (0, 255, 0)
        if "DANGER" in status: color = (0, 0, 255)
        elif "WARNING" in status: color = (0, 165, 255)
        cv2.circle(radar, (px, py), 10, color, -1)

    if "DANGER" in current_alert:
        cv2.rectangle(radar, (0,0), (width,height), (0,0,255), 10)
        cv2.putText(radar, "STOP!", (cx-40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
    elif "WARNING" in current_alert:
        cv2.rectangle(radar, (0,0), (width,height), (0,165,255), 5)

    return radar

# ==========================================
# [모듈 4] 메인 시스템
# ==========================================
def run_system(picam2, H):
    print("\n🚀 시스템 가동! (종료: q, 리셋: r)")
    model = YOLO(MODEL_PATH) 

    while True:
        # ★ Picamera2 프레임 캡처
        frame = picam2.capture_array()
        h, w = frame.shape[:2]

        results = model(frame, verbose=False, conf=0.5)
        detected_objects = []
        max_alert = "Safe"

        for result in results:
            if result.keypoints is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box)
                    box_h = y2 - y1
                    
                    # 사람 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

                    foot_u, foot_v, pose_type = get_foot_point(result.keypoints[i], box_h)
                    
                    if foot_u is not None:
                        real_x, dist, method = calculate_distance_hybrid(foot_u, foot_v, box_h, h, H)
                        
                        if dist < 1.5:
                            status = "DANGER"
                            color = (0, 0, 255)
                            max_alert = status
                        elif dist < 2.5:
                            status = "WARNING"
                            color = (0, 165, 255)
                            if max_alert != "DANGER": max_alert = status
                        else:
                            status = "Safe"
                            color = (0, 255, 0)
                        
                        detected_objects.append((real_x, dist, status))

                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{status} {dist:.2f}m", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        if 0 <= foot_u < w and 0 <= foot_v < h:
                            cv2.circle(frame, (int(foot_u), int(foot_v)), 5, (0, 255, 255), -1)

        radar_img = draw_separate_radar(detected_objects, current_alert=max_alert)

        cv2.imshow('Main Camera', frame)
        cv2.imshow('Radar View', radar_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return "EXIT"
        elif key == ord('r'):
            if os.path.exists(CONFIG_FILE):
                os.remove(CONFIG_FILE)
            return "RESET"

# ==========================================
# 메인 진입점
# ==========================================
def main():
    picam2 = init_camera() # 카메라 켜기

    try:
        while True:
            if os.path.exists(CONFIG_FILE):
                try:
                    pixel_points = np.load(CONFIG_FILE)
                    print("설정 파일 로드됨.")
                except:
                    pixel_points = run_calibration(picam2)
            else:
                pixel_points = run_calibration(picam2)
            
            H = compute_homography(pixel_points)
            
            status = run_system(picam2, H)
            
            if status == "EXIT":
                break
    finally:
        # 프로그램 종료 시 카메라 안전하게 끄기
        picam2.stop()
        cv2.destroyAllWindows()
        print("✅ 프로그램 및 카메라 종료 완료")

if __name__ == "__main__":
    main()
