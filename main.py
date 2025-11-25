import cv2
import numpy as np
import os
import time # 시간 지연을 위해 추가
from ultralytics import YOLO

# ==========================================
# [설정] 상수 및 파일 경로
# ==========================================
CONFIG_FILE = "camera_config.npy"
MODEL_PATH = "yolov8n.tflite" # 라즈베리 파이용 모델 (없으면 .pt)

# 캘리브레이션 할 실제 거리 (1m, 2m, 3m, 4m)
REAL_POINTS_BASE = np.array([
    [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]
], dtype=np.float32)

# ==========================================
# [모듈 1] 캘리브레이션 (안전장치 추가)
# ==========================================
def run_calibration(cap):
    print("\n=== [캘리브레이션 모드] ===")
    print("바닥의 1m, 2m, 3m, 4m 지점을 순서대로 클릭하세요.")
    
    clicked_points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < 4:
                print(f"포인트 {len(clicked_points)+1} 입력: [{x}, {y}]")
                clicked_points.append([x, y])

    window_name = "Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 영상을 읽을 수 없습니다! (연결 확인 필요)")
            break

        cv2.putText(frame, f"Points: {len(clicked_points)}/4", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        for i, pt in enumerate(clicked_points):
            cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
            cv2.putText(frame, f"{i+1}m", (pt[0]+10, pt[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(window_name, frame)
        
        # 4개 다 찍으면 자동 저장
        if len(clicked_points) == 4:
            print("설정 완료! 저장 중...")
            cv2.waitKey(1000)
            break
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("취소됨")
            exit()

    cv2.destroyWindow(window_name)
    
    # [에러 방지] 점을 안 찍고 껐을 경우 처리
    if len(clicked_points) < 4:
        print("경고: 점 4개를 다 찍지 않았습니다. 프로그램을 종료합니다.")
        exit()

    pts_array = np.array(clicked_points, dtype=np.float32)
    np.save(CONFIG_FILE, pts_array)
    return pts_array

# ==========================================
# [모듈 2] 호모그래피 행렬 계산
# ==========================================
def compute_homography(pixel_points):
    if len(pixel_points) < 4:
        return None # 데이터 부족 시 None 반환

    pixels = pixel_points.tolist()
    reals = REAL_POINTS_BASE.tolist()
    
    # 가상 포인트 추가 (1m 지점 기준)
    p1 = pixels[0]
    pixels.append([p1[0] + 100.0, p1[1]]) 
    reals.append([0.5, 1.0]) 

    src = np.array(pixels, dtype=np.float32)
    dst = np.array(reals, dtype=np.float32)
    
    H, _ = cv2.findHomography(src, dst)
    return H

# ==========================================
# [모듈 3] 레이더 그리기
# ==========================================
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
# [모듈 4] 메인 실행
# ==========================================
def run_system(cap, H):
    print("\n시스템 가동! (초기화: r / 종료: q)")
    try:
        model = YOLO(MODEL_PATH)
    except:
        print(f"모델 파일({MODEL_PATH})을 찾을 수 없습니다. yolov8n.pt로 대체합니다.")
        model = YOLO("yolov8n.pt")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("카메라 끊김!")
            break
        
        results = model.predict(frame, classes=[0], verbose=False)
        detected_objects = []
        max_alert = "Safe"

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                foot_u, foot_v = (x1 + x2) / 2, y2
                
                point_pixel = np.array([[[foot_u, foot_v]]], dtype=np.float32)
                point_real = cv2.perspectiveTransform(point_pixel, H)
                
                real_x = point_real[0][0][0]
                real_z = point_real[0][0][1]
                
                if real_z < 1.5:
                    status = "DANGER"
                    color = (0, 0, 255)
                    max_alert = status
                elif real_z < 2.5:
                    status = "WARNING"
                    color = (0, 165, 255)
                    if max_alert != "DANGER": max_alert = status
                else:
                    status = "Safe"
                    color = (0, 255, 0)
                
                detected_objects.append((real_x, real_z, status))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{status} {real_z:.2f}m", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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
    print("📷 카메라 초기화 중...")
    # [핵심] 라즈베리 파이 호환성 코드 (V4L2 백엔드 사용)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    
    # 해상도 안전하게 설정 (640x480)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 카메라가 켜질 때까지 잠시 대기
    time.sleep(2)

    if not cap.isOpened():
        print("카메라를 열 수 없습니다!")
        print("   1. USB 카메라가 연결되어 있는지 확인하세요.")
        print("   2. 라즈베리 파이 설정(raspi-config)에서 카메라가 켜져 있는지 확인하세요.")
        return

    print("카메라 연결 성공!")

    while True:
        if os.path.exists(CONFIG_FILE):
            try:
                pixel_points = np.load(CONFIG_FILE)
                print("설정 파일 로드됨.")
            except:
                print("설정 파일 깨짐. 다시 캘리브레이션 합니다.")
                pixel_points = run_calibration(cap)
        else:
            pixel_points = run_calibration(cap)
        
        H = compute_homography(pixel_points)
        
        status = run_system(cap, H)
        
        if status == "EXIT":
            break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
