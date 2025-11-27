import time
import os
import cv2
import numpy as np
from ultralytics import YOLO

# 작동 확인된 Picamera2 라이브러리 로드import time
import os
import cv2
import numpy as np
import subprocess # 외부 파일 실행용
from ultralytics import YOLO

# ★ Picamera2 로드
try:
    from picamera2 import Picamera2
except ImportError:
    print("picamera2 라이브러리가 필요합니다.")
    exit()

# ==========================================
# [1] 설정
# ==========================================
CONFIG_FILE = "camera_config.npy"
MODEL_PATH = "yolov8n-pose.pt"

# 캘리브레이션 실제 거리 (1m~4m)
REAL_POINTS_BASE = np.array([
    [0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]
], dtype=np.float32)

# 현장 맞춤형 계수 (parameter_update.py로 구한 값을 여기에 넣으세요!)
ALPHA = 830.93
BETA = 1.09

# 2차 보정 계수 (필요시 사용, 지금은 1:1)
CORRECT_A, CORRECT_B, CORRECT_C = 0, 1, 0 

# ==========================================
# [모듈 0] 초기화 및 캘리브레이션 체크
# ==========================================
def check_calibration():
    if not os.path.exists(CONFIG_FILE):
        print("설정 파일이 없습니다. 캘리브레이션 모드를 실행합니다.")
        # calibration.py 실행 (현재 프로세스 대기)
        subprocess.run(["python", "calibration.py"])
        
        if not os.path.exists(CONFIG_FILE):
            print("캘리브레이션이 완료되지 않았습니다. 종료합니다.")
            exit()
    else:
        print("설정 파일 로드됨.")

def init_camera():
    print("메인 시스템 카메라 초기화...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    return picam2

# ==========================================
# [모듈 2] 계산 함수들 (앙상블 로직 포함)
# ==========================================
def compute_homography(pixel_points):
    pixels = pixel_points.tolist()
    reals = REAL_POINTS_BASE.tolist()
    # 가상 포인트 추가
    p1 = pixels[0]
    pixels.append([p1[0] + 100.0, p1[1]]) 
    reals.append([0.5, 1.0]) 
    
    src = np.array(pixels, dtype=np.float32)
    dst = np.array(reals, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst)
    return H

def get_features(keypoints, box_h):
    """ 발 위치(좌표)와 상반신 길이(길이) 추출 """
    kps = keypoints.data[0].cpu().numpy()
    l_ankle, r_ankle = kps[15], kps[16]
    l_knee, r_knee = kps[13], kps[14]
    l_sh, r_sh = kps[5], kps[6]
    l_hip, r_hip = kps[11], kps[12]

    # 1. 발 위치 (좌표)
    foot_pt = None
    pose_type = "Unknown"
    if l_ankle[2] > 0.5 or r_ankle[2] > 0.5:
        pts = [p[:2] for p in [l_ankle, r_ankle] if p[2] > 0.5]
        foot_pt = np.mean(pts, axis=0)
        pose_type = "Real"
    elif l_knee[2] > 0.5 or r_knee[2] > 0.5:
        pts = [p[:2] for p in [l_knee, r_knee] if p[2] > 0.5]
        avg = np.mean(pts, axis=0)
        # 무릎 아래 가상 발
        foot_pt = [avg[0], avg[1] + (box_h * 0.25)]
        pose_type = "Virtual"

    # 2. 상반신 길이 (길이)
    torso_len = None
    if l_sh[2] > 0.5 or r_sh[2] > 0.5 or l_hip[2] > 0.5 or r_hip[2] > 0.5:
        # 보이는 점들의 평균으로 계산
        sh_ys = [p[1] for p in [l_sh, r_sh] if p[2] > 0.5]
        hip_ys = [p[1] for p in [l_hip, r_hip] if p[2] > 0.5]
        if sh_ys and hip_ys:
            torso_len = abs(np.mean(hip_ys) - np.mean(sh_ys))

    return foot_pt, torso_len, pose_type

def apply_correction(d):
    return max(0, (CORRECT_A * d**2) + (CORRECT_B * d) + CORRECT_C)

def calculate_ensemble_distance(foot_pt, torso_len, img_h, H):
    """ 호모그래피 + 통계 앙상블 계산 """
    
    # 1. Stat (통계) 방식 계산
    dist_stat = 0
    if torso_len and torso_len > 0:
        raw_stat = (ALPHA / torso_len) + BETA
        dist_stat = apply_correction(raw_stat)

    # 2. Homo (호모그래피) 방식 계산
    dist_homo = 0
    is_clipped = True
    
    if foot_pt is not None:
        # 발이 화면 하단 5% 이내면 잘림으로 간주
        if foot_pt[1] < (img_h * 0.95):
            is_clipped = False
            pt_px = np.array([[[foot_pt[0], foot_pt[1]]]], dtype=np.float32)
            pt_real = cv2.perspectiveTransform(pt_px, H)
            dist_homo = apply_correction(pt_real[0][0][1])
            
            # X 좌표 (레이더용)
            real_x = pt_real[0][0][0]
        else:
            # 잘렸을 땐 X좌표 대략 추정
            real_x = (foot_pt[0] - (640/2)) * dist_stat * 0.002

    # 3. 최종 결정 (Ensemble)
    final_dist = 0
    method = ""

    if is_clipped:
        # 발 잘림 -> 무조건 통계 공식
        final_dist = dist_stat
        method = "Stat"
    else:
        # 발 보임 -> 평균값 (Homo + Stat) / 2
        if dist_homo > 0 and dist_stat > 0:
            final_dist = (dist_homo + dist_stat) / 2
            method = "Mix"
        elif dist_homo > 0:
            final_dist = dist_homo
            method = "Homo"
        else:
            final_dist = dist_stat
            method = "Stat"

    return real_x, final_dist, method

# ... (draw_separate_radar 함수는 이전과 동일하므로 생략해도 되지만, 편의상 포함) ...
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
        px = np.clip(int(cx + (x * (width / 4.0))), 0, width)
        py = np.clip(int(height - (z * scale_z)), 0, height)
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
# [모듈 4] 메인 루프
# ==========================================
def main():
    # 1. 캘리브레이션 확인
    check_calibration()
    
    # 2. 설정 로드
    pixel_points = np.load(CONFIG_FILE)
    H = compute_homography(pixel_points)
    
    # 3. 카메라 및 모델 로드
    picam2 = init_camera()
    model = YOLO(MODEL_PATH)
    print("\n시스템 가동! (종료: q, 설정초기화: r)")

    try:
        while True:
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
                        
                        # 특징 추출
                        foot_pt, torso_len, pose_type = get_features(result.keypoints[i], box_h)
                        
                        # 앙상블 거리 계산
                        real_x, dist, method = calculate_ensemble_distance(foot_pt, torso_len, h, H)
                        
                        # 위험 판단
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

                        # 그리기
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, f"{status} {dist:.1f}m ({method})", (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 레이더 표시
            radar_img = draw_separate_radar(detected_objects, current_alert=max_alert)
            cv2.imshow('Main Camera', frame)
            cv2.imshow('Radar View', radar_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                print("설정 초기화...")
                picam2.stop()
                cv2.destroyAllWindows()
                os.remove(CONFIG_FILE)
                # 재시작 (재귀 호출 대신 루프 방식이 안전하나, 간단히 재실행 유도)
                check_calibration() # 다시 캘리브레이션
                pixel_points = np.load(CONFIG_FILE)
                H = compute_homography(pixel_points)
                picam2 = init_camera() # 카메라 재시작

    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
try:
    from picamera2 import Picamera2
except ImportError:
    print("'picamera2' 라이브러리가 없습니다. 설치해주세요.")
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

# 1차 거리 공식 계수 (어깨-골반 기준)
ALPHA = 830.93
BETA = 1.09

# 2차 곡선 보정 계수
CORRECT_A = -0.036743
CORRECT_B = 1.604291
CORRECT_C = -2.325063

REALITY_SCALE = 1.0 # 현장 상황에 따른 미세 조정

# ==========================================
# [모듈 0] 카메라 초기화
# ==========================================
def init_camera():
    print("PiCamera2 초기화 중...")
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    print("카메라 시작됨! (Warmup 2초)")
    time.sleep(2)
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
                print(f"포인트 {len(clicked_points)+1} 입력: [{x}, {y}]")
                clicked_points.append([x, y])

    window_name = "Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    while True:
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

# ★ 2차 곡선 보정 함수
def apply_curve_correction(raw_dist):
    corrected_dist = (CORRECT_A * (raw_dist ** 2)) + (CORRECT_B * raw_dist) + CORRECT_C
    return max(0.0, corrected_dist)

# ★ 거리 계산 로직 (보정 적용)
def calculate_distance_hybrid(u, v, box_h, img_h, H):
    is_clipped = v >= (img_h * 0.95)
    
    if is_clipped:
        # 통계 공식 사용 (1차 -> 2차 보정)
        raw_dist = (ALPHA / box_h) + BETA
        method = "Stat"
        real_x = (u - (img_h * 1.33 / 2)) * raw_dist * 0.002
    else:
        # 호모그래피 사용 (1차 -> 2차 보정)
        pt = cv2.perspectiveTransform(np.array([[[u, v]]], dtype=np.float32), H)
        real_x = pt[0][0][0]
        raw_dist = pt[0][0][1]
        method = "Homo"
    
    # 최종 보정 적용
    final_dist = apply_curve_correction(raw_dist)
    
    return real_x, final_dist * REALITY_SCALE, method

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
    print("\n시스템 가동! (종료: q, 리셋: r)")
    model = YOLO(MODEL_PATH) 

    while True:
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
    picam2 = init_camera()
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
        picam2.stop()
        cv2.destroyAllWindows()
        print("종료 완료")

if __name__ == "__main__":
    main()

