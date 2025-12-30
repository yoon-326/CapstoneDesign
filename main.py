import src.common
import src.tilt.tilt_detection as td
import threading
import time
from picamera2 import Picamera2, Preview
import motion_detector as md
import cv2

# YOLOE 및 기능 모듈 임포트
from src.models.yoloe_loader import load_yoloe_model
from src.common.camera_input import init_camera1, init_camera2, get_frame
from src.detection.object_detection import run_inference
from src.tilt.tilt_detection import analyze_tilt_fast, analyze_tilt_hough
from src.common.visualization import draw_box, draw_label, show_frame

# pose 및 기능 모듈 임포트
from src.models.pose_loader import load_pose_model
from src.person_detection.distance_estimation import load_calibration_data, process_distance_estimation

# 공유 자원 및 조건 변수 생성
current_state = "STOPPED" # 상태 저장 변수
condition = threading.Condition() # Condition 객체 생성
model = load_yoloe_model()

pose_model = load_pose_model()
homography_matrix = load_calibration_data()

# [추가] 화면에 표시할 프레임을 저장할 공유 변수
global_display_frame = None
frame_lock = threading.Lock() # 프레임 쓰기/읽기 충돌 방지용

def set_display_frame(frame):
    """서브 스레드에서 결과 이미지를 업데이트하는 함수"""
    global global_display_frame
    with frame_lock:
        global_display_frame = frame

def car_moved_task(picam2): # [수정] picam2 인자 받도록 통일
    """차가 움직일 때 실행되는 태스크"""
    while True:
        with condition:
            condition.wait_for(lambda: current_state == "MOVING")
        
        # --- [실제 작업 영역] ---
        # print("car moved: monitoring...") # 로그 너무 많으면 주석 처리
        frame = get_frame(picam2)
            
        # 2. 거리 추정 로직 수행
        result_frame, objects = process_distance_estimation(pose_model, frame, homography_matrix)
        
        # 3. 콘솔 로그 (사람 감지 시)
        if objects:
            dist_str = ", ".join([f"{obj[1]:.1f}m" for obj in objects])
            print(f"[MOVING] Person Detected: {dist_str}")

        # 4. 화면 출력 대신 전역 변수 업데이트 [수정됨]
        set_display_frame(result_frame)
        
        # CPU 과점유 방지 (필요 시 미세 조정)
        time.sleep(0.01)

def car_stopped_task(picam2):
    frame_count = 0
    """차가 멈췄을 때 실행되는 태스크"""
    while True:
        with condition:
            condition.wait_for(lambda: current_state == "STOPPED")

        # --- [실제 작업 영역] ---
        # print("car stopped: detecting tilt...")
        frame = get_frame(picam2)
        frame = cv2.flip(frame, -1)
        frame_count += 1
        result = run_inference(model, frame, frame_count)

        if result:
            for box, cls in zip(result.boxes.xyxy, result.boxes.cls):
                x1, y1, x2, y2 = map(int, box)
                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                status, color, angle = analyze_tilt_hough(crop)
                label = f"{cls} | {status} {angle:.1f}°"

                draw_box(frame, x1, y1, x2, y2, color)
                draw_label(frame, label, x1, max(10, y1 - 10), color)

        # 화면 출력 대신 전역 변수 업데이트 [수정됨]
        # show_frame 내부에는 resize 로직이 있으므로 여기서 수동으로 resize 후 넘김
        display_frame = cv2.resize(frame, (640, 480))
        set_display_frame(display_frame)
        
        time.sleep(0.01)



if __name__ == "__main__":
    try:
        md.initialize_bmi160()
    except Exception as e:
        print(f"센서 초기화 실패, 안전 모드(MOVING)로 시작: {e}")

    
    picam0 = init_camera1()
    picam1 = init_camera2()
    # picam2.start() # [삭제] init_camera 내부에서 이미 start()를 호출함

    # 스레드 생성 (인자 통일)
    t1 = threading.Thread(target=car_moved_task, args=(picam0,), daemon=True)
    t2 = threading.Thread(target=car_stopped_task, args=(picam1,), daemon=True)
    
    t1.start()
    t2.start()
    
    last_state = None 

    print("System Started. Press 'q' to exit.")

    while True:
        # 1. 센서 상태 확인 및 상태 전환
        try:
            car_moving = md.check_motion_state()
        except NameError:
            car_moving = True

        new_state = "MOVING" if car_moving else "STOPPED"

        if new_state != last_state:
            with condition:
                current_state = new_state
                print(f"\n--- State changed to: {current_state} ---\n")
                condition.notify_all()
            last_state = new_state
        
        # 2. [핵심 수정] 메인 스레드에서 화면 출력 (GUI 이벤트 처리)
        current_display = None
        with frame_lock:
            if global_display_frame is not None:
                current_display = global_display_frame.copy()
        
        if current_display is not None:
            # 창 이름은 하나로 통일하는 것이 좋습니다
            cv2.imshow("Smart Forklift System", current_display)

        
        # waitKey는 메인 스레드에서만 호출!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # time.sleep(0.1) -> waitKey(1)이 sleep 역할을 일부 수행하므로 제거하거나 아주 짧게 설정