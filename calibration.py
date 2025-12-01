import cv2
import numpy as np
import time
from picamera2 import Picamera2

CONFIG_FILE = "camera_config.npy"

def init_camera():
    print("[캘리브레이션] 카메라 초기화 중...")
    picam2 = Picamera2()
    # [핵심] 해상도를 1640x1232로 설정하여 전체 시야(Full FOV) 확보
    config = picam2.create_preview_configuration(main={"size": (1640,1232), "format": "BGR888"})
    picam2.configure(config)
    picam2.start()
    time.sleep(2)
    return picam2

def main():
    picam2 = init_camera()
    print("\n=== [캘리브레이션 모드 - Wide View] ===")
    print("바닥의 1m, 2m, 3m, 4m 지점을 순서대로 클릭하세요.")
    
    clicked_points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points) < 4:
                # 좌표는 리사이즈된 640x480 화면 기준으로 찍힙니다.
                print(f"포인트 {len(clicked_points)+1} 입력: [{x}, {y}]")
                clicked_points.append([x, y])

    window_name = "Calibration"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)

    try:
        while True:
            # 1. 고해상도 원본 캡처 (넓은 시야)
            raw_frame = picam2.capture_array()
            
            # 2. 화면 출력 및 좌표 설정을 위해 640x480으로 압축 (시야 유지됨)
            frame = cv2.resize(raw_frame, (640, 480))
            
            # 안내 텍스트
            cv2.putText(frame, f"Points: {len(clicked_points)}/4", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            for i, pt in enumerate(clicked_points):
                cv2.circle(frame, tuple(pt), 5, (0, 0, 255), -1)
                cv2.putText(frame, f"{i+1}m", (pt[0]+10, pt[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow(window_name, frame)
            
            # 4개 다 찍으면 저장 후 종료
            if len(clicked_points) == 4:
                print("설정 완료! 저장 중...")
                cv2.waitKey(1000)
                pts_array = np.array(clicked_points, dtype=np.float32)
                np.save(CONFIG_FILE, pts_array)
                print(f"{CONFIG_FILE} 저장됨 (640x480 기준).")
                break
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("취소됨")
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
