import cv2
import numpy as np
import time
from scipy.optimize import curve_fit

# [통합 모듈 임포트]
# 프로젝트 구조에 맞춰 src 폴더에서 가져옵니다.
from src.common.camera_input import init_camera1, get_frame
from src.models.pose_loader import load_pose_model

# ==========================================
# [함수] 데이터 처리
# ==========================================
def get_torso_length(keypoints):
    """ 어깨 중점 ~ 골반 중점 길이 (상반신) """
    kps = keypoints.data[0].cpu().numpy()
    
    # 8개 관절 모델 기준 인덱스
    # 0: Left Shoulder, 1: Right Shoulder
    # 2: Left Hip,      3: Right Hip
    l_sh, r_sh = kps[0], kps[1]
    l_hip, r_hip = kps[2], kps[3]
    
    # 신뢰도 체크 (0.5 미만이면 무시)
    if l_sh[2] < 0.5 or r_sh[2] < 0.5 or l_hip[2] < 0.5 or r_hip[2] < 0.5:
        return None
    
    shoulder_y = (l_sh[1] + r_sh[1]) / 2
    hip_y = (l_hip[1] + r_hip[1]) / 2
    
    # 절대값 반환
    return abs(hip_y - shoulder_y)

# 1차 공식 함수 형태 (반비례)
def inverse_func(h, a, b):
    return a * (1/h) + b

# ==========================================
# [메인] 실행 로직
# ==========================================
def main():
    print("[시스템] 통합 환경에서 파라미터 산출을 시작합니다.")
    
    # 1. 통합 모델 로드
    model = load_pose_model()
    if model is None:
        print("모델 로드 실패. figure_pose.pt 경로를 확인하세요.")
        return

    # 2. 통합 카메라 초기화
    # (src/common/camera_input.py의 설정을 그대로 사용)
    # 이미 1640x1232로 설정되어 있음
    picam2 = init_camera1()
    
    torso_data = []      # 상반신 길이 (픽셀)
    real_dist_data = []  # 실제 거리 (미터)
    
    # 측정할 거리 목록
    target_distances = [1.0, 1.5, 2.0, 2.5, 3.0]
    current_idx = 0
    
    print("\n" + "="*40)
    print("파라미터 자동 산출 모드")
    print("화면의 지시에 따라 거리를 맞추고 'Space' 키를 누르세요.")
    print("="*40)
    
    try:
        while current_idx < len(target_distances):
            # 3. 프레임 획득 (통합 함수 사용)
            raw_frame = get_frame(picam2)
            
            # [중요] 640x480 리사이징
            # distance_estimation.py 로직과 해상도를 일치시켜야 
            # 산출된 파라미터(ALPHA, BETA)가 정상 작동합니다.
            frame = cv2.resize(raw_frame, (640, 480))
            
            target_dist = target_distances[current_idx]
            
            # 가이드 텍스트
            msg = f"Target: {target_dist}m -> Press SPACE"
            cv2.putText(frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # 데이터 개수 표시
            cv2.putText(frame, f"Collected: {len(torso_data)}/{len(target_distances)}", (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow("Parameter Update", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32: # Spacebar
                # 모델 추론
                results = model(frame, verbose=False)
                
                max_torso = 0
                found = False
                
                for result in results:
                    if result.keypoints is not None:
                        for i in range(len(result.keypoints)):
                            # 텐서 구조 맞추기
                            temp_wrapper = type('', (), {})()
                            temp_wrapper.data = result.keypoints.data[i].unsqueeze(0)
                            
                            torso = get_torso_length(temp_wrapper)
                            if torso and torso > max_torso:
                                max_torso = torso
                                found = True
                
                if found:
                    torso_data.append(max_torso)
                    real_dist_data.append(target_dist)
                    print(f"[OK] 거리 {target_dist}m : 상반신 {max_torso:.1f}px")
                    current_idx += 1
                    time.sleep(0.5)
                else:
                    print("사람을 찾지 못했습니다. 전신이 잘 보이게 서주세요.")
            
            elif key == ord('q'):
                print("중단됨.")
                break
        
        # 종료 처리
        picam2.stop()
        cv2.destroyAllWindows()

        # --------------------------------------------
        # 데이터 분석 및 계수 산출
        # --------------------------------------------
        if len(torso_data) >= 5:
            print("\n데이터 분석 중...")
            
            X = np.array(torso_data)
            y = np.array(real_dist_data)
            
            # 1. 1차 공식 계수 (Alpha, Beta)
            popt, _ = curve_fit(inverse_func, X, y)
            alpha, beta = popt
            
            # 2. 1차 예측값 생성
            y_pred_1st = (alpha / X) + beta
            
            # 3. 2차 보정 계수 (a, b, c)
            coeffs = np.polyfit(y_pred_1st, y, 2)
            a, b, c = coeffs
            
            print("\n" + "="*50)
            print("[최종 결과] 아래 내용을 src/person_detection/distance_estimation.py 에 붙여넣으세요!")
            print("="*50)
            print(f"ALPHA = {alpha:.2f}")
            print(f"BETA  = {beta:.2f}")
            print(f"\nCORRECT_A = {a:.6f}")
            print(f"CORRECT_B = {b:.6f}")
            print(f"CORRECT_C = {c:.6f}")
            print("="*50)
            
        else:
            print("데이터가 부족하여 계산을 중단합니다.")

    except Exception as e:
        print(f"에러 발생: {e}")
        try:
            picam2.stop()
        except:
            pass
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()