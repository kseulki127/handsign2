import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# Page Setup
st.set_page_config(page_title="한국 수화 통역기", page_icon="🤟", layout="centered")

st.title("🤟 한국 수화 실시간 통역기")
st.write("카메라에 수화(1, 3, 6)를 보여주면 실시간으로 인식하여 번역합니다.")

# 파일 경로 설정 (Streamlit Cloud 절대경로 문제 방지)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# GitHub에 저장된 모델 파일명에 맞게 선택 ('model.h5' 또는 'keras_model.h5')
MODEL_PATH = os.path.join(BASE_DIR, 'model.h5') 
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(BASE_DIR, 'keras_model.h5')

LABELS_PATH = os.path.join(BASE_DIR, 'labels.txt')

# Keras 모델 및 라벨 로드 (캐싱 적용)
@st.cache_resource
def load_keras_model():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    labels = []
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            cleaned_line = line.strip()
            if cleaned_line:
                # Teachable Machine 라벨 형식('0 1', '1 3', '2 6')에서 앞의 '0 ', '1 ' 등 인덱스 숫자를 잘라내고
                # 수화 의미인 '1', '3', '6'만 추출합니다.
                parts = cleaned_line.split(' ', 1)
                label_text = parts[1] if len(parts) > 1 else parts[0]
                labels.append(label_text)
            
    return model, labels

try:
    model, labels = load_keras_model()
    st.success("수화 번역 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    st.error(f"모델을 로드하는 중 오류가 발생했습니다: {e}")
    st.stop()

# 티쳐블 머신 전처리 함수 (224x224, 정규화)
def preprocess_frame(frame):
    # BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224 (Teachable Machine 기본 크기)
    resized = cv2.resize(rgb_frame, (224, 224), interpolation=cv2.INTER_AREA)
    
    # Array conversion & Reshape (1, 224, 224, 3)
    image_array = np.asarray(resized, dtype=np.float32)
    normalized_image_array = (image_array / 127.5) - 1.0
    data = np.expand_dims(normalized_image_array, axis=0)
    
    return data

# 비디오 프레임 처리 클래스
class SignLanguageProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # 모델 전처리 및 추론
        input_data = preprocess_frame(img)
        prediction = model.predict(input_data, verbose=0)
        index = np.argmax(prediction)
        class_name = labels[index]  # 정제된 '1', '3', '6' 값
        confidence_score = prediction[0][index]
        
        # 인식 결과 자막 생성 (예: "Sign: 1 (95%)")
        # OpenCV cv2.putText는 기본적으로 한글 폰트를 지원하지 않으므로 영문/숫자 표기로 설정하여 글자 깨짐 방지
        text = f"translate: {class_name} ({int(confidence_score * 100)}%)"
        
        # 영상 위에 결과 자막 출력 (초록색 글씨)
        cv2.putText(img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1.2, (0, 255, 0), 3, cv2.LINE_AA)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# STUN 서버 설정 (웹캠 연결 안정화용)
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# 실시간 비디오 스트리머 실행
webrtc_streamer(
    key="sign-language-detection",
    video_processor_factory=SignLanguageProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
