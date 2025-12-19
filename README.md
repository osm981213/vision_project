# 프로젝트 목적
YOLO를 기반한 CCTV 예측 모델 솔루션

[**팀 노션 링크**](https://www.notion.so/02-2bb3c414e0cc80ee9194dbbb3e74dd87?source=copy_link)

# 팀원
하준혁, 위영국, 오성민, 주홍연

# 기술 스택
```
Vision Model : YOLO
Language : python
Backend Web Framework : fastapi
Frontend Web Framework : vite, react
```

# 실행 방법
```
python 3.10 (추천 3.9도 가능 추측)
backend 에서 requirement.txt 설치
frontend 에서 npm install 이후 npm run (dev 패러미터 가능)
```

# 프로젝트 구조 설명
```
VISION_PROJECT
    └─solutions - 프로젝트에 사용될 솔루션들이 모인 폴더이다
        └─ CalibratedSpeed - 변수 기반 조정 속도 예측 솔루션
        └─ TOFSpeed TOF(Time of flight) 통과속도기반 속도 예측 솔루션
    └─platform - 프로젝트에 사용될 플랫폼 위치
        └─ backend 플랫폼의 백엔드
            └─ model 플랫폼에 사용될 모델 보관소
                model_registry.py 모델들의 format
                model.json 모델들의 명칭, 특징, 플랫폼에 제공될 명칭을 저장할 json
            └─ module 플랫폼에 사용될 모듈 보관소
                └─ CalibaratedSpeed 변수 기반 속도 측정 모듈
                └─ region Region 기반 통계 제공 모듈
                └─ TOFSpeed 통과 스피드 기반
                └─ utils 편의성 유틸
            └─ uploaded_videos 플랫폼을 통해 업로드를 할 시 저장할 폴더
        └─ frontend 플랫폼의 화면단 (Vite 기반)
```

# 플랫폼 + 통계 탐지 솔루션 
platform - backend

# 속도 탐지 솔루션 (변수 기반)
solutions - CalibratedSpeed (담당자 : 오성민)

# 속도 탐지 솔루션 (통과 속도 기반)
solutions - TOFSpeed (담당자 : 위영국)
