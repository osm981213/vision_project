# 그래픽카드의 최신 드라이버를 설치후 
# cmd에서 nvidia-smi를 치고 cuda version 확인하고 진행하기 바람

# gpu : rtx3070 기준
# cuda_version : 12.6
# cudnn : 9.6
# python : 3.10
# torch : 2.9.0+cu126
# torchvision : 0.24.0+cu126
# torchaudio : 2.9.0+cu126

# 이렇게 세팅해봄  wtdc폴더에서 돌려봤는데 cpu안쓰고 gpu쓰는거 확인완료

# conda create -n vision_project python=3.10
# conda activate vision_project

# YOLO 
# pip install ultralytics

# ultralytics를 install하면 torch 관련라이브러리를 다운받아서 혹시 모르니깐 지우고 하는게 좋음
# pip uninstall torch torchvision torchaudio

# torch관련 라이브러리(cuda 12.6) pip install
# pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu126 

# 잘 설치가 완료되었는지 확인
# conda list << 치게되면 뭐가 깔려있는지 나옴


# 아직 numpy 같은 다른 라이브러리는 출동이 생기는지 확인 못함 ㅠ

