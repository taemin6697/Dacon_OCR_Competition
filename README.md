# Dacon 2023 교원그룹 AI 챌린지 <예선> Private 9등
## 사용한 모델 
### Trocr-large-handwritten(https://arxiv.org/abs/2109.10282).

## 주요 설치 라이브러리
#### !pip install transformers
#### !pip install pandas
#### !pip install Pillow
#### !pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
#### !pip install scikit-learn
#### !pip install datasets
#### !pip install evaluate
#### !pip install tqdm
#### !pip install imgaug
#### !pip install matplotlib
#### !pip install imageio

## Dacon 2023 교원그룹 AI챌린지<예선> 코드 사용법
### -훈련 데이터 경로./train
### -테스트 데이터 경로./tesdt
### -학습 코드 : ./train1~4.py
### -테스트 코드 : ./infer1~4.py
### -앙상블 코드 : ./ESNB.py

## 처음부터 학습

### train1~4까지의 학습 파일을 실행 
### 실행 시 파일이 ./first ./second ./three ./four 폴더가 생성되고 그 안에 수많은 checkpoint-xxxx 폴더가 생성
### 그 후 infer1~4까지의 모든 파일을 실행 
### 실행 시 csv파일이 총 6개 나오는데 이때 ESNB파일을 실행하여 최종 submission_esnb_sota_FINAL.csv을 생성

## 가중치 파일로부터 추론
### first,second,three,four 폴더를 다운받게 되면 아래 파일 구조에 맞게 다운받은 파일을 넣고 
### infer1~4까지의 모든 파일을 실행 6개의 csv파일이 나오면 ESNB 파일을 실행하여 최종 submission_esnb_sota_FINAL.csv을 생성
