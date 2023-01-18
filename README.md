# Dacon 2023 교원그룹 AI 챌린지 <예선> Private 9등

## 팀원
### 한성대학교_김태민, LastStar, 3o3

## 사용한 모델 및 GPU
### Trocr-large-handwritten(https://arxiv.org/abs/2109.10282).
(https://huggingface.co/microsoft/trocr-large-handwritten)

RTX4090

### python=3.7
## 주요 설치 라이브러리
```
pip install transformers
pip install pandas
pip install Pillow
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
pip install scikit-learn
pip install datasets
pip install evaluate
pip install tqdm
pip install imgaug
pip install matplotlib
pip install imageio
```
## Dacon 2023 교원그룹 AI챌린지<예선> 코드 사용법
```
-훈련 데이터 경로./train
-테스트 데이터 경로./test
-학습 코드 : ./train1~4.py
-테스트 코드 : ./infer1~4.py
-앙상블 코드 : ./ESNB.py
``` 
## 처음부터 학습
```
-train1~4까지의 학습 파일을 실행 
실행 시 파일이 ./first ./second ./three ./four 폴더가 생성되고 그 안에 수많은 checkpoint-xxxx 폴더가 생성
그 후 infer1~4까지의 모든 파일을 실행 
실행 시 csv파일이 총 6개 나오는데 이때 ESNB파일을 실행하여 최종 submission_esnb_sota_FINAL.csv을 생성
```


## 가중치 파일로부터 추론
```
(first zip폴더는 구글 드라이브를 통해 링크를 받으시면 됩니다.)
first의 압축을 풀어 first,second,three,four 폴더를 다운받게 되면 아래 파일 구조에 맞게 다운받은 파일을 넣고
학습용 train과 test를 아래 폴더 구조에 맞게 넣어줍니다.  
infer1~4까지의 모든 파일을 실행 6개의 csv파일이 나오면 ESNB 파일을 실행하여 최종 submission_esnb_sota_FINAL.csv을 생성
```
## 파일 구조
```bash
├── git
│   ├── train1.py
│   ├── train2.py
│   ├── train3.py
│   ├── train4.py
│   ├── infer1.py
│   ├── infer1_2.py
│   ├── infer2.py
│   ├── infer3.py
│   ├── infer3_2.py
│   ├── infer4.py
│   ├── ESNB.py
│   ├── train
│   ├── test
│   ├── first
│   ├── second
│   ├── three
│   └── four

``` 
