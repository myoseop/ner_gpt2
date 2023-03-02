# ner_gpt2

GPT2 Base 모델로 학습된 NER 모델

학습 데이터
--------
  #### KLUE NER 데이터셋 사용. 
  #### 6개의 개체명(사람, 위치, 기관, 날짜, 시간, 수량).
  #### 약 2.6만 어절
 
Requirements
------------
  #### pip3 install transformers
  #### CPU 인스턴스에서 사용. GPU 인스턴스 사용 시 CUDA 버전 문제 발생.
   - 에러 내용 : oserror torch_scatter/_version_cpu.so: undefined symbol
   - GPU 인스턴스에서 사용하려면 CUDA 버전과 Torch 버전을 맞춰주어야 함.
  #### Github 파일 용량 제한(100MB)으로 인해 모델 파일은 On-premise 서버의 아래 경로에 공유
   - /nfsdata/home/myoseop.sim/docker_storage/gpt_generation_NER/
   - 해당 경로의 파일 중 두개 파일(optimizer.pt ,  pytorch_model.bin) 을 코드의 gpt_generation_NER 디렉토리에 복사
 
Inference
---------


    python3 GPTinf_sample.py
    


  #### 샘플데이터 Process_sample.json 에 대해 Inference  
  
Result
------
  #### result.txt 파일 확인
