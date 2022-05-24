# ElectraClassifierModeling
SequenceClassification, TokenClassification 학습 시 기존 Dense/linear 를 CNN/LSTM 으로 변경하여 테스트

## Classifier Fine-tuning의 이해
### 학습 형태의 이해
![image](https://user-images.githubusercontent.com/45644085/169935670-b0359798-fe13-41f4-b985-d2b70452d178.png)

분류를 위한 학습 시 에 Electra, Bert 모델을 통해 임베딩되어 Linear 를 태우기 전의 형태는 Batch_size X Sequence_lenth X Embedding_size와 같다.

task에 맞추어진 example을 feature로 변경한 후 모델을 통해 나온 hidden_state 중 가장 마지막을 가져온다.
마지막 hidden_state는 초기의 임베딩 출력에서 각 hidden_state layer를 통과하며 추가된 최종 값을 의미한다. 
model의 마지막 hidden_state는 model에 input을 넣은 결과 튜플의 첫번째 값을 의미한다. (last_hidden_state = model(**input)[0])






**modeling_electra.py** 파일 참조
