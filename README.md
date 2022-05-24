# ElectraClassifierModeling
SequenceClassification, TokenClassification 학습 시 기존 Dense/linear 를 CNN/LSTM 으로 변경하여 테스트

## Classifier Fine-tuning의 이해
### 학습 형태의 이해
![image](https://user-images.githubusercontent.com/45644085/169961080-1e8f3f67-cbbb-4a3c-84d9-59b0bb02e322.png)

분류를 위한 학습 시 에 Electra, Bert 모델을 통해 임베딩되어 Linear 를 태우기 전의 형태는 Batch_size X Sequence_lenth X Embedding_size와 같다.

학습을 하기 위해 task에 맞추어진 example을 feature로 변경한 후 모델을 통해 나온 hidden_state 중 가장 마지막을 가져온다.
마지막 hidden_state는 초기의 임베딩 출력에서 각 hidden_state layer를 통과하며 추가된 최종 값을 의미한다. 
model의 마지막 hidden_state는 model에 input을 넣은 결과 튜플의 첫번째 값을 의미한다. (last_hidden_state = model(**input)[0])

해당 last_hidden_state는 Batch_size X Sequence_lenth X Embedding_size의 형태를 가지는데 실제로 한 개의 example(학습 단위 문장 혹은 문단)의 임베딩 값은 Sequence_lenth X Embedding_size 이라고 보면 될 것이다. 

![image](https://user-images.githubusercontent.com/45644085/169961793-b92431c2-2a79-4ac2-8e1b-248b09b54ec3.png)

"김일훈은 자연어처리 공부를 하는 중이다." 라는 문장이 가지는 임베딩 값은 다음과 같은 형태를 가질 것이다.
하이퍼파라미터를 통해 최대 문장 길이를 128, Pretrain model을 Base 모델로 사용하여 Embedding size가 768이라고 가정한다면, 
해당 문장은 128 x 768의 사이즈를 가지게 된다.


### Sequence Classification 학습 구조
Transformer 계열의 모델에서 sequence 임베딩 정보를 output으로 만들 시에 재미있는 점이 있다.
바로 문장 가장 앞에 오는 특수 토큰인 [CLS]에 해당 문장의 모든 임베딩 정보가 담긴다는 것이다. 
그래서 x = last_hidden_state[:, 0, :] 의 형태를 가진다.

![image](https://user-images.githubusercontent.com/45644085/169959711-e4f05225-6422-4861-b194-4b10b2af27d8.png)

따라서 한 문장 임베딩값의 형태는 다음과 같이 Embedding size이다.
파인튜닝에서는 Embedding size를 Linear를 통해 label 개수대로 학습 시킨 후 확률을 계산하면 되는 일이다.

만약 긍부정 이진 분류라면 다음과 같은 형태일 것이다.
![image](https://user-images.githubusercontent.com/45644085/169960741-34d3e0f4-44e0-4ee7-b370-403df899235d.png)

해당 형태를 컴퓨팅 사양에 맞게 batch size만큼 수행하는 것이 문장 분류 파인튜닝이다. 

### Question_Answering 학습 구조
일반적인 Sequence Classification과는 달리 나머지 파인튜닝의 경우는 [CLS]토큰이 아닌 모든 토큰의 임베딩을 사용한다.
어렵게 이유를 생각할 필요는 없다. output 형태가 (sequence_length X 2)가 되어야 하기 때문이다.
기계독해 학습의 경우는 각 토큰이 답의 시작(start_logits), 답의 끝(end_logits)이 될 확률을 계산하여 답의 시작이 되는 토큰과 답의 끝이 되는 토큰을 찾는 것이다.
그러므로 모든 토큰의 개수(sequence length)가 각각 2개의 확률을 가지므로 output 형태가 (sequence_length X 2)가 되는 것이다.
![image](https://user-images.githubusercontent.com/45644085/169963905-c4742d74-fab1-4a38-b390-c20794ec3f6b.png)


**modeling_electra.py** 파일 참조
