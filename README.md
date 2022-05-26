# ElectraClassifierModeling
SequenceClassification, TokenClassification 학습 시 기존 단순 Linear FC layer를 CNN/LSTM 으로 변경하여 테스트

## Classifier Fine-tuning의 이해
### 학습 형태의 이해
![image](https://user-images.githubusercontent.com/45644085/169961080-1e8f3f67-cbbb-4a3c-84d9-59b0bb02e322.png)

분류를 위한 학습 시 에 Electra, Bert 모델을 통해 임베딩되어 Linear 를 통과하기 전의 형태는 Batch_size X Sequence_lenth X Embedding_size와 같다.

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
Sequence Classification은 한 Sequence가 라벨 개수만큼의 확률을 계산하면 되므로 Linear를 통과한 output 모양이 label 개수만큼 (1 x 라벨 개수)면 된다.
반면에 기계독해 학습의 경우는 각 토큰의 답의 시작(start_logits), 답의 끝(end_logits)이 될 확률을 계산하여 답의 시작이 되는 토큰과 답의 끝이 되는 토큰을 찾는 것이다.
그러므로 모든 토큰의 개수(sequence length)가 각각 2개의 확률을 가지므로 Linear output 형태가 (sequence_length X 2)가 되는 것이다.

학습 형태를 그림으로 설명하자면 다음과 같을 것이다.

![image](https://user-images.githubusercontent.com/45644085/169963905-c4742d74-fab1-4a38-b390-c20794ec3f6b.png)

(sequence length x 768) x Linear(768 X2)  = sequence length x 2 가 되는 간단한 구조이다.
 
## 학습 신경망 layer 변경해보기 
기존에 파인 튜닝의 경우 Linear (단순 선형 회귀)만을 사용한 형태이다.
파인튜닝 학습 형태인 Batch_size X Sequence_lenth X Embedding_size를 이해한다면 LSTM, CNN 등 다른 신경망 layer를 통과시켜서 파인튜닝을 해볼 수 있다.

~~~
class ElectraClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = get_activation("gelu")(x)  # although BERT uses tanh here, it seems Electra authors used gelu here
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
~~~

self.classifier = ElectraClassificationHead() 이고, logits = self.classifier(sequence_output)의 형태로 학습하는데 해당 ElectraClassificationHead의 
신경망 구조를 변경하면 다른 신경망 형태로 파인튜닝을 시도해볼 수 있다.

### CNN으로 신경망 변경하기

CNN 신경망 레이어를 그림으로 쉽게 설명하자면 아래와 같다.
![image](https://user-images.githubusercontent.com/45644085/170154390-0c9a56af-6ae1-47ae-88de-2ffff35fb52a.png)

~~~
class ElectraCnnClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1 ,out_channels=128, kernel_size=(2, 768), stride=1),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=(31,1), stride=1),
            torch.nn.ReLU()
        )
        self.out_proj = nn.Linear(128, config.num_labels)   
        
    def forward(self, x):
        x = self.dropout(x)
        x = get_activation("gelu")(x) 
        batch_size = x.size()[0]
        out = x.unsqueeze(1)
        out1 = self.layer1(out)
        out2 = self.layer2(out1)
        out3 = self.dropout(out2)
        out_view = out3.view(batch_size,128) 
        out_fin = self.out_proj(out_view)
        return out_fin
~~~

한 토큰이 가지는 임베딩이 1X768이라고 하면 Convolution layer를 통과할 때 2개의 토큰의 임베딩의 값 크기만큼 필터를 주어서 통과시키고 임의의   out channel 개수를 지정해준다.
그렇게 되면  Out channel X (Sequence length-1) 의 사이즈를 가지게 된다.
해당 Tensor를 Maxpooling layer를 통과시키면서 (Sequence length-1) X 1 의 필터를 주면 최종적으로는 Out channel 사이즈만큼의 Tensor가 남게 된다.
최종적으로는 이 Out channel 사이즈를 라벨 개수만큼 줄여주는 FC lyaer를 통과시키면 원하는 파인튜닝이 된다.

### BiLSTTM으로 신경망 변경하기
LSTM을 신경망으로도 변경이 가능하며 변경된 코드는 다음과 같다.
~~~
class ElectraBiLSTMClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LSTM = nn.LSTM(input_size=config.hidden_size, hidden_size=config.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(config.hidden_size*2, config.num_labels)
    def forward(self, features, **kwargs):
        x = self.dropout(features) 
        x = get_activation("gelu")(x) 
        h_1 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        c_1 = torch.zeros(2, x.size(0), self.hidden_size).to(device)
        out, (hn, cn) = self.LSTM(x, (h_1,c_1)) ### input = ((batch, hidden size, input size), (h_1,c_1))
        out = self.dropout(out)
        final_state = out[:,-1,:]
        x = self.out_proj(final_state)
        return x    
~~~

LSTM 신경망의 경우에 LSTM layer에 입력되는 input 형태가 조금 생소하다. (x, (h_1,c_1)
이는 아래의 그림을 통해 설명이 가능하다.

![image](https://user-images.githubusercontent.com/45644085/170165232-82a2d631-5002-489b-b386-4bfb9e3eba2a.png)

LSTM layer는 input과 이전의 단계의 hidden(h)과 cell(c)값을 다음 layer로 반환하여 준다.
그러므로 초기 hidden과 cell값을 h_1, c_1으로 함께 넣어주면 된다.

![image](https://user-images.githubusercontent.com/45644085/170165559-11736f39-f6ff-4b81-ba2f-04fbc5c02a47.png)

전체적인 LSTM layer의 형태는 다음과 같을 것이며 output, (hn, cn) 형태로 출력되는데 output(all the hidden states outputs)의 마지막 state를 가져와서 사용하기 위하여
out[:,-1,:]과 같이 마지막 state만 가져와서 FC layer에 넣어도 되고 hn을 그대로 FC layer에 넣어도 무방해보인다. 
biLstm은 양 방향에서 정보를 가져와서 담아서 out hidden size는 기존의 2배가 된다.

### 학습 결과 
|FC|CNN|BiLSTM|
|:-----|:---:|:---:|
|0.90644|0.90642|0.90758|

NSMC 파인튜닝 테스트 결과, 큰 차이는 없지만 BiLSTM이 근소하게 성능이 높은 것을 확인할 수 있다.

**modeling_electra.py** 파일 참조
