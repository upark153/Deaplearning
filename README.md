# Deaplearning
- - -
인공신경망

딥러닝(deep learning)이란 용어는 2006년에 등장했다.

인공 신경망(artificial neural network) 연구는, 인간의 뉴런(neuron)과 비슷한 메커니즘을 인공적으로 구현하면, 

'지능'이 있는 인간을 닮은 무언가를 만들 수 있을 것이란 상상에서 시작했다.
- - -
![image](https://user-images.githubusercontent.com/115389450/233240142-f8588c63-2120-4349-9db7-fe3b65c57deb.png)
- - -
1943년

'최초의 인공 신경망' 모델인 매컬러-피츠(McCulloch-Pitts) 모델은 인간의 신경계를 이진 뉴런으로 표현 했다.

매컬러와 피츠는 이진 뉴런 인공 신경망이 튜링 머신과 동등한 연산을 수행함을 증명했고, 

인간의 두뇌가 수학적으로 논리 연산과 산술 연산이 가능한 강력한 연산 장치임을 증명하고자 했다.
- - -
1956년

퍼셉트론(perceptron)은 초기 형태의 인공 신경망으로 다수의 입력으로부터 하나의 결과를 내보내는 알고리즘이다.

프랭크 로젠블랫(Frank Rosenblatt)이 개발했다.

퍼셉트론의 학습 알고리즘은 헵의 학습 가설에 따라 새로운 입력에 대한 오차가 발생하면 뉴런의 연결 강도를 조절하는 방식이다.

퍼셉트론을 이용해서 알파벳과 숫자를 인식하는데 성공했고, 뉴욕 타임스는 당시의 사회적 기대감과 분위기를 다음과 같이 전했다.

```
“머지않아 퍼셉트론이 사람을 인식하고 이름을 부르며,실시간으로 통역을 하고 글을 번역하는 날이 올 것이다”

- 1958년 7월 7일 뉴욕 타임스 -
```
- - -
# 학습 준비
1. 파이썬 패키지, 모듈 가져오기


문제 1

- 아래와 같이 패키지와 모듈을 가져올 때, 경고가 발생합니다.
- 경고 없이 실행하세요.
- 힌트1. 링크 를 참고하세요.
- 힌트2. from, import 의 의미를 검색하세요
- 힌트3. 단어를 잘 보세요.

![image](https://user-images.githubusercontent.com/115389450/233264042-eab6cc76-f5f1-420c-8b90-3b4df8549c7b.png)
Answer 1
- [텐서플로, 케라스 비교](https://backgomc.tistory.com/78)
- [케라스 소스코드 - models, layers](https://github.com/keras-team/keras)
- [텐서플로 소스코드 - import keras lib](https://github.com/tensorflow/tensorflow/blob/57a903a9ea32f02731a1e89b8067b20206633ae7/tensorflow/api_template.__init__.py#L91.)

# 1. 텐서플로와 케라스의 구분
```
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense
```
# 2. 패키지와 모듈, 함수 사용에 대한 구분
```
import tensorflow as tf
model = tf.keras.Sequential()

import tensorflow as tf
from keras.models import Sequential
model = Sequential()
```
# 3. 참고
```
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
```

# 2. 텐서플로 공식 문서 사용법
문제 2
- [tensorflow 공식 문서](https://www.tensorflow.org/api_docs/python/tf) 에서 random.normal( ) 정보 찾기
- 찾은 정보로 랜덤값 1개 출력하기

정답 2
- [tf.random.normal](https://www.tensorflow.org/api_docs/python/tf/random/normal)
```
import tensorflow as tf

tf.random.set_seed(2023)
```
```
print(tf.random.normal([1,], 0, 1))
```
> tf.Tensor([-1.1771783], shape=(1,), dtype=float32)

문제 3
- [tensorflow 가이드](https://www.tensorflow.org/guide?hl=ko) 에서 랜덤 숫자 생성 찾기
- 정규분포 (normal distribution) 함수를 찾아서
- 난수 1개 생성하여 출력하기
- 랜덤 8개 생성하여 출력하기
- 2번과 비교해보기 (버전)

정답 3
- [랜덤 숫자 생성 가이드](https://www.tensorflow.org/guide/random_numbers?hl=ko)
- 주의: tf.random.uniform 와 tf.random.normal 같은 구버전 TF 1.x의 RNG들은 아직 삭제되지 않았지만 사용을 권장하지 않습니다.

```
import tensorflow as tf

rand = tf.random.Generator.from_seed(1)
```
```
print(rand.normal(shape=[2]))
```
>tf.Tensor([-0.79253083  0.37646857], shape=(2,), dtype=float32)

# 2 퍼셉트론이란?
1. 입력 데이터(x)가 들어온다.
2. 입력 데이터(x)와 가중치(w)를 곱해서 "가중 합산"을 한다.
3. "가중 합산" 결과를 "계단 함수(step function)"를 통해 평가한다.
4. "계단 함수(step function)"는 0보다 크면 1을 출력하고, 그렇지 않으면 0을 출력한다.
5. "계단 함수(step function)"는 퍼셉트론의 활성 여부를 결정하기에 "활성 함수(activation function)"라고 부른다.

## 2.1 퍼셉트론 1:1
### 2.1.1 학습 목표
- 퍼셉트론의 목적 
  - 입력값(x)
  - 출력값(y)
- 페셉트론의 실행구조 
  - 가중치(w)
  - 가중합(s)
- 페셉트론의 활성화 
  - 예측값(pred_y)
  - 활성함수(step, sigmoid)
- 퍼셉트론의 학습 
  - 오차(e)
  - 학습률(lr)
![image](https://user-images.githubusercontent.com/115389450/233268567-c8dcab90-61f8-4173-8962-68bbbede29b9.png)
![image](https://user-images.githubusercontent.com/115389450/233268638-720a68a2-7c55-463a-bed0-c6b4e2d37bf6.png)

### 2.1.2 퍼셉트론의 목적과 실행구조 이해
```
import tensorflow as tf
import numpy as np


# 퍼셉트론의 목적 ---------------------------------------------------------------------------

x = 1  # 입력값(x) 1 일때, 
y = 0  # 출력값(y) 0 을 예측하는 것을 목표로 한다.


# 페셉트론의 실행 구조 ----------------------------------------------------------------------

w = tf.random.normal([1,], 0, 1)  # 가중치(w)는 정규 분포의 무작위 값이다.
y_pred = x * w                    # 입력값(x)과 가중치(w)를 곱하여, 출력값(y)를 예측(y_pred)한다.


# 결과 출력 ---------------------------------------------------------------------------------
print(f"입력값(x):{x:>2}")
print(f"가중치(w):{round(float(w[0]), 5):{' '}>8}")
print(f"출력값(y):{y:{' '}>2}")
print(f"예측값(s):{round(float(y_pred[0]), 5):{' '}>8}")

print(f"\n퍼셉트론의 목적은, 입력값(x)가 {x}일 때, 출력값(y이) {y}가 예측이다.")
print(f"입력값(x)가 {x}일 때, 임의의 가중치(w) {w}를 곱하여 출력값과 같은 {y}을 만드는 임의의 가중치(w)를 찾아야 한다.")
```
> 입력값(x): 1

> 가중치(w):-0.18303

> 출력값(y): 0

> 예측값(s):-0.18303

> 퍼셉트론의 목적은, 입력값(x)가 1일 때, 출력값(y이) 0가 예측이다.
> 입력값(x)가 1일 때, 임의의 가중치(w) [-0.18302704]를 곱하여 출력값과 같은 0을 만드는 임의의 가중치(w)를 찾아야 한다.

### 2.1.2  퍼셉트론의 학습 이해 - 오차

```
import tensorflow as tf
import numpy as np

# 퍼셉트론의 목적 ---------------------------------------------------------------------------

x = 1  # 입력값(x) 1 일때, 
y = 0  # 출력값(y) 0 을 예측하는 것을 목표로 한다.

# 페셉트론의 실행 구조 ----------------------------------------------------------------------

w = tf.random.normal([1,], 0, 1)  # 가중치(w)는 정규 분포의 무작위 값이다.
y_pred = x * w                    # 입력값(x)과 가중치(w)를 곱하여, 출력값(y) 0 을 예측하는 것을 목표

# 페셉트론의 학습 ---------------------------------------------------------------------------

error  = y - y_pred               # 출력값과 예측값 사이의 오차값(error)을 계산

lr = 0.1                          # lr(학습률): 가중치 조정을 위한 하이퍼 파라미터
w1 = w + (x * lr * error)         # 오차값(e)을 기반으로, 가중치(w) 수정 w > w1


# 결과 출력 ---------------------------------------------------------------------------------
print(f"입력값(x) :{x:>2}")
print(f"가중치(w) :{round(float(w[0]), 5):{' '}>8}")
print(f"출력값(y) :{y:{' '}>2}")
print(f"예측값(s) :{round(float(y_pred[0]), 5):{' '}>8}")
print(f"오차값(e) :{round(float(error[0] ), 5):{' '}>8}")
print(f"수정값(w1):{round(float(w1[0] ), 5):{' '}>8}")
```
> 입력값(x) : 1

> 가중치(w) : 0.29743

> 출력값(y) : 0

> 예측값(s) : 0.29743

> 오차값(e) :-0.29743

> 수정값(w1): 0.26768

### 2.1.4 퍼셉트론의 실행 이해 - 활성화 함수

####  2.1.4.1 오차(e) 기반의 가중치(w) 수정에서 문제 발생

```
import tensorflow as tf
import numpy as np

# 퍼셉트론의 목적 ---------------------------------------------------------------------------

x = 1  # 입력값(x) 1 일때, 
y = 0  # 출력값(y) 0 을 예측하는 것을 목표로 한다.
w = tf.random.normal([1,], 0, 1)  # 가중치(w)는 정규 분포의 무작위 값이다.

# 페셉트론의 학습 ---------------------------------------------------------------------------

epoch = 1000                       # 에포크(epoch): 모든 샘플에 대해 한 번 실행되어 학습하는 것
lr = 0.01                          # lr(학습률): 가중치 조정을 위한 하이퍼 파라미터

# 학습 시작 ----------------------------------------------------------------------------------
for i in range(epoch) :
    y_pred = x * w                 # 입력값(x)에 대해, 가중치(w)를 곱하여, 예측값(y_pred) 도출
    error  = y - y_pred            # 실제값과 예측값 사이의 오차값(e)을 계산
    w = w + x * lr * error         # 오차값(e)을 기반으로, 가중치(w) 수정
    
    if i%100 == 0 :
        print(f"학습 횟수n: {i:>5}  ", end=' ')
        print(f"가중치(w):{round(float(w[0]), 5):{' '}>8}  ", end=' ')    
        print(f"출력(y): {y}  ", end=' ')
        print(f"예측(s):{round(float(y_pred[0]), 5):{' '}>8}  ", end=' ')
        print(f"오차(e):{round(float(error[0]),  5):{' '}<8}  ")

print("\n가중치 수정에 문제 발생")
```
```
학습 횟수n:     0   가중치(w):-0.14187   출력(y): 0   예측(s): -0.1433   오차(e):0.1433    
학습 횟수n:   100   가중치(w):-0.05193   출력(y): 0   예측(s):-0.05245   오차(e):0.05245   
학습 횟수n:   200   가중치(w):-0.01901   출력(y): 0   예측(s): -0.0192   오차(e):0.0192    
학습 횟수n:   300   가중치(w):-0.00696   출력(y): 0   예측(s):-0.00703   오차(e):0.00703   
학습 횟수n:   400   가중치(w):-0.00255   출력(y): 0   예측(s):-0.00257   오차(e):0.00257   
학습 횟수n:   500   가중치(w):-0.00093   출력(y): 0   예측(s):-0.00094   오차(e):0.00094   
학습 횟수n:   600   가중치(w):-0.00034   출력(y): 0   예측(s):-0.00034   오차(e):0.00034   
학습 횟수n:   700   가중치(w):-0.00012   출력(y): 0   예측(s):-0.00013   오차(e):0.00013   
학습 횟수n:   800   가중치(w):  -5e-05   출력(y): 0   예측(s):  -5e-05   오차(e):5e-05     
학습 횟수n:   900   가중치(w):  -2e-05   출력(y): 0   예측(s):  -2e-05   오차(e):2e-05     
```

#### 2.1.4.2 시그모이드 함수
```
# 시그모이드 함수 matplotlib 구현

import numpy as np
import matplotlib.pylab as plt

def sigmoid(y_pred) :
    return 1/(1 + np.exp(-y_pred))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x,y)
plt.ylim(-0.1, 1.1)
plt.xlim(-6.0, 6.0)
plt.show()
```
![image](https://user-images.githubusercontent.com/115389450/233269685-f245ffda-5c41-43ad-9204-a0d1b864f96b.png)
#### 2.1.4.3 오차 업데이트에 시그모이드 함수 사용
```
import tensorflow as tf
import numpy as np

# 퍼셉트론의 목적 ---------------------------------------------------------------------------

x = 1  # 입력값(x) 1 일때, 
y = 0  # 출력값(y) 0 을 예측하는 것을 목표로 한다.
w = tf.random.normal([1,], 0, 1)  # 가중치(w)는 정규 분포의 무작위 값이다.

# 페셉트론의 학습 ---------------------------------------------------------------------------

epoch = 1000                       # 에포크(epoch): 모든 샘플에 대해 한 번 실행되어 학습하는 것
lr = 0.1                          # lr(학습률): 가중치 조정을 위한 하이퍼 파라미터

# 활성화 함수 -------------------------------------------------------------------------------

def sigmoid(y_pred) :              # 활성화 함수 sigmoid() 정의의
    return 1/(1 + np.exp(-y_pred))

# 학습 시작 ----------------------------------------------------------------------------------
for i in range(epoch) :
    s = x * w                      # 입력값(x)에 대해, 가중치(w)를 곱하여, 예측값(y_pred) 도출
    pred_y = sigmoid(s)            # 활성화 함수(sigmoid)로 평가
    error  = y - pred_y            # 실제값과 예측값 사이의 오차값(e)을 계산
    w = w + x * lr * error         # 오차값(e)을 기반으로, 가중치(w) 수정정
    
    if i%100 == 0 :
        print(f"학습 횟수n: {i:>5}  ", end=' ')
        print(f"가중치(w):{round(float(w[0]), 5):{' '}>8}  ", end=' ')    
        print(f"출력(y): {y}  ", end=' ')
        print(f"예측(s):{round(float(y_pred[0]), 5):{' '}>8}  ", end=' ')
        print(f"오차(e):{round(float(error[0]),  5):{' '}<8}  ")


print(f"\n입력값(x)가 {x}일 때, 출력값(y)가 {y}이 결과로 예측되는, 최적의 가중치(w)를 학습한다.")
```
```
학습 횟수n:     0   가중치(w):-1.25585   출력(y): 0   예측(s):  -1e-05   오차(e):-0.27216  
학습 횟수n:   100   가중치(w):-4.62241   출력(y): 0   예측(s):  -1e-05   오차(e):-0.00983  
학습 횟수n:   200   가중치(w):-5.30527   출력(y): 0   예측(s):  -1e-05   오차(e):-0.00497  
학습 횟수n:   300   가중치(w):-5.70775   출력(y): 0   예측(s):  -1e-05   오차(e):-0.00332  
학습 횟수n:   400   가중치(w):-5.99409   출력(y): 0   예측(s):  -1e-05   오차(e):-0.00249  
학습 횟수n:   500   가중치(w):-6.21648   출력(y): 0   예측(s):  -1e-05   오차(e):-0.002    
학습 횟수n:   600   가중치(w):-6.39834   출력(y): 0   예측(s):  -1e-05   오차(e):-0.00166  
학습 횟수n:   700   가중치(w):-6.55218   출력(y): 0   예측(s):  -1e-05   오차(e):-0.00143  
학습 횟수n:   800   가중치(w):-6.68549   출력(y): 0   예측(s):  -1e-05   오차(e):-0.00125  
학습 횟수n:   900   가중치(w):-6.80311   출력(y): 0   예측(s):  -1e-05   오차(e):-0.00111  

입력값(x)가 1일 때, 출력값(y)가 0이 결과로 예측되는, 최적의 가중치(w)를 학습한다.\
```

## 문제 1
- 입력값이 0.91 일때, 출력값이 0을 찾는 학습 퍼셉트론 모델을 만드세요
- 학습은 3000번 실행하세요.
- 학습률을 0.01로 수정하세요.
- 가중치 수정에 대한 출력을 300번마다 출력 하세요.
```
import tensorflow as tf
import numpy as np

x = 0.91  # 입력값(x) 1 일때, 
y = 0  # 출력값(y) 0 예측하는 과정을 퍼샙트론의 이해하기

w = tf.random.normal([1,], 0, 1)  # 가중치(w)는 정규분포의 무작위 값(적절하지 않다.)

epoch = 3000                      # 에포크(epoch): 모든 샘플에 대해 한 번 실행되어 학습하는 것

lr = 0.1                         # lr(학습률): 가중치 조정을 위한 하이퍼 파라미터

def sigmoid(y_pred) :
    return 1/(1 + np.exp(-y_pred))
    
for i in range(epoch) :
    s = x * w                     # 입력값(x)에 대해, 가중치(w)를 곱하여, 예측값(y_pred) 도출
    pred_y = sigmoid(s)           # 활성화 함수(sigmoid)로 평가
    error  = y - pred_y           # 실제값과 예측값 사이의 오차값(e)을 계산
    w = w + x * lr * error        # 오차값(e)을 기반으로, 가중치(w) 수정정
    
    if i%300 == 0 :
        print(f"학습 횟수n: {i:>5}  ", end=' ')
        print(f"가중치w:{round(float(w[0]), 5):{' '}>8}  ", end=' ')    
        print(f"출력y: {y}  ", end=' ')
        print(f"예측y:{round(float(pred_y[0]), 5):{' '}<8}  ", end=' ')
        print(f"오차e:{round(float(error[0]),  5):{' '}<8}  ")


print(f"\n입력값x가 {x}일 때, 출력값y이 {y}가 예측되는, 최적의 가중치(w)를 학습을 {epoch}번 실행")
```
```
학습 횟수n:     0   가중치w: 0.18734   출력y: 0   예측y:0.55388    오차e:-0.55388  
학습 횟수n:   300   가중치w:-3.42182   출력y: 0   예측y:0.04268    오차e:-0.04268  
학습 횟수n:   600   가중치w:-4.22274   출력y: 0   예측y:0.02102    오차e:-0.02102  
학습 횟수n:   900   가중치w:-4.68566   출력y: 0   예측y:0.01389    오차e:-0.01389  
학습 횟수n:  1200   가중치w:-5.01174   출력y: 0   예측y:0.01036    오차e:-0.01036  
학습 횟수n:  1500   가중치w:-5.26347   출력y: 0   예측y:0.00825    오차e:-0.00825  
학습 횟수n:  1800   가중치w:-5.46845   출력y: 0   예측y:0.00686    오차e:-0.00686  
학습 횟수n:  2100   가중치w:-5.64133   출력y: 0   예측y:0.00586    오차e:-0.00586  
학습 횟수n:  2400   가중치w:-5.79078   출력y: 0   예측y:0.00512    오차e:-0.00512  
학습 횟수n:  2700   가중치w: -5.9224   출력y: 0   예측y:0.00455    오차e:-0.00455  

입력값x가 0.91일 때, 출력값y이 0가 예측되는, 최적의 가중치(w)를 학습을 3000번 실행
```
## 2.2 matplotlib
- [[WikiDocs] Matplotlib Tutorial - 파이썬으로 데이터 시각화하기](https://wikidocs.net/book/5011)

### 2.2.1 matplolib 한글 깨짐 해결
- 한글 지원 폰트를 찾아서 nanum 폰트 시스템에 적용
#### STEP 1. matplotlib 한글 깨짐 확인
```
한글 폰트를 설정하지 않고 matplotlib에 출력하면, 한글 출력이 깨진다.
```
```
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [1, 4, 9, 16],label='가격') #한글 깨짐 확인인
plt.xlabel('X-축')
plt.ylabel('Y-축')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/115389450/233276672-c922a647-4107-4561-9939-fd27cb3d8b7c.png)

#### STEP 2 시스템 폰트에서 한글 폰트 찾기
```
# matplotlib 폰트 관련 라이브러리 import
import matplotlib.font_manager as fm  

sys_font = fm.findSystemFonts()
print(f"시스템에 설치된 폰트 개수: {len(sys_font)}\n") # Nanum 포함된 폰트 확인인
for i, x in enumerate(sys_font):
    print(i, " ", x)
```
```
시스템에 설치된 폰트 개수: 27

0   /usr/share/fonts/truetype/nanum/NanumMyeongjoBold.ttf
1   /usr/share/fonts/truetype/liberation/LiberationSansNarrow-Regular.ttf
2   /usr/share/fonts/truetype/nanum/NanumGothic.ttf
3   /usr/share/fonts/truetype/humor-sans/Humor-Sans.ttf
4   /usr/share/fonts/truetype/liberation/LiberationSansNarrow-BoldItalic.ttf
5   /usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf
6   /usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf
7   /usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf
8   /usr/share/fonts/truetype/liberation/LiberationSerif-Bold.ttf
9   /usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf
10   /usr/share/fonts/truetype/liberation/LiberationSansNarrow-Bold.ttf
11   /usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf
12   /usr/share/fonts/truetype/nanum/NanumSquareB.ttf
13   /usr/share/fonts/truetype/liberation/LiberationSans-BoldItalic.ttf
14   /usr/share/fonts/truetype/nanum/NanumSquareRoundR.ttf
15   /usr/share/fonts/truetype/liberation/LiberationMono-BoldItalic.ttf
16   /usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf
17   /usr/share/fonts/truetype/nanum/NanumSquareR.ttf
18   /usr/share/fonts/truetype/nanum/NanumBarunGothicBold.ttf
19   /usr/share/fonts/truetype/liberation/LiberationMono-Italic.ttf
20   /usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf
21   /usr/share/fonts/truetype/nanum/NanumSquareRoundB.ttf
22   /usr/share/fonts/truetype/liberation/LiberationSerif-BoldItalic.ttf
23   /usr/share/fonts/truetype/nanum/NanumGothicBold.ttf
24   /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf
25   /usr/share/fonts/truetype/liberation/LiberationSans-Italic.ttf
26   /usr/share/fonts/truetype/liberation/LiberationSansNarrow-Italic.ttf
```
#### STEP 3.1 폰트 경로 리스트에 Nanum 관련 폰트가 있다면, 해당 폰트 설정
```
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

fontpath = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf' # 위 경로 중 nanum 폰트 파일 경로 입력
font = fm.FontProperties(fname=fontpath, size=8)
plt.rc('font', family='NanumGothic') # nanum 폰트 지정
plt.rc('axes', unicode_minus=False) # 마이너스 폰트 설정

# STEP4 로 이동
```
#### STEP 3.2 리스트에 Nanum 관련 폰트가 없다면, Nanum 폰트를 설치한다.
```
# 1. Nanum 한글 폰트 설치
!sudo apt-get install -y fonts-nanum
!sudo fc-cache -fv
!rm ~/.cache/matplotlib -rf

# 2. 설치 후, 코랩 상단의 런타임 > 런타임 다시 시작

# 3. STEP 2, STEP 3.1을 진행한다.
```
```
Reading package lists... Done
Building dependency tree       
Reading state information... Done
fonts-nanum is already the newest version (20180306-3).
0 upgraded, 0 newly installed, 0 to remove and 24 not upgraded.
/usr/share/fonts: caching, new cache contents: 0 fonts, 1 dirs
/usr/share/fonts/truetype: caching, new cache contents: 0 fonts, 3 dirs
/usr/share/fonts/truetype/humor-sans: caching, new cache contents: 1 fonts, 0 dirs
/usr/share/fonts/truetype/liberation: caching, new cache contents: 16 fonts, 0 dirs
/usr/share/fonts/truetype/nanum: caching, new cache contents: 10 fonts, 0 dirs
/usr/local/share/fonts: caching, new cache contents: 0 fonts, 0 dirs
/root/.local/share/fonts: skipping, no such directory
/root/.fonts: skipping, no such directory
/usr/share/fonts/truetype: skipping, looped directory detected
/usr/share/fonts/truetype/humor-sans: skipping, looped directory detected
/usr/share/fonts/truetype/liberation: skipping, looped directory detected
/usr/share/fonts/truetype/nanum: skipping, looped directory detected
/var/cache/fontconfig: cleaning cache directory
/root/.cache/fontconfig: not cleaning non-existent cache directory
/root/.fontconfig: not cleaning non-existent cache directory
fc-cache: succeeded
```
#### STEP 4 첫번째 STEP 1. 재실행 후 한글 출력 확인
```
import matplotlib.pyplot as plt

plt.plot([1, 2, 3, 4], [1, 4, 9, 16],label='가격') #한글 깨짐 확인인
plt.xlabel('X-축')
plt.ylabel('Y-축')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/115389450/233277273-ab0a468e-5340-45c4-bd15-0c390d224815.png)
### 2.2.2 코랩의 matplotlib 버전 확인
```
import matplotlib as plt

plt.__version__
```
### 2.2.3 그래프 만들기
- x_axis = [1,2,3,4], y_axis = [2,5,11,14] 그래프 출력
```
import matplotlib.pyplot as plt

x_axis = [1,2,3,4]
y_axis = [2,5,11,14]

plt.plot(x_axis, y_axis)
plt.show()

# plot: 선도
# 선도(線圖)란, 여러 값 사이의 수치적인 관계를 좌표계에 점으로 찍어서 선 또는 면으로 나타낸 것이다.
# 플롯(plot)과 그래프(graph)는 둘 다 '선도(線圖)'라는 뜻
```
![image](https://user-images.githubusercontent.com/115389450/233277571-9eb4a343-2d58-4334-b846-b884a3c8ac5b.png)
```
문제 2

matpolotlib 패키지를 사용해서 그래프를 만드세요.

plot함수에 x축 입력값 없이, y축만 y_axis = [2,5,11,17] 인 그래프 출력하세요.

그리고 2번과의 차이점을 찾으세요
```
```
import matplotlib.pyplot as plt

y_axis = [2,5,11,14]
plt.plot(y_axis)
plt.show()
```
![image](https://user-images.githubusercontent.com/115389450/233277749-3c3e702a-b9ff-4a57-885d-0b3acc29d663.png)
```
문제 3

matpolotlib 패키지를 사용해서 그래프를 만드세요.

1. x축은, numpy arrange() 사용해서, 1 ~ 5 사이에 0.1 단위로 사용하세요.

2. y축은, y = 3.2x + 1.8 함수를 구현하여 사용하세요.
```
```
import numpy as np
import matplotlib.pylab as plt

def func(x):
  y = (3.2*x) + 1.8
  return y

x = np.arange(1, 5.0, 0.1)
y2 = func(x)

plt.plot(x, y2)
plt.show()
```
![image](https://user-images.githubusercontent.com/115389450/233277874-99c5b91c-52ee-4af7-9b49-ecb773007e86.png)

```
문제 4

matpolotlib 패키지를 사용해서 그래프를 만드세요.

1. 넘파이 arrange() 함수를 사용해서, 1 ~ 5 사이에 0.5 단위로 x 축 값을 만들고,

2. x의 제곱을 y 축으로 그래프에 출력하세요.

3. x의 세제곱을 y 축으로 그래프에 출력하세요.
```
```
import matplotlib.pyplot as plt
import numpy as np

# 0.1 간격으로 균일한 x축
x = np.arange(0, 5, 0.5)

# 녹색 대쉬, 노란 사각형
plt.plot(x, x**2, 'ys')
plt.show()
```
![image](https://user-images.githubusercontent.com/115389450/233278106-d49515ed-3790-4d17-ab15-39b60b4f2d48.png)
```
문제 5

문제 4번의 그래프를, "빨간색 원형('o')" 마커를 사용한 그래프를 출력하세요.
```
```
import matplotlib.pyplot as plt
import numpy as np

# 0.1 간격으로 균일한 x축
x = np.arange(0, 5, 0.5)

# 녹색 대쉬, 노란 사각형
plt.plot(x, x**2, 'ro')
plt.show()
```
![image](https://user-images.githubusercontent.com/115389450/233278225-3bafb0ce-07d2-4fa6-868e-1de6c6373d79.png)
```
문제 6

4번의 그래프에서, 녹색 원형('*') 마커에 -- 점선으로 그래프를 출력하세요.
```
```
import matplotlib.pyplot as plt
import numpy as np

# 0.5 간격으로 균일한 x축
x = np.arange(0, 5, 0.5)

# 녹색 대쉬, 노란 사각형
plt.plot(x, x**2, 'g--')
plt.show()
```
![image](https://user-images.githubusercontent.com/115389450/233278417-dbb61a90-63a0-471b-be29-2c27e89428c5.png)
```
문제 7 그래프 여러개 그리기 연습

1. 넘파이 arrange() 함수를 사용해서, 1 ~ 10 사이에 0.4 단위로 x 축 값을 만들고,

2. x의 제곱과 x의 세제곱을 y 축으로 그래프에 동시에출력하세요.
```
```
import matplotlib.pyplot as plt
import numpy as np

# 1 ~ 10 사이에 0.4 간격 x값
x = np.arange(0, 10, 0.4)

# 빨간 대쉬, 파란 사각형, 녹색 삼각형
plt.plot(x, x, 'g--', x, x**2, 'ys', x, x**3, 'b^')
plt.show()
```
![image](https://user-images.githubusercontent.com/115389450/233279093-26fd7ca9-1a72-41f2-a119-b4fd8eaa3bfa.png)

```
*문제 8
"*문제 1"에서, 학습 횟수를 기준으로 '예측값y'와 '오차값e'의 변화를 그래프에 동시에 출력하세요.
```
```
import tensorflow as tf
import numpy as np

x = 0.91 
y = 0 

w = tf.random.normal([1,], 0, 1)

epoch = 3000 

lr = 0.1 

def sigmoid(y_pred) :
    return 1/(1 + np.exp(-y_pred))
    
# 그래프 출력 데이터 저장 리스트 준비 ----------------------------------------------------

lst_i = [] # 그래프 x 축 값 저장 리스트
lst_y = [] # 그래프 y 축 값 저장 리스트
lst_e = [] # 그래프 e 축 값 저장 리스트

    
for i in range(epoch) :
    s = x * w                     # 입력값(x)에 대해, 가중치(w)를 곱하여, 예측값(y_pred) 도출
    pred_y = sigmoid(s)           # 활성화 함수(sigmoid)로 평가
    error  = y - pred_y           # 실제값과 예측값 사이의 오차값(e)을 계산
    w = w + x * lr * error        # 오차값(e)을 기반으로, 가중치(w) 수정정
    
    if i%300 == 0 :
        print(f"학습 횟수n: {i:>5}  ", end=' ')
        print(f"가중치w:{round(float(w[0]), 5):{' '}>8}  ", end=' ')    
        print(f"출력y: {y}  ", end=' ')
        print(f"예측y:{round(float(pred_y[0]), 5):{' '}<8}  ", end=' ')
        print(f"오차e:{round(float(error[0]),  5):{' '}<8}  ")


        # 그래프 출력 데이터 리스트에 저장 ------------------------------------------------
        lst_i.append(i)
        lst_y.append(pred_y[0])
        lst_e.append(error[0])

# 그래프 출력 -----------------------------------------------------------------------------
plt.figure(figsize=(4,4))
plt.plot(lst_i, lst_y, 'r--', label='예측값y 변화')
plt.plot(lst_i, lst_e, label='오차값e 변화')
plt.xlabel('학습 횟수')
plt.ylabel('예측 결과')
plt.legend()
plt.show()
```
![image](https://user-images.githubusercontent.com/115389450/233279514-8f4e70e2-e985-4484-a84b-c015fc1dbf1c.png)

## 2.3 편향 bias
- 입력값이 0이기 때문에 어떠한 학습률 값을 넣어도 가중치 조정 불가\
- 이러한 경우를 방지하고자 '편향' 개념 등장
- 편향 : 한쪽으로 치우쳐진 고정 값
- 편향값도 가중치처럼 난수로 초기화되며 뉴런에 더해져 출력을 계산
### 2.3.1 입력(x)이 0 일때
```
문제 9
"*문제 1" 을 참고해서

입력값 0, 출력값 1 으로 실행하는 코드를 작성해보고, 차이점을 찾아보세요
```
```
import tensorflow as tf
import numpy as np

x = 0     # 입력값(x) 0 일때, 
y = 1     # 출력값(y) 1 예측하는 과정을 퍼샙트론의 이해하기

w = tf.random.normal([1,], 0, 1)  # 가중치(w)는 정규분포의 무작위 값(적절하지 않다.)

epoch = 3000                      # 에포크(epoch): 모든 샘플에 대해 한 번 실행되어 학습하는 것

lr = 0.1                         # lr(학습률): 가중치 조정을 위한 하이퍼 파라미터

def sigmoid(y_pred) :
    return 1/(1 + np.exp(-y_pred))
    
for i in range(epoch) :
    s = x * w                     # 입력값(x)에 대해, 가중치(w)를 곱하여, 예측값(y_pred) 도출
    pred_y = sigmoid(s)           # 활성화 함수(sigmoid)로 평가
    error  = y - pred_y           # 실제값과 예측값 사이의 오차값(e)을 계산
    w = w + x * lr * error        # 오차값(e)을 기반으로, 가중치(w) 수정정
    
    if i%300 == 0 :
        print(f"학습 횟수n: {i:>5}  ", end=' ')
        print(f"가중치w:{round(float(w[0]), 5):{' '}>8}  ", end=' ')    
        print(f"출력y: {y}  ", end=' ')
        print(f"예측y:{round(float(pred_y[0]), 5):{' '}<8}  ", end=' ')
        print(f"오차e:{round(float(error[0]),  5):{' '}<8}  ")


print(f"\n입력값(x)가 {x}일 때, 출력값y이 {y}가 예측되는, 최적의 가중치(w)를 학습을 {epoch}번 실행")
print(f"입력값(x)가 {x}일 때, 최적의 가중치(w)를 찾기 위한 업데이트가 불가능하다.")
```
```
학습 횟수n:     0   가중치w: 0.68148   출력y: 1   예측y:0.5        오차e:0.5       
학습 횟수n:   300   가중치w: 0.68148   출력y: 1   예측y:0.5        오차e:0.5       
학습 횟수n:   600   가중치w: 0.68148   출력y: 1   예측y:0.5        오차e:0.5       
학습 횟수n:   900   가중치w: 0.68148   출력y: 1   예측y:0.5        오차e:0.5       
학습 횟수n:  1200   가중치w: 0.68148   출력y: 1   예측y:0.5        오차e:0.5       
학습 횟수n:  1500   가중치w: 0.68148   출력y: 1   예측y:0.5        오차e:0.5       
학습 횟수n:  1800   가중치w: 0.68148   출력y: 1   예측y:0.5        오차e:0.5       
학습 횟수n:  2100   가중치w: 0.68148   출력y: 1   예측y:0.5        오차e:0.5       
학습 횟수n:  2400   가중치w: 0.68148   출력y: 1   예측y:0.5        오차e:0.5       
학습 횟수n:  2700   가중치w: 0.68148   출력y: 1   예측y:0.5        오차e:0.5       

입력값(x)가 0일 때, 출력값y이 1가 예측되는, 최적의 가중치(w)를 학습을 3000번 실행
입력값(x)가 0일 때, 최적의 가중치(w)를 찾기 위한 업데이트가 불가능하다.
```
### 2.3.2 편향 개념 추가
```
문제 10

 입력이 0 일때 발생하는 문제를 해결하기 위해 '편향 bias' 개념이 필요합니다.  
 *문제 1 을 참고해서 아래 내용을 추가하여 코드를 완성하세요
 
1. 가중치 소스 코드를 참고해서, 편향을 추가 하세요.
  - b = tf.random.normal([1], 0, 1)
2. 입력값(x)에 대해 가중치(w)를 곱해서 예측값(y_pred)를 구했습니다.
  - y_pred = x * w 를 예측값(y_pred)에 편향(bias)를 더하는 코드로 변경하세요
  - y_pred = x * w + (1 * b)
3. 가중치(w) 업데이트 소스코드를 참고하여, 편향(b)을 업데이트 하는 코드를 추가하세요
  - b = b + 1 * 0.1 * error # 편향값 업데이트
4. 예측(output[0])를 출력하는 print 문을 참고해서, 편향(b) 업데이트 값을 추가로 출력하세요.

```
```
import tensorflow as tf
import numpy as np

# 입력
x = 0

# 출력
y = 1

# 가중치 : 정규분포의 무작위 값
w = tf.random.normal([1,], 0, 1)

# 편향 추가
b = tf.random.normal([1], 0, 1) 

# lr(학습률): 가중치 조정을 위한 하이퍼 파라미터
lr = 0.1                         

# 에포크(epoch): 모든 샘플에 대해 한 번 실행되어 학습하는 것
epoch = 3000                      

# 활성화 함수(Activation Function) 역전파 과정에서 미분값을 통해 학습이 진행될 수 있게 합니다.
def sigmoid(y_pred) :
    return 1/(1 + np.exp(-y_pred))
    
for i in range(epoch) :
    s = x * w + (1 * b)           # 입력값(x)에 대해, 가중치(w)를 곱하여, 예측값(y_pred) 도출
    pred_y = sigmoid(s)           # 활성화 함수(sigmoid)로 평가
    error  = y - pred_y           # 실제값과 예측값 사이의 오차값(e)을 계산
    w = w + x * lr * error        # 오차값(e)을 기반으로, 가중치(w) 수정정
    
    if i%300 == 0 :
        print(f"학습 횟수n: {i:>5}  ", end=' ')
        print(f"가중치w:{round(float(w[0]), 5):{' '}>8}  ", end=' ')    
        print(f"출력y: {y}  ", end=' ')
        print(f"예측y:{round(float(pred_y[0]), 5):{' '}<8}  ", end=' ')
        print(f"오차e:{round(float(error[0]),  5):{' '}<8}  ")


print(f"\n입력값x가 {x}일 때, 출력값y이 {y}가 예측되는, 최적의 가중치(w)를 학습을 {epoch}번 실행")
```
```
학습 횟수n:     0   가중치w: 1.22127   출력y: 1   예측y:0.26074    오차e:0.73926   
학습 횟수n:   300   가중치w: 1.22127   출력y: 1   예측y:0.26074    오차e:0.73926   
학습 횟수n:   600   가중치w: 1.22127   출력y: 1   예측y:0.26074    오차e:0.73926   
학습 횟수n:   900   가중치w: 1.22127   출력y: 1   예측y:0.26074    오차e:0.73926   
학습 횟수n:  1200   가중치w: 1.22127   출력y: 1   예측y:0.26074    오차e:0.73926   
학습 횟수n:  1500   가중치w: 1.22127   출력y: 1   예측y:0.26074    오차e:0.73926   
학습 횟수n:  1800   가중치w: 1.22127   출력y: 1   예측y:0.26074    오차e:0.73926   
학습 횟수n:  2100   가중치w: 1.22127   출력y: 1   예측y:0.26074    오차e:0.73926   
학습 횟수n:  2400   가중치w: 1.22127   출력y: 1   예측y:0.26074    오차e:0.73926   
학습 횟수n:  2700   가중치w: 1.22127   출력y: 1   예측y:0.26074    오차e:0.73926   

입력값x가 0일 때, 출력값y이 1가 예측되는, 최적의 가중치(w)를 학습을 3000번 실행
```
```
문제 11

문제 10번을 참고하여, 아래 내용을 추가하세요

1. matplotlib을 사용하여 그래프에 예측값(y), 편향(b), 오차(e)를 그래프에 한번에 출력하세요.

2. 그래프의 색과 종류가 다르게 설정하세요.

3. 각 x 축, y 축 이름 을 출력하세요.

4. 각 예측값(y), 편향(b), 오차(e) 과 각 범례(legend)를 출력하세요.
```
```
import tensorflow as tf
import numpy as np

# 입력
x = 0

# 출력
y = 1

# 가중치 : 정규분포의 무작위 값
w = tf.random.normal([1,], 0, 1)

# 편향 추가
b = tf.random.normal([1], 0, 1) 

# lr(학습률): 가중치 조정을 위한 하이퍼 파라미터
lr = 0.1                         

# 에포크(epoch): 모든 샘플에 대해 한 번 실행되어 학습하는 것
epoch = 3000                      

# 활성화 함수(Activation Function) 역전파 과정에서 미분값을 통해 학습이 진행될 수 있게 합니다.
def sigmoid(y_pred) :
    return 1/(1 + np.exp(-y_pred))
    
lst_i = [] # 그래프 x 축 값 저장 리스트
lst_y = [] # 그래프 y 축 값 저장 리스트
lst_e = [] # 그래프 e 축 값 저장 리스트
lst_b = [] # 그래프 b 값 저장 리스트

# 학습 시작
for i in range(epoch) :
    s = x * w + (1 * b)           # 입력값(x)에 대해, 가중치(w)를 곱하고, 편향(b)를 더하여, 가중합 도출
    pred_y = sigmoid(s)           # 활성화 함수(sigmoid)로 평가
    error  = y - pred_y           # 실제값과 예측값 사이의 오차값(e)을 계산
    w = w + x * lr * error        # 오차값(e)을 기반으로, 가중치(w) 수정
    b = b + 1 * lr * error        # 편향값(b) 업데이트

    lst_i.append(i)
    lst_y.append(pred_y[0])
    lst_e.append(error[0])
    lst_b.append(b[0])
  
    if i%100 == 0 :
        print(f"학습 횟수n: {i:>5}  ", end=' ')
        print(f"가중치w:{round(float(w[0]), 5):{' '}>8}  ", end=' ')    
        print(f"실제y: {y}  ", end=' ')
        print(f"예측y:{round(float(pred_y[0]), 5):{' '}>8}  ", end=' ')
        print(f"편향b:{round(float(b[0]), 5):{' '}>8}  ", end=' ')
        print(f"오차e:{round(float(error[0] ), 5):{' '}>8}  ")
     
# 그래프 출력
plt.figure(figsize=(3,3))
plt.plot(lst_i, lst_y, 'r--', label='예측값y 변화')
plt.plot(lst_i, lst_e, label='오차값e 변화')
plt.plot(lst_i, lst_b, label='편향값b 변화')
plt.xlabel('학습 횟수')
plt.legend()
plt.show()
```
```
학습 횟수n:     0   가중치w:-0.42509   실제y: 1   예측y: 0.53777   편향b: 0.19759   오차e: 0.46223  
학습 횟수n:   100   가중치w:-0.42509   실제y: 1   예측y: 0.90173   편향b: 2.22641   오차e: 0.09827  
학습 횟수n:   200   가중치w:-0.42509   실제y: 1   예측y: 0.94874   편향b: 2.92328   오차e: 0.05126  
학습 횟수n:   300   가중치w:-0.42509   실제y: 1   예측y: 0.96565   편향b: 3.33953   오차e: 0.03435  
학습 횟수n:   400   가중치w:-0.42509   실제y: 1   예측y: 0.97424   편향b: 3.63559   오차e: 0.02576  
학습 횟수n:   500   가중치w:-0.42509   실제y: 1   예측y: 0.97943   편향b: 3.86505   오차e: 0.02057  
학습 횟수n:   600   가중치w:-0.42509   실제y: 1   예측y: 0.98289   편향b: 4.05227   오차e: 0.01711  
학습 횟수n:   700   가중치w:-0.42509   실제y: 1   예측y: 0.98535   편향b: 4.21031   오차e: 0.01465  
학습 횟수n:   800   가중치w:-0.42509   실제y: 1   예측y:  0.9872   편향b: 4.34702   오차e:  0.0128  
학습 횟수n:   900   가중치w:-0.42509   실제y: 1   예측y: 0.98864   편향b: 4.46744   오차e: 0.01136  
학습 횟수n:  1000   가중치w:-0.42509   실제y: 1   예측y: 0.98979   편향b: 4.57503   오차e: 0.01021  
학습 횟수n:  1100   가중치w:-0.42509   실제y: 1   예측y: 0.99073   편향b: 4.67226   오차e: 0.00927  
학습 횟수n:  1200   가중치w:-0.42509   실제y: 1   예측y: 0.99151   편향b: 4.76093   오차e: 0.00849  
학습 횟수n:  1300   가중치w:-0.42509   실제y: 1   예측y: 0.99217   편향b: 4.84243   오차e: 0.00783  
학습 횟수n:  1400   가중치w:-0.42509   실제y: 1   예측y: 0.99273   편향b: 4.91783   오차e: 0.00727  
학습 횟수n:  1500   가중치w:-0.42509   실제y: 1   예측y: 0.99322   편향b: 4.98797   오차e: 0.00678  
학습 횟수n:  1600   가중치w:-0.42509   실제y: 1   예측y: 0.99365   편향b: 5.05354   오차e: 0.00635  
학습 횟수n:  1700   가중치w:-0.42509   실제y: 1   예측y: 0.99403   편향b:  5.1151   오차e: 0.00597  
학습 횟수n:  1800   가중치w:-0.42509   실제y: 1   예측y: 0.99436   편향b: 5.17311   오차e: 0.00564  
학습 횟수n:  1900   가중치w:-0.42509   실제y: 1   예측y: 0.99466   편향b: 5.22795   오차e: 0.00534  
학습 횟수n:  2000   가중치w:-0.42509   실제y: 1   예측y: 0.99493   편향b: 5.27996   오차e: 0.00507  
학습 횟수n:  2100   가중치w:-0.42509   실제y: 1   예측y: 0.99517   편향b:  5.3294   오차e: 0.00483  
학습 횟수n:  2200   가중치w:-0.42509   실제y: 1   예측y:  0.9954   편향b: 5.37652   오차e:  0.0046  
학습 횟수n:  2300   가중치w:-0.42509   실제y: 1   예측y:  0.9956   편향b: 5.42154   오차e:  0.0044  
학습 횟수n:  2400   가중치w:-0.42509   실제y: 1   예측y: 0.99578   편향b: 5.46462   오차e: 0.00422  
학습 횟수n:  2500   가중치w:-0.42509   실제y: 1   예측y: 0.99595   편향b: 5.50593   오차e: 0.00405  
학습 횟수n:  2600   가중치w:-0.42509   실제y: 1   예측y: 0.99611   편향b:  5.5456   오차e: 0.00389  
학습 횟수n:  2700   가중치w:-0.42509   실제y: 1   예측y: 0.99625   편향b: 5.58377   오차e: 0.00375  
학습 횟수n:  2800   가중치w:-0.42509   실제y: 1   예측y: 0.99639   편향b: 5.62054   오차e: 0.00361  
학습 횟수n:  2900   가중치w:-0.42509   실제y: 1   예측y: 0.99651   편향b:   5.656   오차e: 0.00349  
```
![image](https://user-images.githubusercontent.com/115389450/233282400-d57f7d9c-e979-4ad4-b32b-18b0bbd9bbdc.png)
