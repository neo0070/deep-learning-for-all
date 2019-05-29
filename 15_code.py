#-*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split

import numpy
import pandas as pd
import tensorflow as tf

'''
## 선형회귀(linear regression)
+ 변수 사이에 선형적인 관계를 모델링 한 것(ex - 키/나이, 노동시간/수입)
 - 많은 현상들은 선형적인 성격을 가짐
+ 회귀(regression) 
<> 
분류(Classification) : 학습된 클레스로 분류해서 응답

+ 직선에 멀어짐을 에러로 표현하며 square error로 표현
 - 평균은 Mean Square Error이며 Least Mean Square Erro를 통해 가장 근접한 직선을 구함
 - Cost Function(= Mean Square Error) : 비용함수, 실제값과 가설간의 차이, 가설이 얼마나 정확한 지 판단하는 기준
 - 목표는 Cost Function을 최저값으로 만드는 것 = 목표함수(Object Function)
+ h(x) = @x, 목표는 (세타)@값=기울기를 찾는 것
 - @ := @ - a(Mean Square Error)
 - Learning rate(a)와 세타(@)값을 조합하여 최적의 세타값을 찾아야 함 = converge 되다.
 - Learning rate 너무 작으면 오래걸리고 크면 최저점으로 수렴하지 않을 수 있음..
  . 곡선의 특성 상 초반에는 많은 폭으로 변화
  . 2차 함수에서 직선 기울기 따라다니기(미분 : 2차함수에서 기울기 구하기) > 경사 하강법
'''

# seed 값 설정
seed = 0
numpy.random.seed(seed)
tf.set_random_seed(seed)

# 엑셀 파일에서 데이터 불러오기
df = pd.read_csv("../dataset/housing.csv", delim_whitespace=True, header=None)


print(df.info())
print(df.head())


dataset = df.values
X = dataset[:,0:13] # class 'numpy.ndarray', slice notation, 0~12컬럼
Y = dataset[:,13]   # class 'numpy.ndarray', slice notation, 13 컬럼

# train_test_split : 전체 데이터셋 배열을 받아서 랜덤하게 훈련/테스트 데이터 셋으로 분리
# output이 array 4EA 속한 리스트
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=seed)

# 모델의 설정
model = Sequential()
model.add(Dense(30, input_dim=13, activation='relu')) ## 층 추가
model.add(Dense(6, activation='relu'))
model.add(Dense(1))

# 모텔 컴파일
model.compile(loss='mean_squared_error',
              optimizer='adam') # adam : Momentum과 AdaGrad를 섞은 기법
# 모델 실행
model.fit(X_train, Y_train, epochs=20, batch_size=10)
# epochs : 학습 반복 횟수
# batch_size : 몇 개의 샘플로 가중치를 갱신할 것인지 지정

# 예측(prediction) 값과 실제 값의 비교
Y_prediction = model.predict(X_test).flatten()
for i in range(10):
    label = Y_test[i]
    prediction = Y_prediction[i]
    print("실제가격: {:.3f}, 예상가격: {:.3f}".format(label, prediction))
