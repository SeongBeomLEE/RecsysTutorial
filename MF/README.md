# (2009, IEEE) Matrix Factorization Techniques for Recommender Systems

# 01. Introduction
- 소비자와 가장 적합한 제품을 매칭하는 것은 사용자 만족도와 충성도를 높이는 데 있어 핵심이다.
- 높은 사용자 만족도와 충성도는 회사의 이윤과도 깊게 연관이 있다.
- 그러면 소비자와 가장 적합한 제품을 매칭하는 것, 즉 제품 추천을 어떻게 할 것인가?
- 고객들은 특정 상품에 대한 만족도를 기꺼이 표시하기 때문에, 이러한 방대한 양의 데이터를 활용하면 특정 고객에 맞는 맞춤화된 상품을 추천할 수 있다.
- 많은 고객들은 같은 영화를 보지만, 각각의 고객들은 수많은 다양한 영화를 보기 때문에(개인적으로 이 문장이 추천에사 제일 고려되어야 하는 문장이라고 생각함), 각각의 고객의 데이터를 활용하면 맞춤화된 추천이 가능하다.

# 02. Recommender System Strategies
추천 시스템의 일반적인 전략은 크게 Profile Data를 사용하는 Content Filtering 방식과 User behavior Data를 사용하는 Collaborative Filtering 방식으로 나뉘어짐

## 1) Content Filtering 방법
- product profile (상품 속성을 의미, ex.장르,감독 등)와 user profile (유저 속성을 의미, ex.성별,나이 등) Data를 사용하여 상품을 추천해주는 방식
- 각 상품에 대한 속성을 정보를 기반으로 추천을 해주기 때문에 cold start problem이 존재하지 않음
- product profile or user profile Data를 가지고 Embedding을 생성하여 유사한 product or 유사한 user가 사용한 product를 추천해주는 것

## 2) Collaborative Filtering 방법
- 과거의 user behavior Data를 사용하여 상품을 추천해주는 방식
- 사용자와 제품 간의 상호 작용을 분석하여 사용자에게 새로운 항목을 연결하는 방식임
- 단순히 user behavior Data를 사용하기 때문에, domain free와 속성 정보를 얻기 어려운 Task에서 쉽게 사용할 수 있다는 장점이 존재함
- 일반적으로 Content Filtering 방법 보다 성능이 우수하다고 함
- 그러나 새로운 아이템 or 새로운 유저 등이 발생하면 user behavior Data에 존재하지 않아 추천을 해주지 못하는 cold start problem이 발생함
- 본 방법은 크게 Neighborhood Method 와 Latent Factor Model로 나뉨

### (1) Neighborhood Method

![](https://images.velog.io/images/2712qwer/post/1c51b71c-2a51-4efc-a734-0abd9f63700c/image.png)

- Neighborhood Method는 항목 간 또는 사용자 간 관계를 계산하는 데 중점을 둔 방식
- 위 처럼 하나의 유저가 좋은 평가를 내린 3개의 영화를 선택하면, 이 3개의 영화를 모두 본 유사한 user를 찾고, 유사한 유저들이 또 좋게 선호한 영화를 찾고, 찾은 영화 속에서 서로 겹치는 영화를 먼저 추천해주는 방식임 (유사한 유저가 좋아한 영화는 해당 유저도 좋아할 것 이라는 느낌) 


### (2) Latent Factor Model

![](https://images.velog.io/images/2712qwer/post/785d300c-0c33-4180-aa20-d6b51df6be51/image.png)

- Latent Factor Model은 임의의 User와 Item Latent Factor를 설정한 후 User의 Rating Matrix(꼭 평점일 필요는 없음)속 잠재된 패턴을 추론하여 Latent Factor 즉 잠재 요인을 학습하는 방식
- User의 Rating Matrix는 곧 User와 Item의 상호 작용이 반영된 데이터라고 볼 수 있음
- 따라서 추론된 Latent Factor는 다양한 User와 Item의 상호 작용이 반영된 dimension으로 정의된다고 볼 수 있음 (dimension에는 좋아하는 컬러, 좋아하는 감독 등의 정보가 반영되어 있음)
- 각 차원이 무엇을 의미하는 지는 확실하게 모르지만 User와 Item의 상호 작용으로 만들어진 데이터로 추론했기 때문에 만들어진 차원에는 상호 작용의 패턴이 반영되어 있어 의미있는 차원을 가질 것이라는 느낌임 (ML과 DL의 기본이 결국 데이터의 패턴을 바탕으로 학습하는 것이기 때문에, 개인적으로 패턴이 반영된 Latent Factor는 중요한 정보를 반영할 수 있다고 생각함)

# 03. Matrix Factorization Methods
- Matrix Factorization Methods는 Latent Factor Model의 가장 기본이라고 할 수 있음
- Matrix Factorization Methods는 User와 Item 간의 평점 패턴에서 추론된 잠재 요인으로 User와 Item을 특징짓는다. 즉 추론된 User와 Item의 잠재 요인 간의 높은 상관성을 바탕으로 추천이 이루지는 것
- Matrix Factorization Methods는 good scalability, 높은 예측 정확성, 실생활을 모델링하는데 높은 flexibility를 제공해준다.
- Matrix Factorization Methods는 explicit feedback 데이터인 평점 행렬를 활용할 수 있는데, 이러한 explicit feedback는 매우 sparse matrix 이기 때문에 특정 상품에 편향된 모델이 만들어질 수 있다.
- Matrix Factorization Methods는 높은 유연성을 가진 모델이기 때문에 implicit feedback 데이터인 구매 이력, 검색 이력 등을 활용해서도 모델링 할 수 있다. implicit feedback은 explicit feedback과는 다르게 dense matrix일 확률이 높다. (자신에게 주어진 데이터에 맞게 Matrix Factorization Methods를 활용할 수 있다는 것)

# 04. A Basic Matrix Factorization Model

## (1) SVD

![](https://images.velog.io/images/2712qwer/post/e4bbfd03-bf15-4fb4-932c-7dbd4e79b8b7/image.png)

- Matrix Factorization Model은 singular value decomposition (SVD)와 큰 연관이 있는 모델임
- SVD 기반의 기본적인 Matrix Factorization Model은 위 공식을 바탕으로 user와 item의 Latent Factor의 내적으로 행렬을 근사하는 것임
- user와 item의 Latent Factor을 같은 차원에 위치시키는 것이 목표이고, 이 Latent Factor을 바탕으로 추천이 이루어짐
- 전통적인 SVD는 결측치가 존재하는 행렬에 대해서는 정의되지 않음, 그런데 추천시스템에서 우리가 복원하고하는 대부분의 행렬(평점 데이터 등)은 결측치가 매우 많이 존재함
- 또한 이러한 결측치가 매우 많이 존재하는 데이터는 비교적 적은 수의 알려진 항목만 다루게 되기 때문에 쉽게 과적합이 발생할 수 있음
- 그래서 전통적인 SVD는 활용하기 어려움

## (2) SVD 기반의 변형된 Matrix Factorization Model 

![](https://images.velog.io/images/2712qwer/post/048c69e5-5075-4768-b5d8-ce1354b0d5cf/image.png)

- 결측치가 존재하는 행렬에 대해서도 사용할 수 있게 위와 같은 방식의 SVD 기반의 변형된 Matrix Factorization Model을 사용함(본 공식을 기반으로 조금씩 모델이 발전되는 형태로 논문이 진행됨)
- User와 Item의 존재하는 쌍, 존재하는 평점에 대해서만 모델을 학습시킴
- 과거의 존재하는 데이터만을 사용하여 미래에 알려지지 않는 데이터를 예측하는 것이 모델의 목표이기 때문에 모델의 일반화는 필수임
- 모델의 일반화를 위해서 상수 람다(규제 파라미터)를 활용하여 모델에 규제를 가함(상수 람다는 교차 검증을 통해서 적절한 값을 찾는다고 함)

# 05. Learning Algorithms
04 - (2) 에 주어진 공식을 최적화 하기 위해서 본 논문은 학습 알고리즘으로 Stochastic Gradient Descent(SGD)와 Alternating Least Squares(ALS) 방식을 제안함 

## (1) Stochastic Gradient Descent

![](https://images.velog.io/images/2712qwer/post/cbaf408b-9f3c-4679-828b-f6278d875d26/image.png)

- SGD는 먼전 왼쪽의 식을 바탕으로 Error를 계산한 후, 계산된 Error를 바탕으로 기울기를 구한 후 오른쪽의 식을 바탕으로 User와 Item의 Latent Factor를 동시에 최적화 한다.
- 기울기를 활용하는 방법과 User와 Item의 Latent Factor를 동시에 최적화하기 때문에, 구현이 쉽고 속도가 ALS 방식보다 빠르다.

## (2) Alternating Least Squares
- 04 - (2) 에 주어진 공식의 경우 not convex 할 것인데, 이러한 볼록하지 않은 목적함수를 조금더 잘 학습시킬 수 있는 방식이 ALS이다.
- ALS는 학습시에 User 또는 Item의 Latent Factor 중 하나를 고정시킨 후에 고정시키지 않은 Latent Factor 먼저 최적화시키는 방식으로, 이 방식을 최적화가 될 때까지 두 Latent Factor를 고정과 비고정을 계속해가며 반복하는 것이다.
- ALS의 이러한 학습 방식은 SGD 보다 학습 속도는 느릴지라도, 알고리즘의 대규모 병렬화와 implicit data에 대해서는 효율적인 방식이다. (그리고 개인적인 경험상 1개를 먼저 고정시키면서 학습 하는 방식이 성능이 더 좋았음)

# 06. Adding Biases

![](https://images.velog.io/images/2712qwer/post/55783c71-e011-4101-91b2-5d924dc9811f/image.png)

- 단순히 User와 Item의 Latent Factor의 내적으로만 rating 행렬을 추정하는 것은 매우 현명하지 못한 방법이다.
- 예를들어 사람들마다 영화에 대해 주는 평점의 경향이 다를 것이고, 영화마다도 평점의 경향이 다를 것이다. 이에 biase라는 개념으로 이러한 평점의 경향을 모델링 했다.
- Titanic이라는 영화의 경우 전체 영화의 평점보다 높은 평점을 받는 경향을 가지고 있고, 어떤 유저가 낮은 평점을 주는 경향이 있다고 할때, 타이타닉 영화의 평점 3.9 은 3.7 (전체 영화 평점) + 0.5 (아이템 편향) - 0.3 (유저 편향) + User와 Item의 Latent Factor의 내적 으로 표현될 수 있다.
- 편향이 포함된 위와 같은 공식을 활용해 조금더 rating 행렬 정교하게 추정할 수 있게 된다.

# 07. Additional Input Sources

![](https://images.velog.io/images/2712qwer/post/b6de32d6-02cc-427c-8112-46a6d7e3472b/image.png)

- 사용자의 클릭 유뮤, 행동 이력 등의 implicit feedback 데이터를 추가적으로 활용하면 사용자들이 매우 적은 평점을 제공하여 발생하는 cold start problem을 다룰 수 있다.
- 위와 같은 공식처럼 아이템에 선호도를 보였던 사용자들의 벡터을 활용해 최적화를 진행함으로써 매우 적은 데이터가 존재했던 아이템 또는 유저들의 문제에도 일반화된 모델을 얻을 수 있게 된다. (본 공식은 유저에 대해서만 다뤄져 있지만, 이를 반대로 생각하면 아이템에 대해서도 다룰 수 있다.)

# 08. Temploral Dynamics
![](https://images.velog.io/images/2712qwer/post/77b69b94-322e-451c-a726-5d6edc78e560/image.png)

- 시간이 지남에 따라서 영화에 대한 평점이 달라질 수 있고, 사용자가 선호허는 장르 또한 달라질 수 있다. 
- 따라서 현실에서는 이러한 시간적 역동성이 반영된 모델을 만드는 것은 중요하다.
- 우리가 앞에서 다룬 모델들은 정적인 모델이었기 때문에, 위의 공식 처럼 시간에 대한 파라미터를 추가하면 조금더 시간의 역동성을 반영할 수 있는 모델을 만들 수 있게 된다.


# 09. Inputs With Varying Confidence Levels

![](https://images.velog.io/images/2712qwer/post/ed1d2f8c-e505-4794-8a01-6ec7b3273a20/image.png)

- 모든 관측 데이터가 동일한 가중치와 신뢰도를 가지는 것은 아니다.
- 예를 들어 암묵적 피드백을 중심(0, 1)으로 구축된 시스템에서 선호에 대한 신뢰도를 파악하는 것이 중요한데, 여기서 신뢰도는 프로그램을 시청한 시간 또는 특정 항목을 얼마나 자주 구입했는지와 같은 동작 빈도를 설명하는 사용 가능한 수치 값으로 표현될 수 있다. (반복된 행동이 더 비중있는 데이터일 수 있음)
- 따라서 위 공식에 c 처럼 신뢰도 점수를 모델 학습시에 고려하여, 의미 없는 관측치에 더 적은 가중치를 부여함으로써 조금더 일반화된 모델을 얻을 수 있다.

# 10. Netflix Prize Competition

![](https://images.velog.io/images/2712qwer/post/7d7e2430-6ccf-4b55-91ec-dcb6128bdac8/image.png)

- Netflix Prize Competition 에서는 Matrix Factorization 기반의 Model들이 좋은 성능을 보여줬다.
- 위의 사진처럼 Matrix Factorization Model들은 파라미터의 수가 증가할수록(Laten Factor), 더 다양한 방식을 추가할 수록 높은 성능을 보여줬다.

# Reference
- https://d2l.ai/chapter_recommender-systems/mf.html
- https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf
