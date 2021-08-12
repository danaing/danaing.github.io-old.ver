---
layout: post
title:  "TensorFlow Developer Certificate(텐서플로우 개발자 전문 자격증) 취득 후기"
date:   2021-06-30
author: danahkim
tags: TensorFlow
categories: ETC
---

![162873713695](https://user-images.githubusercontent.com/62828866/129131720-299a1329-c499-4834-aeec-13140d46db3b.png)

## 들어가며

안녕하세요! TensorFlow Developer Certificate(텐서플로우 개발자 자격증)를 이번 6월에 취득하여 간단한 자격증 소개와 후기를 남겨보려합니다. 



## TensorFlow Developer Certificate이란?
먼저 TensorFlow Developer Certificate(텐서플로우 개발자 자격증)이란 Google에서 직접 공인하는 자격증입니다. 

![image](https://user-images.githubusercontent.com/62828866/129132700-428a7860-c7bb-447f-a09b-2cd32276fe48.png)

[TensorFlow 인증 프로그램](https://www.tensorflow.org/certificate)에 대한 자세한 설명은 위 공식 사이트를 참고하시면 됩니다.

본 자격증 시험에서는 **TensorFlow 2.x**를 사용하여 모델을 빌딩하여 문제를 해결하는 능력을 테스트합니다. 분야는 이미지, 자연어 처리, 시계열으로 다양합니다. 또한 사전 훈련(pretrained) 모델 사용 및 feature 추출, batch loading 이해, image augmentation 등 다양한 능력을 테스트합니다.
자격증의 유효기간은 3년이며, 자격증을 취득하면 [인증 네트워크](https://developers.google.com/certification/directory/tensorflow)에 자신을 TensorFlow 개발자로 등록할 수 있습니다!

![image](https://user-images.githubusercontent.com/62828866/129133169-e4993042-1d0b-403a-8df6-c0a914a89d35.png)



## 시험 구성
시험은 TensorFlow2.x를 사용하여 5개의 모델을 구현해야 합니다. 각각의 문제는 아래 카테고리로 구성되어 있습니다.

* 문제 구성
  * Category 1: Basic / Simple model

  * Category 2: Model from learning datset

  * Category 3: Convolutional Neural Network with real-world image dataset

  * Category 4: NLP Text Classification with real-world text dataset

  * Category 5: Sequence Model with real-world numeric dataset

    ※ 각 카테고리당 90점 이상 합격

* 시험 환경: PyCharm IDE
* 시간: 최대 5시간
* 비용: 100 US 달러
* 합격 안내: 약 24시간 내

환경 세팅 및 응시 관련 자세한 내용은 [수험자 가이드북](https://www.tensorflow.org/extras/cert/TF_Certificate_Candidate_Handbook_ko.pdf)를 참고하시면 됩니다.



## 시험 준비
시험 설명이 잘 되어있어 준비에 특별히 어려운 사항은 없었습니다만 아무래도 주중에는 일을 병행하다보니 주말을 사용하여 1달 정도의 기간이 소요된 것 같습니다. 사실 마지막 1주일 정도 바짝 준비하였는데, 딥러닝에 대한 이해도와 환경에 따라 개인 별 시험 준비기간은 다를 것 같습니다.

시험 준비 과정은 구글이 잘 마련해 놓았습니다. 본 자격증 취득을 위해 Coursera에서 무료로 수강할 수 있는 [DeepLearning.AI TensorFlow 개발자 전문 인증 과정](https://www.coursera.org/professional-certificates/tensorflow-in-practice) 강의가 있습니다. 2명의 강사가 나오는데 Google에서 AI Advocacy를 이끌고 있는 Laurence Moroney가 직접 DeepLearning과 TensorFlow에 대해 강의하시고, Stanford 대학의 유명 교수인 Andrew Ng 교수님이 담화에 나오십니다. 강의 자료와 설명도 매우 좋습니다. 1개의 강의 당 1주일동안 무료로 수강이 가능하니 모두 수강할 시에 4주가 걸립니다. 이 강의에서 시험에 관련된 모든 내용을 거의 커버하고 또한 강의에서 나온 문제가 시험에 그대로 나오니 딥러닝 기초가 없으신 분들에게는 강의 수강을 적극 권장드립니다.

만약 강의가 필요 없다면 위 강의의 Laurence Moroney의 [깃헙](https://github.com/lmoroney/dlaicourse) 활용하시면 데이터셋과 예제 그리고 과제의 답을 볼 수 있습니다. 저의 경우는 Coursera 강의 중 필요 부분을 들으면서 깃헙에서 예제를 구해 스스로 모델을 만들어 성능을 개선하는 방법으로 공부하였습니다.



## 마치며

이번 자격증 취득을 통해서 TensorFlow 2.x에 대한 이해도를 높일 수 있었습니다. 사실 그보다 무언가 해냈다는 **자신감**과 자격증이 주는 **성취감**이 꾸준히 공부를 하는 긍정적 영향으로 동기부여가 되지 않을까 싶네요. 온라인 자격증 취득이 처음이다 보니 환경세팅에 애즐거운 경험이었습니다.

소박하지만 이번 자격증 준비를 통해 느낀 점 입니다.

- Colab Pro에 무한한 감사를... 

  GPU가 없다면 ~~정신건강을 위해~~ Colab Pro사용은 선택이 아닌 필수입니다

- Callback 도 필수!

- Dense 뿐만 아니라 Augmentation 각각의 parameter가 성능에 더 예민하다.

  어떤 데이터로 train하느냐가 역시 더 중요한 것 같습니다.

- 역시 Pretrained 모델이 성능이 막강하다는 것을 다시금 느꼈습니다.

  제 경우에는 Google에서 만든 VGG Net을 사용하였고, 기존 CNN 모델에 비해 넘사 성능 개선을 하여 문제를 통과하였습니다.



![image](https://user-images.githubusercontent.com/62828866/129131796-8540df67-2f7d-4701-9710-94a911bc5bc6.png)

[TensorFlow Developer Certificate - Kim Dan Ah](https://www.credential.net/75d16cdd-ee28-4bc3-9222-59f0f87ee479)

