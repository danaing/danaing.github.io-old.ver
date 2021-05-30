---
layout: post
title:  "[TensorFlow] Introduction to TensorFlow (DeepLearning.AI)"
date:   2021-05-17
author: danahkim
tags: TensorFlow
categories: DeepLearning
---

## 0. 들어가며

TensorFlow Developer Certificate를 취득하기 위해 수강한 Coursera의 [DeepLearning.AI TensorFlow 개발자 전문 자격증 강의](https://www.coursera.org/professional-certificates/tensorflow-in-practice)의 [**Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning**](https://www.coursera.org/learn/introduction-tensorflow/) 강좌를 수강하였습니다.

![](/assets/images/2021-05-17-coursera--67ad01c3.png)

본 강좌는 2명의 강사가 나옵니다. Google에서 AI Advocacy를 이끌고 있는 **Laurence Moroney** 가 직접 DeepLearning과 TensorFlow에 대해 강의하시고, Stanford 대학의 유명 교수인 **Andrew Ng** 교수님이 담화에 나오십니다.

TensorFlow에 대한 많은 책과 강의가 있지만, 이 강좌를 수강한 이유는 구글의 저명한 과학자가 '직접' 설명하는 Deeplearning과 TensorFlow는 어떤걸지 궁금했기 때문입니다. 집에서 편히 앉아서 구글 과학자의 강의를 들을 수 있다니 얼마나 큰 혁명인가요?

Class material과 제가 푼 exercise는 제 [Github 링크](https://github.com/danaing/Coursera-TensorFlow/)에 정리해두었습니다.

-----------
## 1. A New Programming Paradigm

![](/assets/images/2021-05-17-coursera--10e0800b.png)

> "We built a super simple neural network that fit data like an x and y data onto a line but that was just **"Hello, World"**. Right, Andrew? So fitting straight lines seems like the "Hello, world" most basic implementation learning algorithm."

먼저 머신러닝와 딥러닝에 대한 개요를 설명합니다. 컴퓨터 언어를 배우기 시작할 때 으레 그 **세계 입문 의식**으로 `Hello, World!`를 먼저 프린트하곤 합니다. 마찬가지로 머신러닝의 세계에 입문할 때는 x와 y의 simple linear regression을 먼저 fitting하는 것이 'Hello, World!'와 같다는 담화가 인상깊었습니다.

아래와 같은 $x$와 $y$의 1차 선형 관계가 있을 때, Neural Net 1개에 fitting하여 Simple Linear Regression문제를 해결해보겠습니다.

$$
y = 0.5x + 0.5
$$

TensorFlow는 Keras의 Sequential을 사용하여 Neural Networks Model을 한줄로 간단히 구현할 수 있습니다. *(저는 이게 이 주차의 이름인 '새로운 프로그래밍 패러다임'을 뜻하는 것 같습니다.)* Optimizer와 loss, epoch을 지정하고 주어진 데이터셋에 fitting합니다.

```python
import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)
model.fit(xs, ys, epochs=1000)
print(model.predict([7.0]))
```

![](/assets/images/2021-05-17-coursera--05d65183.png)

epoch 1000으로 모형 학습이 끝났습니다. $ x=7 $ 을 predict한 결과 $ y=4 $가 아니라 $ 4.0027223 $으로 $ 4 $에 매우 근접한 숫자가 나옵니다. 이는 모형이 확률적으로 접근하고 있기 때문에 자연스러운 현상입니다.

-----------
## 2. Introduction to Computer Vision

Neural을 깊게 쌓는 Deap Neural Network를 사용한 image classification은 Computer Vision의 입문입니다.

흑백 이미지의 **fashion mnist** 데이터셋으로 예를 들겠습니다. 흑백 색상은 0과 255 사이의 pixel 값을 가지고 있습니다. 그러나 다양한 이유로 모든 값이 0과 1 사이의 값을 가질 때 다루기 쉬우므로 0과 1 사이의 값을 가지도록 Normalize합니다.

* **Sequential**: neural network에서 layer의 순서를 정의합니다.
* **Flatten**: 이미지는 사각형이기 때문에 Flatten을 사용하여 1-dimensional-set으로 바꾸어 줍니다. inpute_shape을 이미지 크기에 맞게 잘 지정해주어야 합니다. 예를 들어, 28*28 행렬 이미지는 784 벡터가 됩니다.
* **Dense**: 뉴런층을 추가합니다.

각각의 layer는 activation fuction이 필요한데 'Relu'와 'Softmax' 함수를 option으로 지정할 수 있습니다.

특히 **Callback**을 추가하여 자신이 원하는 성능에 도달하면 학습을 멈추게 하는 방법이 유용합니다. 아래 코드에 myCallback이라는 class를 보면 99%의 Accuracy를 달성하면 학습이 멈추게 됩니다.

```Python
import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>=0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the images
training_images=training_images/255.0
test_images=test_images/255.0

model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
```

![](/assets/images/2021-05-17-coursera--4e5c667d.png)

Accuracy가 99%에 도달하여 학습이 중단된 모습을 확인할 수 있습니다.

-----------
## 3. Enhancing Vision with Convolutional Neural Networks

( → Course Materials는 [여기](https://github.com/lmoroney/dlaicourse/blob/master/Course%201%20-%20Part%206%20-%20Lesson%202%20-%20Notebook.ipynb)에서 확인하실 수 있습니다. )

위 DNN에서 레이어 층의 크기, 학습 epoch의 수가 Accuray에 영향을 미치는 것을 확인했습니다. 아래 모델을 살펴보겠습니다.

```Python
import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images / 255.0
test_images=test_images / 255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)

test_loss = model.evaluate(test_images, test_labels)
```

![](/assets/images/2021-05-17-coursera--436b07cf.png)

위 DNN 모델의 Test Accuracy는 약 87% 입니다.

성능을 어떻게 더 향상시킬 수 있을까요? 한 가지 방법은 **Convolutions**를 추가하는 것입니다.
[위키피디아](https://en.wikipedia.org/wiki/Kernel_(image_processing))에 이미지를 처리하는 Convolution matrix(also called as kernel or mask) 에 대한 정리가 잘 되어있으니 참고해주세요!

Convolutions를 추가한 Neural Network의 궁극적인 컨셉은 **구체적이고 뚜렷한 디테일에 집중하기 위해 이미지의 내용을 줄이는 것**입니다.

또한 kernel 안에서 가장 큰 값만 가져오는 **MaxPooling**을 사용하여 강조 효과를 볼 수 있습니다.
![](/assets/images/2021-05-17-coursera--057e5d56.png)
<center> <small> 출처: https://youtu.be/8oOgPUO-TBY </small> </center> <br/>

```Python
import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(training_images, training_labels, epochs=5)
test_loss = model.evaluate(test_images, test_labels)
```

![](/assets/images/2021-05-17-coursera--e4add9d1.png)

똑같이 5개의 epoch에서 CNN의 성능은 Test Accuracy가 약 90%로 성능이 향상되었습니다!

그렇다면 DNN과의 차이점은 무엇일까요?
먼저 CNN은 train이 더 느리다는 것입니다. 그러나 컨폴루션과 풀링이 효율성과 학습에 영향을 미치기 때문에 loss도 더 낮고, Accuracy도 더 높습니다. 한마디로 이미지 데이터에 CNN의 성능이 훨씬 좋습니다.


#### (참고) Convolutional Neural Networks by Andrew Ng
Covolutional Neural Networks 에 대해 더 자세히 알고 싶다면, 아래 링크의 Andrew Ng 교수님의 강의를 추천합니다.
Coursera에  Convolutional Neural Networks (Course 4 of the Deep Learning Specialization) 코스로도 있습니다.
* Youtube 링크: <https://bit.ly/2UGa7uH>

-----------
## 4. Using Real-world Images

간단하게 Convolution만 추가해서 모델 성능을 향상할 수 있었습니다. 보다시피 위의 fashion mnist를 분류한 모델은 나쁘지 않았습니다. 이러한 이미지 처리 방법은 복잡한 현실 이미지에도 적용할 수 있습니다.

그러나 CNN 방법의 단점은 이미지가 아주 uniform해야한다는 것입니다. fashion mnist는 모두 fashion이라는 하나의 주제에 관한 이미지이고, 모두 가운데에 위치하며 확대되어 있습니다. 또한 이미지 크기가 모두 28*28로 동일합니다.

*하지만 많은 현실 데이터는 그렇게 uniform하지 않습니다.* GoogleAPI에서 제공하는 말과 사람을 분류하는 이미지를 예시로 보겠습니다.

![](/assets/images/2021-05-17-coursera--56e1f522.png)
<center> <small> horse-or-human dataset </small> </center> <br/>

위 이미지를 보면 다른 색깔, 다른 생김새의 말과 다른 성별과 외모, 다양한 자세를 취한 여러 인물 사진이 있습니다. 심지어 말의 다리가 3개만 나온 것도 있고, 사람 다리가 중간까지만 나온 것도 있습니다. 이 이미지를 가지고 binary-classification을 해보겠습니다.

첫번째 할일은 바로 '쉽게' 어떤 이미지가 사람이고 말인지 labeling하는 것입니다. TensorFlow에 **ImageDataGenerator**라는 유용한 클래스가 있습니다. 우리가 분류 모형을 만들 때 class에 대한 labeling이 필요한데, 이미지에 대한 labeling을 subdirectory를 사용하여 쉽게 도와주는 것입니다.

![](/assets/images/2021-05-17-coursera--a6d86419.png)
<center> <small> 출처: https://youtu.be/0kYIZE8Gl90 </small> </center> <br/>

위의 directory 구조를 보면, Training 안에 Horses와 Humans라는 directory가 있고 그 안에 이미지가 있습니다. 이렇게 ImageDataGenerator는 directory 이름에 따라 자동으로 labeling을 해줍니다. 코드는 아래와 같습니다.

```Python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = '/tmp/horse-or-human/'
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(rescale=1/255)

# Flow training images in batches of 128 using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=128,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

validation_dir = '/tmp/validation-horse-or-human/'
# All images will be rescaled by 1./255
validation_datagen = ImageDataGenerator(rescale=1/255)

# Flow validation images in batches of 32 using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=32,
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
```

이제 위 generator를 사용해서 쉽게 labeling을 해주고 이미지 분류 모델을 만들어 보겠습니다.

```Python
model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 300x300 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('horses') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])
from tensorflow.keras.optimizers import RMSprop

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,
      validation_data = validation_generator,
      validation_steps=8)

```

옵티마이져로 Adam대신 RMSprop 를 사용하였는데, 이는 Learning Rate를 Gradient Descent방법에 따라 바꾸어주는 방법입니다. 더 알고 싶다면 아래 링크를 참고해주세요. (Gradient Descent in Practice II Learning Rate by Andrew Ng: <https://goo.gle/3bQvJgM> )

모델 학습이 끝나고, 직접 이미지를 업로드해서 Prediction 해볼 수 있습니다. 이 과정을 통해 어디서 inference가 잘못됐는지 보고 모델을 수정할 수 있습니다.

```Python
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():

  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(300, 300))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a horse")
```
![](/assets/images/2021-05-17-coursera-introduction-to-tensorflow-598cd377.png)
<center> <small> 6개의 이미지 Prediction 결과 (출처: https://youtu.be/0kYIZE8Gl90) </small> </center> <br/>

위 코드를 사용해서 말 사진 3개, 사람 사진 3개를 테스트 한 결과, 말 사진 3개는 맞게 분류했지만, 아래 사람 사진을 horse라고 잘못 분류했습니다!

![](/assets/images/2021-05-17-coursera-introduction-to-tensorflow-dcbaf740.png)
<center> <small> 오분류한 사람 사진 (출처: https://youtu.be/0kYIZE8Gl90) </small> </center> <br/>

아마 긴 금발 머리의 특징이 훈련이 안되어 있을 수 있습니다. 또한 다리가 나오지 않은 인물 사진을 predict했을 때도 말로 오분류했습니다. 이는 train 데이터셋에만 Over-Fitting된 문제입니다. 이럴 때는 어디가 잘못 훈련되었는지 inference를 통해 모델을 다시 구성해야합니다. 또한 Data augutation을 사용해서 더 많은 이미지를 생성하여 학습하면 Over-Fitting을 방지할 수 있으며 이는 다음 강의에서 다뤄볼 예정입니다.

-----------
## Course Certificate

![](/assets/images/2021-05-17-coursera-introduction-to-tensorflow-5c92122e.png)

이렇게 수료증을 발급받으면서 본 코스를 완강했습니다.👏 군더더기없는 현업 전문가(그것도 구글)의 강의로 개념을 정리하고 주어진 데이터셋으로 직접 실습 문제를 다룰 수 있습니다. DeepLearning.AI에서 제공하는 다른 강의도 수강할 예정입니다.
