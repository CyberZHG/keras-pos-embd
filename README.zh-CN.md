# Keras Position Embedding

[![Travis](https://travis-ci.org/CyberZHG/keras-pos-embd.svg)](https://travis-ci.org/CyberZHG/keras-pos-embd)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-pos-embd/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-pos-embd)
[![Version](https://img.shields.io/pypi/v/keras-pos-embd.svg)](https://pypi.org/project/keras-pos-embd/)

\[[中文](https://github.com/CyberZHG/keras-pos-embd/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-pos-embd/blob/master/README.md)\]

位置嵌入层。

## 安装

```bash
pip install keras-pos-embd
```

## 使用

### 可训练位置嵌入

基本使用方法和嵌入层一致，模式使用`PositionEmbedding.MODE_EXPAND`：

```python
import keras
from keras_pos_embd import PositionEmbedding

model = keras.models.Sequential()
model.add(PositionEmbedding(
    input_shape=(None,),
    input_dim=10,     # 最大的位置的绝对值
    output_dim=2,     # 嵌入的维度
    mask_zero=10000,  # 作为padding的位置下标（因为0被占用了）
    mode=PositionEmbedding.MODE_EXPAND,
))
model.compile('adam', 'mse')
model.summary()
```

如果跟在嵌入层使用，则不需要设置`mask_zero`。嵌入特征与位置嵌入相加使用`PositionEmbedding.MODE_ADD`模式，相连使用`PositionEmbedding.MODE_CONCAT`模式：

```python
import keras
from keras_pos_embd import PositionEmbedding

model = keras.models.Sequential()
model.add(keras.layers.Embedding(
    input_shape=(None,),
    input_dim=10,
    output_dim=5,
    mask_zero=True,
))
model.add(PositionEmbedding(
    input_dim=100,
    output_dim=5,
    mode=PositionEmbedding.MODE_ADD,
))
model.compile('adam', 'mse')
model.summary()
```

### 三角函数嵌入

[三角函数嵌入](https://arxiv.org/pdf/1706.03762)没有可训练权重，使用方法和`PositionEmbedding`相同，不需要输入的维度：

```python
import keras
from keras_pos_embd import TrigPosEmbedding

model = keras.models.Sequential()
model.add(TrigPosEmbedding(
    input_shape=(None,),
    output_dim=30,
    mode=TrigPosEmbedding.MODE_EXPAND,
))
model.compile('adam', 'mse')
model.summary()
```

相加模式：

```python
import keras
from keras_pos_embd import TrigPosEmbedding

model = keras.models.Sequential()
model.add(keras.layers.Embedding(
    input_shape=(None,),
    input_dim=10,
    output_dim=5,
    mask_zero=True,
))
model.add(TrigPosEmbedding(
    output_dim=5,
    mode=TrigPosEmbedding.MODE_ADD,
))
model.compile('adam', 'mse')
model.summary()
```
