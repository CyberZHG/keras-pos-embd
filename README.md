# Keras Position Embedding

[![Travis](https://travis-ci.org/CyberZHG/keras-pos-embd.svg)](https://travis-ci.org/CyberZHG/keras-pos-embd)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-pos-embd/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-pos-embd)
[![Version](https://img.shields.io/pypi/v/keras-pos-embd.svg)](https://pypi.org/project/keras-pos-embd/)

\[[中文](https://github.com/CyberZHG/keras-pos-embd/blob/master/README.zh-CN.md)|[English](https://github.com/CyberZHG/keras-pos-embd/blob/master/README.md)\]

Position embedding layers in Keras.

## Install

```bash
pip install keras-pos-embd
```

## Usage

### Trainable Embedding

```python
import keras
from keras_pos_embd import PositionEmbedding

model = keras.models.Sequential()
model.add(PositionEmbedding(
    input_shape=(None,),
    input_dim=10,     # The maximum absolute value of positions.
    output_dim=2,     # The dimension of embeddings.
    mask_zero=10000,  # The index that presents padding (because `0` will be used in relative positioning).
    mode=PositionEmbedding.MODE_EXPAND,
))
model.compile('adam', 'mse')
model.summary()
```

Note that you don't need to enable `mask_zero` if you want to add/concatenate other layers like word embeddings with masks:

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

### Sin & Cos Embedding

The [sine and cosine embedding](https://arxiv.org/pdf/1706.03762) has no trainable weights. The layer has three modes, it works just like `PositionEmbedding` in `expand` mode:

```python
import keras
from keras_pos_embd import TrigPosEmbedding

model = keras.models.Sequential()
model.add(TrigPosEmbedding(
    input_shape=(None,),
    output_dim=30,                      # The dimension of embeddings.
    mode=TrigPosEmbedding.MODE_EXPAND,  # Use `expand` mode
))
model.compile('adam', 'mse')
model.summary()
```

If you want to add this embedding to existed embedding, then there is no need to add a position input in `add` mode:

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
