# Keras Position Embedding

[![Travis](https://travis-ci.org/CyberZHG/keras-pos-embd.svg)](https://travis-ci.org/CyberZHG/keras-pos-embd)
[![Coverage](https://coveralls.io/repos/github/CyberZHG/keras-pos-embd/badge.svg?branch=master)](https://coveralls.io/github/CyberZHG/keras-pos-embd)
[![PyPI](https://img.shields.io/pypi/pyversions/keras-pos-embd.svg)](https://pypi.org/project/keras-pos-embd/)

Position embedding layers in Keras.

## Install

```bash
pip install keras-pos-embd
```

## Usage

```python
import keras
from keras_pos_embd import PositionEmbedding

model = keras.models.Sequential()
model.add(PositionEmbedding(
    input_dim=10,     # The maximum absolute value of positions.
    output_dim=2,     # The dimension of embeddings.
    mask_zero=10000,  # The index that presents padding (because `0` will be used in relative positioning).
    input_shape=(None,),
    name='Pos-Embd',
))
```

(Note that you don't need to enable `mask_zero` if you would concatenate other layers like word embeddings with masks)
