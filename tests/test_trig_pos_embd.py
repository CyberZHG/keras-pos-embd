import os
import tempfile
import unittest

import numpy as np

from keras_pos_embd.backend import keras
from keras_pos_embd import TrigPosEmbedding


class TestSinCosPosEmbd(unittest.TestCase):

    def test_invalid_output_dim(self):
        with self.assertRaises(NotImplementedError):
            TrigPosEmbedding(
                mode=TrigPosEmbedding.MODE_EXPAND,
                output_dim=5,
            )

    def test_missing_output_dim(self):
        with self.assertRaises(NotImplementedError):
            TrigPosEmbedding(
                mode=TrigPosEmbedding.MODE_EXPAND,
            )

    def test_brute(self):
        seq_len = np.random.randint(1, 10)
        embd_dim = np.random.randint(1, 20) * 2
        indices = np.expand_dims(np.arange(seq_len), 0)
        model = keras.models.Sequential()
        model.add(TrigPosEmbedding(
            input_shape=(seq_len,),
            mode=TrigPosEmbedding.MODE_EXPAND,
            output_dim=embd_dim,
            name='Pos-Embd',
        ))
        model.compile('adam', 'mse')
        model_path = os.path.join(tempfile.gettempdir(), 'test_trig_pos_embd_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'TrigPosEmbedding': TrigPosEmbedding})
        model.summary()
        predicts = model.predict(indices)[0].tolist()
        for i in range(seq_len):
            for j in range(embd_dim):
                actual = predicts[i][j]
                if j % 2 == 0:
                    expect = np.sin(i / 10000.0 ** (float(j) / embd_dim))
                else:
                    expect = np.cos(i / 10000.0 ** ((j - 1.0) / embd_dim))
                self.assertAlmostEqual(expect, actual, places=6, msg=(embd_dim, i, j, expect, actual))

    def test_add(self):
        seq_len = np.random.randint(1, 10)
        embed_dim = np.random.randint(1, 20) * 2
        inputs = np.ones((1, seq_len, embed_dim))
        model = keras.models.Sequential()
        model.add(TrigPosEmbedding(
            input_shape=(seq_len, embed_dim),
            mode=TrigPosEmbedding.MODE_ADD,
            name='Pos-Embd',
        ))
        model.compile('adam', 'mse')
        model_path = os.path.join(tempfile.gettempdir(), 'test_trig_pos_embd_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'TrigPosEmbedding': TrigPosEmbedding})
        model.summary()
        predicts = model.predict(inputs)[0].tolist()
        for i in range(seq_len):
            for j in range(embed_dim):
                actual = predicts[i][j]
                if j % 2 == 0:
                    expect = 1.0 + np.sin(i / 10000.0 ** (float(j) / embed_dim))
                else:
                    expect = 1.0 + np.cos(i / 10000.0 ** ((j - 1.0) / embed_dim))
                self.assertAlmostEqual(expect, actual, places=6, msg=(embed_dim, i, j, expect, actual))

    def test_concat(self):
        seq_len = np.random.randint(1, 10)
        feature_dim = np.random.randint(1, 20)
        embed_dim = np.random.randint(1, 20) * 2
        inputs = np.ones((1, seq_len, feature_dim))
        model = keras.models.Sequential()
        model.add(TrigPosEmbedding(
            input_shape=(seq_len, feature_dim),
            output_dim=embed_dim,
            mode=TrigPosEmbedding.MODE_CONCAT,
            name='Pos-Embd',
        ))
        model.compile('adam', 'mse')
        model_path = os.path.join(tempfile.gettempdir(), 'test_trig_pos_embd_%f.h5' % np.random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'TrigPosEmbedding': TrigPosEmbedding})
        model.summary()
        predicts = model.predict(inputs)[0].tolist()
        for i in range(seq_len):
            for j in range(embed_dim):
                actual = predicts[i][feature_dim + j]
                if j % 2 == 0:
                    expect = np.sin(i / 10000.0 ** (float(j) / embed_dim))
                else:
                    expect = np.cos(i / 10000.0 ** ((j - 1.0) / embed_dim))
                self.assertAlmostEqual(expect, actual, places=6, msg=(embed_dim, i, j, expect, actual))
