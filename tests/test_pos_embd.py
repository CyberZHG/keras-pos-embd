import unittest
import os
import tempfile
import random
import numpy as np
from keras_pos_embd.backend import keras
from keras_pos_embd import PositionEmbedding


class TestPosEmbd(unittest.TestCase):

    def test_index(self):
        indices = np.asarray([[-4, 10]])
        weights = np.random.random((21, 2))
        weights[6, :] = np.asarray([0.25, 0.1])
        weights[20, :] = np.asarray([0.6, -0.2])
        model = keras.models.Sequential()
        model.add(PositionEmbedding(
            input_dim=10,
            output_dim=2,
            mode=PositionEmbedding.MODE_EXPAND,
            input_shape=(None,),
            weights=[weights],
            name='Pos-Embd',
        ))
        model.compile('adam', 'mse')
        model_path = os.path.join(tempfile.gettempdir(), 'test_pos_embd_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'PositionEmbedding': PositionEmbedding})
        model.summary()
        predicts = model.predict(indices)
        expected = np.asarray([[
            [0.25, 0.1],
            [0.6, -0.2],
        ]])
        self.assertTrue(np.allclose(expected, predicts))

    def test_mask_zero(self):
        indices = np.asarray([[-4, 10, 100]])
        weights = np.random.random((21, 2))
        weights[6, :] = np.asarray([0.25, 0.1])
        weights[20, :] = np.asarray([0.6, -0.2])
        model = keras.models.Sequential()
        model.add(PositionEmbedding(
            input_dim=10,
            output_dim=2,
            mode=PositionEmbedding.MODE_EXPAND,
            mask_zero=100,
            input_shape=(None,),
            weights=[weights],
            name='Pos-Embd',
        ))
        model.build()
        model.compile('adam', 'mse')
        model_path = os.path.join(tempfile.gettempdir(), 'keras_pos_embd_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'PositionEmbedding': PositionEmbedding})
        model.summary()
        predicts = model.predict(indices)
        expected = np.asarray([[
            [0.25, 0.1],
            [0.6, -0.2],
            [0.6, -0.2],
        ]])
        self.assertTrue(np.allclose(expected, predicts))

    def test_add(self):
        inputs = np.ones((1, 5, 2))
        weights = np.random.random((10, 2))
        weights[1, :] = np.asarray([0.25, 0.1])
        weights[3, :] = np.asarray([0.6, -0.2])
        model = keras.models.Sequential()
        model.add(PositionEmbedding(
            input_dim=10,
            output_dim=2,
            mode=PositionEmbedding.MODE_ADD,
            input_shape=(None, 2),
            weights=[weights],
            name='Pos-Embd',
        ))
        model.compile('adam', 'mse')
        model_path = os.path.join(tempfile.gettempdir(), 'test_pos_embd_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'PositionEmbedding': PositionEmbedding})
        model.summary()
        predicts = model.predict(inputs)
        self.assertTrue(np.allclose([1.25, 1.1], predicts[0][1]), predicts[0])
        self.assertTrue(np.allclose([1.6, 0.8], predicts[0][3]), predicts[0])

    def test_concat(self):
        inputs = np.ones((1, 5, 2))
        weights = np.random.random((10, 2))
        weights[1, :] = np.asarray([0.25, 0.1])
        weights[3, :] = np.asarray([0.6, -0.2])
        model = keras.models.Sequential()
        model.add(PositionEmbedding(
            input_dim=10,
            output_dim=2,
            mode=PositionEmbedding.MODE_CONCAT,
            input_shape=(None, 2),
            weights=[weights],
            name='Pos-Embd',
        ))
        model.compile('adam', 'mse')
        model_path = os.path.join(tempfile.gettempdir(), 'test_pos_embd_%f.h5' % random.random())
        model.save(model_path)
        model = keras.models.load_model(model_path, custom_objects={'PositionEmbedding': PositionEmbedding})
        model.summary()
        predicts = model.predict(inputs)
        self.assertTrue(np.allclose([1.0, 1.0, 0.25, 0.1], predicts[0][1]), predicts[0])
        self.assertTrue(np.allclose([1.0, 1.0, 0.6, -0.2], predicts[0][3]), predicts[0])
