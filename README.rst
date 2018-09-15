
Keras Position Embedding
========================


.. image:: https://travis-ci.org/CyberZHG/keras-pos-embd.svg
   :target: https://travis-ci.org/CyberZHG/keras-pos-embd
   :alt: Travis


.. image:: https://coveralls.io/repos/github/CyberZHG/keras-pos-embd/badge.svg?branch=master
   :target: https://coveralls.io/github/CyberZHG/keras-pos-embd
   :alt: Coverage


.. image:: https://img.shields.io/pypi/pyversions/keras-pos-embd.svg
   :target: https://pypi.org/project/keras-pos-embd/
   :alt: PyPI


Position embedding layers in Keras.

Install
-------

.. code-block:: bash

   pip install keras-pos-embd

Usage
-----

.. code-block:: python

   import keras
   from keras_pos_embd import PositionEmbedding

   model = keras.models.Sequential()
   model.add(PositionEmbedding(
       input_dim=10,
       output_dim=2,
       mask_zero=10000,
       input_shape=(None,),
       name='Pos-Embd',
   ))

Arguments:


* ``input_dim``\ : The maximum absolute value of positions.
* ``output_dim``\ : The dimension of embeddings.
* ``mask_zero``\ : The index that presents padding (because ``0`` will be used in relative positioning).
