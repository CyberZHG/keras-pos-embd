from setuptools import setup

setup(
    name='keras-pos-embd',
    version='0.3',
    packages=['keras_pos_embd'],
    url='https://github.com/CyberZHG/keras-pos-embd',
    license='MIT',
    author='CyberZHG',
    author_email='CyberZHG@gmail.com',
    description='Position embedding layers in Keras',
    long_description=open('README.rst', 'r').read(),
    install_requires=[
        'numpy',
        'keras',
    ],
    classifiers=(
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
