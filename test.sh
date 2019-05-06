#!/usr/bin/env bash
pycodestyle --max-line-length=120 keras_pos_embd tests && \
    nosetests --with-coverage --cover-erase --cover-html --cover-html-dir=htmlcov --cover-package=keras_pos_embd tests
