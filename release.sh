#!/bin/bash
export PYTHONPATH=./pypaq:$PYTHONPATH
python setup.py sdist bdist_wheel
python -m twine upload -r pypi dist/*
rm -rf build
rm -rf dist
rm -rf torchness.egg-info