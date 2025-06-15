#!/usr/bin/env bash
set -ex

echo "Running git lfs install..."
git lfs install

# 1. Install Python requirements
echo "Installing Python requirements..."
pip install -v --progress-bar=on -r env/requirements.txt

# 2. Install the Blackfyre package in editable mode
echo "Installing Blackfyre in editable mode..."
pushd Blackfyre/src/python
pip install -v -e .
popd

# 3. Download and unzip lab datasets
curl -L -o lab_datasets-v2.zip "https://www.dropbox.com/scl/fi/36tfewp71smsa54pzonqc/lab_datasets-v2.zip?rlkey=ndbefbecgl02sb84txyq8rlkm&st=kwr5ksk6&dl=0"
unzip -o lab_datasets-v2.zip
rm lab_datasets-v2.zip

