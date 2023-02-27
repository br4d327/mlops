#!/bin/bash
echo 'Start data_creation.py'
python3 data_creation.py
echo '##################'
echo 'Start model_preprocessing.py'
python3 model_preprocessing.py
echo '##################'
echo 'Start model_preparation.py'
python3 model_preparation.py
echo '##################'
echo 'Start model_testing.py'
python3 model_testing.py
