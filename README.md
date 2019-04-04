# TensorFlow Serving
This a simple mnist example of using TensorFlow Serving.

## 1 training and saving model 

$ python train.py --model_version=3 models/

## 2 start service

$ tensorflow_model_server --port=9000 --model_name=tf_model --model_base_path=/serving/models/

## 3 test client

$ python client.py --num_tests=100 --server=0.0.0.0:9000
