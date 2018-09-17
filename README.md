# tf_serving
mnist example and myself example

## 1 training and save model 

$ python train.py --model_version=2 models/

## 2 start service

$ tensorflow_model_server --port=9000 --model_name=tf_model --model_base_path=/serving/models/

## 3 test client

$ python client.py --num_tests=100 --server=0.0.0.0:9000
