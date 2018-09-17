from __future__ import print_function
import sys
import threading
import grpc
import numpy
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import mnist_input_data
from read_data import random_mini_batches,read_data

tf.app.flags.DEFINE_integer('concurrency', 1,'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '0.0.0.0:9000', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', 'data/', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS

class _ResultCounter(object):
  """Counter for the prediction results."""

  def __init__(self, num_tests, concurrency):
    self._num_tests = num_tests
    self._concurrency = concurrency
    self._error = 0
    self._done = 0
    self._active = 0
    self._condition = threading.Condition()

  def inc_error(self):
    with self._condition:
      self._error += 1

  def inc_done(self):
    with self._condition:
      self._done += 1
      self._condition.notify()

  def dec_active(self):
    with self._condition:
      self._active -= 1
      self._condition.notify()

  def get_error_rate(self):
    with self._condition:
      while self._done != self._num_tests:
        self._condition.wait()
      return self._error / float(self._num_tests)

  def throttle(self):
    with self._condition:
      while self._active == self._concurrency:
        self._condition.wait()
      self._active += 1


def _create_rpc_callback(label, result_counter):
  """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
  def _callback(result_future):
    """Callback function.

    Calculates the statistics for the prediction result.

    Args:
      result_future: Result future of the RPC.
    """
    exception = result_future.exception()
    if exception:
      result_counter.inc_error()
      print(exception)
    else:
      #sys.stdout.write('.')
      sys.stdout.flush()
      response = numpy.array(result_future.result().outputs['scores'].float_val)
      prediction = numpy.argmax(response)
      print("ture label:",label,"   and   prediction: ",prediction)

      if label != prediction:
        result_counter.inc_error()
    result_counter.inc_done()
    result_counter.dec_active()
  return _callback


def do_inference(hostport, work_dir, concurrency, num_tests):
  """Tests PredictionService with concurrent requests.

  Args:
    hostport: Host:port address of the PredictionService.
    work_dir: The full path of working directory for test data set.
    concurrency: Maximum number of concurrent requests.
    num_tests: Number of test images to use.

  Returns:
    The classification error rate.

  Raises:
    IOError: An error occurred processing test data set.
  """
  #######################################  start ------ Try to use yourself data  ########################
  """ 
  Note:
       input_data.shape = (num_of_samples, dim_of_samples)
       input_label is list.
       input_data : float32 rather than float 64
  """
  #########  example (1) mnist
  #test_data_set = mnist_input_data.read_data_sets(work_dir).test
  #input_data,label = test_data_set.next_batch(num_tests) # input_data.shape = (num_of_samples, dim_of_samples)
  #input_label=list(label)  # input_label is list.  
  
  #########  example (2) 
  _, _, test_x, test_y = read_data(FLAGS.work_dir)
  test_x=numpy.float32(test_x)
  input_data  = test_x[0:num_tests,:]
  input_label = list(numpy.argmax(test_y,axis=1))

  ########################################  end   ########################################################

  channel = grpc.insecure_channel(hostport)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  result_counter = _ResultCounter(num_tests, concurrency)

  for i in range(num_tests):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'tf_serving'
    request.model_spec.signature_name = 'predict_y'

    input_x,label = input_data[i,:],input_label[i]
    request.inputs['input_x'].CopyFrom(tf.contrib.util.make_tensor_proto(input_x, shape=[1, input_x.size]))
    result_counter.throttle()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(_create_rpc_callback(label, result_counter))
  return result_counter.get_error_rate()


def main(_):
  if FLAGS.num_tests > 10000:
    print('num_tests should not be greater than 10k')
    return
  if not FLAGS.server:
    print('please specify server host:port')
    return
  error_rate = do_inference(FLAGS.server, FLAGS.work_dir,FLAGS.concurrency, FLAGS.num_tests)
  print('\nInference error rate: %s%%' % (error_rate * 100))

if __name__ == '__main__':
  tf.app.run()
