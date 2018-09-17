from __future__ import print_function
import os
import sys
import numpy
import tensorflow as tf
import mnist_input_data
from read_data import random_mini_batches,read_data

tf.app.flags.DEFINE_integer('training_iteration', 10,      'number of training epochs.')
tf.app.flags.DEFINE_integer('batch_size',         64,      'mini batch size.')
tf.app.flags.DEFINE_float('learning_rate',        0.0001,  'learning rate.')
tf.app.flags.DEFINE_integer('model_version',      1,       'version number of the model.')
tf.app.flags.DEFINE_string('work_dir',            'data/', 'Working directory.')
FLAGS = tf.app.flags.FLAGS

def main(_):
  if len(sys.argv) < 2 or sys.argv[-1].startswith('-'):
    print('Usage: train.py [--training_iteration=x] '
          '[--model_version=y] export_dir')
    sys.exit(-1)
  if FLAGS.training_iteration <= 0:
    print('Please specify a positive value for training iteration.')
    sys.exit(-1)
  if FLAGS.model_version <= 0:
    print('Please specify a positive value for version number.')
    sys.exit(-1)

  ########################################    step 1 : read data    ###########################################
  
  #############   1(1) examples_1 mnist
  #mnist = mnist_input_data.read_data_sets(FLAGS.work_dir, one_hot=True)
  #train_x, train_y, test_x, test_y = mnist.train.images,mnist.train.labels,mnist.test.images,mnist.test.labels
  
  #############   1(2) examples_2 
  train_x, train_y, test_x, test_y = read_data(FLAGS.work_dir)
  train_x = numpy.float32(train_x)  # float64 convert to float32 
  test_x = numpy.float32(test_x)

  #########################################   step 2: create model   #############################################

  input_dim = train_x.shape[1]
  classes = train_y.shape[1]
  
  sess = tf.InteractiveSession()

  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  feature_configs = {'x': tf.FixedLenFeature(shape=[input_dim], dtype=tf.float32),}
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
  y_ = tf.placeholder('float', shape=[None, classes])

  w1 = tf.get_variable(name='w1',shape=[input_dim,32], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
  b1 = tf.get_variable(name='b1',shape=[32],           initializer = tf.zeros_initializer())
  w2 = tf.get_variable(name='w2',shape=[32,classes],   initializer = tf.contrib.layers.xavier_initializer(seed = 1))
  b2 = tf.get_variable(name='b2',shape=[classes],      initializer = tf.zeros_initializer())

  z1 = tf.nn.relu(tf.matmul(x, w1) + b1)
  #z1_drop = tf.nn.dropout(z1, 0.5)
  y  = tf.nn.softmax(tf.matmul(z1, w2) + b2, name='y')

  cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y),1))
  #optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate).minimize(cost)
  
  sess.run(tf.global_variables_initializer())  # Adam optimizer will generate variables, tf.global_variables_initializer() should be placed after Optimizer.  

  values, indices = tf.nn.top_k(y, classes)
  table = tf.contrib.lookup.index_to_string_table_from_tensor(tf.constant([str(i) for i in range(classes)]))
  prediction_classes = table.lookup(tf.to_int64(indices))

  ########################################   step 3: train model    ###########################################################

  print("start training..........")
  num_batch = int(train_x.shape[0]/FLAGS.batch_size)
  for epoch in range(FLAGS.training_iteration):
    avg_cost = 0.
    mini_batches = random_mini_batches(train_x, train_y, FLAGS.batch_size)

    for batches in mini_batches:                        
      batch_xs, batch_ys = batches
      _, batch_cost = sess.run([optimizer, cost], feed_dict={x: batch_xs, y_: batch_ys})
      avg_cost += batch_cost / num_batch

    if (epoch+1) % 5 == 0:
      print("Epoch:", epoch+1)
      print("cost=", "{:.9f}".format(avg_cost))
      correct  = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
      accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
      print("training set accuracy:", accuracy.eval(feed_dict={x: train_x, y_: train_y}))
      print(" ")

  correct  = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
  print("test set accuracy:", accuracy.eval(feed_dict={x: test_x, y_: test_y}))
  print("train done.............")

  #########################################   step 4 : export model   ###################################################################
 
  export_path_base = sys.argv[-1]
  export_path = os.path.join(tf.compat.as_bytes(export_path_base),tf.compat.as_bytes(str(FLAGS.model_version)))
  print('Exporting trained model to', export_path)
  
  ###############  4(1) create builder
  builder = tf.saved_model.builder.SavedModelBuilder(export_path)

  ###############  4(2) build classification_signature and prediction_signature of signature_def_map
  classification_inputs          = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
  classification_outputs_classes = tf.saved_model.utils.build_tensor_info(prediction_classes)
  classification_outputs_scores  = tf.saved_model.utils.build_tensor_info(values)

  classification_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs ={tf.saved_model.signature_constants.CLASSIFY_INPUTS:classification_inputs},
          outputs={tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:classification_outputs_classes,
                   tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:classification_outputs_scores},
          method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

  tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
  tensor_info_y = tf.saved_model.utils.build_tensor_info(y)

  prediction_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
          inputs={'input_x': tensor_info_x},
          outputs={'scores': tensor_info_y},
          method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

  ##############   4(3) builder.add_meta_graph_and_variables() 
  legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

  builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={'predict_y':prediction_signature,
                         tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:classification_signature,},
      legacy_init_op=legacy_init_op)

  builder.save()
  print('Done exporting!')

if __name__ == '__main__':
  tf.app.run()
