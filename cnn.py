import tensorflow as tf
import numpy as np


# Names of all training files
train_files = []

# Initially use first 8 CHR1 training files, 9th for test
for i in range(8):
  filename = '/mnt/chr1/illumina_chr1.shuffled.training.examples.tfrecord-0000' + str(i) + '-of-00100'
  train_files.append(filename)


# Names of all eval files
eval_files = []
eval_files.append('/mnt/chr1/illumina_chr1.shuffled.training.examples.tfrecord-00008-of-00100')

# Number of features = 100 * 221 * 7?
# (TODO): is this correct
numfeatures = 100 * 221 * 7


# Function to read files and return datasets
def read_and_decode(train_files, eval_files):

  train_filename_queue = tf.train.string_input_producer(
        train_files, num_epochs=1)

  eval_filename_queue = tf.train.string_input_producer(
        eval_files, num_epochs=1)

  reader = tf.TFRecordReader()

  # For Train Files
  _, train_serialized_example = reader.read(train_filename_queue)

  train_features = tf.parse_single_example(
      train_serialized_example,
      features={
          'image/encoded': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  train_data = tf.decode_raw(train_features['image/encoded'], tf.float32)

  # (TODO): not sure if this reshaping is right
  train_data.set_shape([numfeatures])

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  train_labels = tf.cast(train_features['label'], tf.int32)


  # For Eval Files
  _, eval_serialized_example = reader.read(eval_filename_queue)

  eval_features = tf.parse_single_example(
      eval_serialized_example,
      features={
          'image/encoded': tf.FixedLenFeature([], tf.string),
          'label': tf.FixedLenFeature([], tf.int64),
      })

  eval_data = tf.decode_raw(eval_features['image/encoded'], tf.float32)

  # (TODO): not sure if this reshaping is right
  eval_data.set_shape([numfeatures])

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  eval_labels = tf.cast(eval_features['label'], tf.int32)


  return train_data, train_labels, eval_data, eval_labels




def cnn_model_fn(features, labels, mode):
  # input layer
  # (TODO): idk what our input chromosome image size is- assuming is 256
  input_layer = tf.reshape(features["x"], [-1, numfeatures, numfeatures, 1])

  # conv layer with 16 features and 4x4 filter using relu
  conv_layer_0 = tf.layers.conv2d(
      inputs=input_layer,
      filters=16,
      kernel_size=[4, 4],
      padding="same",
      activation=tf.nn.relu)

  # pooling layer with 2x2 filter
  pool_layer_1 = tf.layers.max_pooling2d(inputs=conv_layer_0, pool_size=[2, 2], strides=2)

  # conv layer with 32 features
  conv_layer_2 = tf.layers.conv2d(
      inputs=pool_layer_1,
      filters=32,
      kernel_size=[4, 4],
      padding="same",
      activation=tf.nn.relu)

  # pooling layer
  pool_layer_3 = tf.layers.max_pooling2d(inputs=conv_layer_2, pool_size=[2, 2], strides=2)

  flat_layer_4 = tf.reshape(pool_layer_3, [-1, _ * _ * _])

  # dense layer with 512 neurons
  dense = tf.layers.dense(inputs=flat_layer_4, units=512, activation=tf.nn.relu)

  dropout = tf.layers.dropout(
      inputs=dense, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)

  # logits
  logits = tf.layers.dense(inputs=dropout, units=3)

  predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=3)
  loss = tf.losses.softmax_cross_entropy(
      onehot_labels=onehot_labels, logits=logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
  # (TODO)
  # load chromosome data stuff here
  train_data, train_labels, eval_data, eval_labels = read_and_decode(train_files, eval_files)
  print("Read input files successfully")
  print(train_data, train_labels, eval_data, eval_labels)

  # make classifier with cnn_model_fn
  classifier = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/variant")

  # # log this
  # tensors_to_log = {"probabilities": "softmax_tensor"}
  # logging_hook = tf.train.LoggingTensorHook(
  #     tensors=tensors_to_log, every_n_iter=50)

  # Train the model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": train_data},
      y=train_labels,
      batch_size=100,
      num_epochs=None,
      shuffle=True)

  # classifier.train(
  #     input_fn=train_input_fn,
  #     steps=30000,
  #     hooks=[logging_hook])

  # No logging hook for now because it throws error
  classifier.train(
      input_fn=train_input_fn, steps=30000)

  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": eval_data},
      y=eval_labels,
      num_epochs=1,
      shuffle=False)
  
  eval_results = classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)


if __name__ == "__main__":
  tf.app.run()
