import math
import tensorflow as tf


NUM_CLASSES = 1
WINDOW_SIZE = 10 # Tamaño de ventana


def inference_one_hidden_layer(inputs, hidden1_units):

  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([WINDOW_SIZE, hidden1_units],
                            stddev=1.0 / math.sqrt(float(WINDOW_SIZE))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    #hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
    #hidden1 = tf.nn.tanh(tf.matmul(images, weights) + biases)
    hidden1 = tf.nn.softmax(tf.matmul(inputs, weights) + biases)
  # Linear
  with tf.name_scope('identity'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden1, weights) + biases

  return logits



def loss(logits, outputs):

  #mse = tf.reduce_mean(tf.pow(tf.sub(logits,labels),tf.constant(2.0)))
  mse = tf.reduce_mean(tf.pow(tf.sub(logits, outputs), 2.0))
  cross_entropy = -tf.reduce_sum(outputs * tf.log(logits))

  return mse


def training(loss, learning_rate):
  """Sets up the training Ops.

  Creates a summarizer to track the loss over time in TensorBoard.

  Creates an optimizer and applies the gradients to all trainable variables.

  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.

  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.

  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op


def evaluation(logits, outputs):
    mse = tf.reduce_mean(tf.pow(tf.sub(logits, outputs), 2.0))
    return mse