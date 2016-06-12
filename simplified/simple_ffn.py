import tensorflow as tf
import simplified.input_dummy as input_dummy
import math

BATCH_SIZE = 100
MAX_STEPS = 1000
NUM_CLASSES = 1
WINDOW_SIZE = 10
LEARNING_RAGE = 0.0001

def fill_feed_dict(data_set, inputs_pl, outputs_pl):
  inputs_feed, outputs_feed = data_set.next_batch(BATCH_SIZE)
  feed_dict = {
      inputs_pl: inputs_feed,
      outputs_pl: outputs_feed,
  }
  return feed_dict

def mse(logits, outputs):
  mse = tf.reduce_mean(tf.pow(tf.sub(logits, outputs), 2.0))
  return mse

def training(loss, learning_rate):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    return train_op

def inference_one_hidden_layer(inputs, hidden1_units):
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([WINDOW_SIZE, hidden1_units],stddev=1.0 / math.sqrt(float(WINDOW_SIZE))),name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
    hidden1 = tf.nn.relu(tf.matmul(inputs, weights) + biases)

  with tf.name_scope('identity'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, NUM_CLASSES],stddev=1.0 / math.sqrt(float(hidden1_units))),name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),name='biases')
    logits = tf.matmul(hidden1, weights) + biases

  return logits

def inference_two_hidden_layers(images, hidden1_units, hidden2_units):
  with tf.name_scope('hidden1'):
    weights = tf.Variable(tf.truncated_normal([WINDOW_SIZE, hidden1_units],stddev=1.0 / math.sqrt(float(WINDOW_SIZE))),name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),name='biases')
    hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)

  with tf.name_scope('hidden2'):
    weights = tf.Variable(tf.truncated_normal([hidden1_units, hidden2_units],stddev=1.0 / math.sqrt(float(hidden1_units))),name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)

  with tf.name_scope('identity'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, 1],stddev=1.0 / math.sqrt(float(hidden2_units))),name='weights')
    biases = tf.Variable(tf.zeros([1]),name='biases')

    logits = tf.matmul(hidden2, weights) + biases

  return logits

def run_training(h1_units,h2_units):

    data_sets = input_dummy.read_wind_data_sets()

    with tf.Graph().as_default():

        inputs_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE, WINDOW_SIZE))
        outputs_placeholder = tf.placeholder(tf.float32, shape=(BATCH_SIZE))

        logits = inference_two_hidden_layers(inputs_placeholder, h1_units,h2_units)
        #logits = inference_one_hidden_layer(inputs_placeholder, h1_units)
        loss = mse(logits,outputs_placeholder)

        train_op = training(loss,LEARNING_RAGE)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        for step in range(MAX_STEPS):
            feed_dict = fill_feed_dict(data_sets.train,inputs_placeholder,outputs_placeholder)
            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
            if step % 100 == 0:
                print('Step %d: loss = %.2f' % (step, loss_value))

def main(_):
  run_training(h1_units=10,h2_units=10)

if __name__ == '__main__':
  tf.app.run()