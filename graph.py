import tensorflow as tf
import input_data
import nn

from six.moves import xrange


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 10000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  ''Must divide evenly into the dataset sizes.')


def placeholder_inputs(batch_size):
  inputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size,nn.WINDOW_SIZE))
  outputs_placeholder = tf.placeholder(tf.float32, shape=(batch_size))
  return inputs_placeholder, outputs_placeholder


def fill_feed_dict(data_set, inputs_pl, outputs_pl):
  inputs_feed, outputs_feed = data_set.next_batch(FLAGS.batch_size)
  feed_dict = {
      inputs_pl: inputs_feed,
      outputs_pl: outputs_feed,
  }
  return feed_dict

def run_training():

    data_sets = input_data.read_wind_data_sets()

    with tf.Graph().as_default():

        inputs_placeholder, outputs_placeholder = placeholder_inputs(FLAGS.batch_size)

        logits = nn.inference_one_hidden_layer(inputs_placeholder, 10)
        loss = nn.loss(logits, outputs_placeholder)

        train_op = nn.training(loss, FLAGS.learning_rate)

        eval_correct = nn.evaluation(logits, outputs_placeholder)

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        for step in range(FLAGS.max_steps):

            feed_dict = fill_feed_dict(data_sets.train,
                                       inputs_placeholder,
                                       outputs_placeholder)

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            if (step + 1) == FLAGS.max_steps:
                true_count = 0  # Counts the number of correct predictions.
                steps_per_epoch = data_sets.test.num_examples // FLAGS.batch_size
                num_examples = steps_per_epoch * FLAGS.batch_size
                for step in xrange(steps_per_epoch):
                    feed_dict = fill_feed_dict(data_sets.test,
                                               inputs_placeholder,
                                               outputs_placeholder)
                    true_count += sess.run(eval_correct, feed_dict=feed_dict)
                performance = true_count / num_examples
                print('Performance: %f' % performance)


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()
