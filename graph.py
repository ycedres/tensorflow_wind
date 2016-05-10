import tensorflow as tf
import input_data
import nn

from six.moves import xrange


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 1000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 128, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 32, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  ''Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'data', '/tmp/summary')


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

def run_training(h1_units,h2_units):

    data_sets = input_data.read_wind_data_sets()

    with tf.Graph().as_default():

        inputs_placeholder, outputs_placeholder = placeholder_inputs(FLAGS.batch_size)

        #logits = nn.inference_one_hidden_layer(inputs_placeholder, 10)

        logits = nn.inference_two_hidden_layers(inputs_placeholder, h1_units,h2_units)

        loss = nn.loss(logits, outputs_placeholder)

        train_op = nn.training(loss, FLAGS.learning_rate)

        eval_correct = nn.evaluation(logits, outputs_placeholder)

        summary_op = tf.merge_all_summaries()
        saver = tf.train.Saver()

        sess = tf.Session()
        init = tf.initialize_all_variables()
        sess.run(init)

        summary_writer = tf.train.SummaryWriter('/tmp/summary',graph_def=sess.graph_def)

        for step in range(FLAGS.max_steps):

            feed_dict = fill_feed_dict(data_sets.train,
                                       inputs_placeholder,
                                       outputs_placeholder)

            _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)

            import numpy as np
            if np.isnan(loss_value):
                print('stop')
            if step % 100 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f' % (step, loss_value))
                # Update the events file.
                summary_str = sess.run(summary_op, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)

            if (step + 1) == FLAGS.max_steps:
                saver.save(sess, FLAGS.train_dir, global_step=step)

                performance_mean = 0  # Counts the number of correct predictions.
                steps_per_epoch = data_sets.test.num_examples // FLAGS.batch_size
                #print('Numero de evaluaciones: %i' % steps_per_epoch)
                num_examples = steps_per_epoch * FLAGS.batch_size
                for step in xrange(steps_per_epoch):
                    feed_dict = fill_feed_dict(data_sets.test,
                                               inputs_placeholder,
                                               outputs_placeholder)
                    #performance_mean += sess.run(eval_correct, feed_dict=feed_dict)

                    performance_tmp = sess.run(eval_correct, feed_dict=feed_dict)
                    #print('Performance step %i: %f' % (step,performance_tmp))
                    performance_mean += performance_tmp

                performance = performance_mean / num_examples
                #print('H1 units: %i, H2 units: %i, Performance: %f' % (h1_units, h2_units, performance))
                return performance


def main(h1_units,h2_units):
  run_training(h1_units,h2_units)


if __name__ == '__main__':
  tf.app.run()
