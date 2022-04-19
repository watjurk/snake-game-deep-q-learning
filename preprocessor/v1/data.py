import os
import numpy as np
import tensorflow as tf


data_folder = os.path.join(os.path.dirname(__file__), "pre-train_data")

def encode(action, observation):
  record = tf.train.Example(features=tf.train.Features(feature={
    "action": tf.train.Feature(int64_list=tf.train.Int64List(value=[int(action)])),
    "observation": tf.train.Feature(float_list=tf.train.FloatList(value=np.reshape(observation, -1))),
    "observation_shape": tf.train.Feature(int64_list=tf.train.Int64List(value=np.asarray(observation.shape)))
  }))
  record_bytes = record.SerializeToString()
  return record_bytes


def decode(record_bytes):
  record = tf.io.parse_example(
      # Data
      record_bytes,
      # Schema
      {
        "action": tf.io.FixedLenFeature([], tf.int64),
        "observation": tf.io.VarLenFeature(tf.float32),
        "observation_shape": tf.io.VarLenFeature(tf.int64),
      }
  )

  action = record["action"]

  observation_shape = tf.sparse.to_dense(record["observation_shape"])
  observation = tf.sparse.to_dense(record["observation"])

  observation = tf.reshape(observation, observation_shape)

  return (action, observation)
