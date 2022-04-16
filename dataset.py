import os
from typing import Tuple

import numpy as np
import tensorflow as tf

pre_train_data_folder = "./00_pre-train-data"
def newTFRecordWriter():
  sequence_numbers = []

  file_names = os.listdir(pre_train_data_folder)
  for file_name in file_names:
    sequence_numbers.append(int(file_name.replace(".tfr", "")))

  if len(sequence_numbers) != 0:
    next_sequence_number = max(sequence_numbers) + 1
  else:
    next_sequence_number = 0

  filename = f"{next_sequence_number}.tfr"
  return tf.io.TFRecordWriter(os.path.join(pre_train_data_folder, filename))

def newTFRecordDataset():
  file_names = os.listdir(pre_train_data_folder)
  file_paths = []

  for file_name in file_names:
    file_paths.append(os.path.join(pre_train_data_folder, file_name))

  return tf.data.TFRecordDataset(file_paths)


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

  action = record['action']

  observation_shape = tf.sparse.to_dense(record['observation_shape'])
  observation = tf.sparse.to_dense(record['observation'])

  observation = tf.reshape(observation, observation_shape)

  return (action, observation)
