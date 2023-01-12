import os

import tensorflow as tf


def newTFRecordWriter(data_folder: str) -> tf.io.TFRecordWriter:
  sequence_numbers = []

  file_names = os.listdir(data_folder)
  for file_name in file_names:
    sequence_numbers.append(int(file_name.replace(".tfr", "")))

  if len(sequence_numbers) != 0:
    next_sequence_number = max(sequence_numbers) + 1
  else:
    next_sequence_number = 0

  filename = f"{next_sequence_number}.tfr"
  return tf.io.TFRecordWriter(os.path.join(data_folder, filename))


def newTFRecordDataset(data_folder) -> tf.data.TFRecordDataset:
  file_names = os.listdir(data_folder)

  file_paths = []
  for file_name in file_names:
    file_paths.append(os.path.join(data_folder, file_name))

  return tf.data.TFRecordDataset(file_paths)

