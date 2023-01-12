import tensorflow as tf


def dataset_len(dataset: tf.data.Dataset) -> int:
  return sum(1 for _ in dataset)
