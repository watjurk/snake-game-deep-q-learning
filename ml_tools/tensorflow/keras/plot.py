import math

import matplotlib.pyplot as plt
import numpy as np
import tensorflow.keras

keras = tensorflow.keras

# Start bycia dumnym.

def plot_conv2d(layer: keras.layers.Layer, **kwargs):
  params = layer.get_weights()
  weights = params[0]

  weights = np.array(weights)

  # Flatten filters.
  filter_shape = weights.shape[0:2]
  num_chanels = weights.shape[2]
  num_filters_per_chanel = weights.shape[3]
  num_filters = num_chanels * num_filters_per_chanel

  flat_filters_shape = (*filter_shape, num_filters)
  weights = weights.reshape(flat_filters_shape)

  filters = []
  for i in range(num_filters):
    filters.append(weights[:, :, i])

  plot_filters(filters, filter_shape, **kwargs)


def plot_filters(filters, filter_shape, ax=None, vmin=None, vmax=None, vborder=0.12, ticks=None):
  ax_is_none = True if ax is None else False

  plt_ax = ax
  if ax_is_none:
    plt_ax = plt
    ax = plt.gca()

  border_per_filter_left_bottom = 1
  border_top_right = 1

  # Calculate number of pixels required for one filter.
  x_pix_per_filter = (filter_shape[0] + border_per_filter_left_bottom)
  y_pix_per_filter = (filter_shape[1] + border_per_filter_left_bottom)

  # Calculate size, size is amount of filters on x and y axises, we want our shape
  # to be semi-rectangular so we keep this amount the same for x and axises.
  size = math.ceil(math.sqrt(len(filters)))

  # If all of the space won't be occupied we need to know how much we need to shift
  # the index mapping, such that the 'blanks' would be in the left bottom corner.
  index_offset = size * size - len(filters)

  # Calculate the dimensions of our image.
  x_pix = size * x_pix_per_filter
  y_pix = size * y_pix_per_filter

  pix = np.full(
      # Add border_top_right to x_pix and y_pix so that we would get
      # a nice border at the top and right of our image.
      shape=(x_pix + border_top_right, y_pix + border_top_right),

      # All of the unused or border space would have this value.
      fill_value=vborder
  )

  for x in range(x_pix):
    for y in range(y_pix):
      # Calculate the x and y for the filter index.
      x_filter = math.floor(x / x_pix_per_filter)
      y_filter = math.floor(y / y_pix_per_filter)

      # Map x and y to a index so we could use it to index filters.
      filter_index = (x_filter * size + y_filter) - index_offset

      # Empty space.
      if filter_index >= len(filters) or filter_index < 0:
        continue

      # Calculate the x and y coordinates relative to current filter.
      x_in_filter = x % x_pix_per_filter
      y_in_filter = y % y_pix_per_filter

      # Left and bottom border lines.
      if x_in_filter in [0] or y_in_filter in [0]:
        continue

      # Finally set the value of the current pixel to the filter value at this pixel.
      f = filters[filter_index]
      pix[x][y] = f[x_in_filter - 1][y_in_filter - 1]

  c = plt_ax.pcolormesh(pix, cmap='RdBu_r', vmin=vmin, vmax=vmax)
  plt.colorbar(c, ax=ax, ticks=ticks)
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)


# Koniec bycia dumnym.