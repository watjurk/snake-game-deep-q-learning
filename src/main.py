import threading
import time

import torch

import gui

print("Hello world!")
print("Torch version:", torch.__version__)

print("Torch default device:", torch.get_default_device())
print("Torch mps available:", torch.backends.mps.is_available())
print("Torch cuda available:", torch.cuda.is_available())

threading.Thread(target=gui.start).start()

time.sleep(100)
