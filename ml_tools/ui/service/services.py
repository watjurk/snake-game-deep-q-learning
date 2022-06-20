import threading

from ..protocol import Message, Protocol

from .control import Control
from .video import Video


class Services:
  def __init__(self, protocol: Protocol):
    self.protocol = protocol

    self.control = Control(protocol)
    self.video = Video(protocol)

    self.services = [self.control, self.video]

  def start(self):
    threading.Thread(target=self.start_reading_loop).start()

  def start_reading_loop(self):
    while True:
      message = self.protocol.Read()
      for service in self.services:
        (type_min, type_max) = service.message_type_range
        if message.Type >= type_min and message.Type <= type_max:
          service.on_message(message)
          break
