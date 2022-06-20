# Keep in sync with ml_tools/ui/server/service/control.go

import json

from ..protocol import Message, Protocol
from . import message_type


class Control:
  message_type_range = (100, 199)

  def __init__(self, protocol: Protocol):
    self.protocol = protocol
    self.map = {}

  def on_message(self, message: Message):
    if message.Type == message_type.set_value:
      data = json.loads(message.Data)
      self.map[data['key']] = data['value']

  def get_value(self, key: str) -> str:
    return self.map.get(key)
