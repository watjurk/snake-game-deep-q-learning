# Keep in sync with ml_tools/ui/server/service/video.go
import io

from PIL import Image

from ..protocol import Message, Protocol
from . import message_type


class Video:
  message_type_range = (200, 399)

  def __init__(self, protocol: Protocol):
    self.protocol = protocol
    pass

  def update_stream(self, stream_name: str, pixels):
    # imgBytes = bytearray(pixels, 'utf-8')

    img = Image.fromarray(pixels)
    imgBytesIO = io.BytesIO()
    img.save(imgBytesIO, format='PNG')
    imgBytes = imgBytesIO.getvalue()

    nameBytes = bytearray(stream_name, 'utf-8')

    try:
      nameLenBytes = len(nameBytes).to_bytes(4, byteorder='little')
    except:
      raise 'stream_name is to long'

    message = Message()
    message.Type = message_type.update_stream
    message.Data = nameLenBytes + nameBytes + imgBytes
    self.protocol.Write(message)

  def on_message(self, message: Message):
    pass
