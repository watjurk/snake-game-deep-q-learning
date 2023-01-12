import socket

delimerLen = 16

__all__ = ['Protocol', 'Message']
class Message:
  Type: int
  Data: bytearray

def messageFromBytes(m: Message, messageBytes: bytearray):
  m.Type = int.from_bytes(messageBytes[:4], byteorder='little')
  messageData = messageBytes[4:]

  m.Data = bytearray(messageData)

def messageToBytes(m: Message, writer: socket.socket):
  messageType = m.Type.to_bytes(4, byteorder='little')

  writer.sendall(messageType)
  writer.sendall(m.Data)


class Protocol:
  def __init__(self, conn: socket.socket):
    self.delimer = bytearray(delimerLen)
    conn.recv_into(self.delimer)

    self.encoder = Encoder(conn, self.delimer)
    self.decoder = Decoder(conn, self.delimer)

  def Write(self, m: Message):
    self.encoder.Write(m)

  def Read(self) -> Message:
    return self.decoder.Read()


class Decoder:
  def __init__(self, reader: socket.socket, delimer: bytearray):
    self.r = reader
    self.delimer = delimer

    self.needRead = True
    self.buffer = bytearray()
    self.readBuffer = bytearray(128)

  def Read(d) -> Message:
    offset = 0
    while True:
      if d.needRead:
        n = d.r.recv_into(d.readBuffer)

        d.buffer += d.readBuffer[:n]
        d.needRead = False

      searchBytes = d.buffer[offset:]
      searchBytesIndex = 0

      delimerIndex = 0
      delimerFound = False

      for searchBytesIndex in range(len(searchBytes)):
        if searchBytes[searchBytesIndex] == d.delimer[delimerIndex]:
          delimerIndex += 1
        else:
          delimerIndex = 0

        if delimerIndex == len(d.delimer):
          delimerFound = True
          break

      offset += searchBytesIndex
      if delimerFound:
        messageBytes = d.buffer[:offset - len(d.delimer) + 1]
        message = Message()
        messageFromBytes(message, messageBytes)

        newBuffer = d.buffer[offset + 1:]
        # Will it work?
        d.buffer[:] = newBuffer
        d.buffer = d.buffer[:len(newBuffer)]

        return message

      d.needRead = True


class Encoder:
  def __init__(self, writer: socket.socket, delimer: bytearray):
    self.w = writer
    self.delimer = delimer

  def Write(self, m: Message):
    messageToBytes(m, self.w)
    self.w.sendall(self.delimer)
