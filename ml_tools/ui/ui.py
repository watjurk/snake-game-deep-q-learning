import os
import socket
import subprocess
from os import path

from .protocol import Protocol
from .service.control import Control
from .service.services import Services
from .service.video import Video


class UI:
  control: Control
  video: Video

  def connect(self, public_folder_path: str, server_address: str = None):
    """
    Connect will try to connect to server specified at server_address
    If it won't succeed then it will start up the server.
    """
    if server_address == None:
      server_address = self.start_server(public_folder_path)

    self.connect_to_server(server_address)
    return server_address

  def start_server(self, public_folder_path: str):
    """
    Start server will start the ui server and return the addres of the server.
    """
    server_path = path.join(path.dirname(path.realpath(__file__)), 'server')

    if not path.isabs(public_folder_path):
      public_folder_path = path.normpath(path.join(os.getcwd(), public_folder_path))

    server = subprocess.Popen(['go', 'run', '.', '--public_folder_path={}'.format(public_folder_path)], stdout=subprocess.PIPE, cwd=server_path)
    return server.stdout.readline().decode('utf-8').rstrip()

  def connect_to_server(self, server_address: str):
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server_address = server_address.split(':')
    server_address[1] = int(server_address[1])
    client.connect(tuple(server_address))
    protocol = Protocol(client)

    services = Services(protocol)
    services.start()

    self.control = services.control
    self.video = services.video
