"""Print a free TCP port on 127.0.0.1 (the judge port of run_packfix.sh)."""
import socket

s = socket.socket()
s.bind(("127.0.0.1", 0))
print(s.getsockname()[1])
s.close()
