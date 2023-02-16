# Deploy app on network

import socket
import subprocess

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80))
IP = s.getsockname()[0]

print('Deploying app at IP address: ' + IP)
subprocess.run('panel serve app.py --port 80 --allow-websocket-origin={}:80'.format(IP).split())