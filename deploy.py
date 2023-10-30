# Deploy app on network

import socket
import subprocess
import optparse


# parse command line
p = optparse.OptionParser()
p.add_option('--port', '-p', default='80',  type="string",
                help="Port to deploy SpinSight")
options, arguments = p.parse_args()

# get IP number
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80))
IP = s.getsockname()[0]

# serve application
print('Deploying SpinSight at IP address: {} (port {})'.format(IP, options.port))
subprocess.run('panel serve spinsight/spinsight.py --port {} --allow-websocket-origin={}:{}'.format(options.port, IP, options.port).split())