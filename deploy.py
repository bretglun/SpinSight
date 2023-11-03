# Deploy app on network

import socket
import subprocess
import optparse


# parse command line
p = optparse.OptionParser()
p.add_option('--port', '-p', default='80',  type="string",
                help="Port to deploy SpinSight")
p.add_option('--url', '-u', default='',  type="string",
                help="URL identifying server")
options, arguments = p.parse_args()

# get IP number
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 80))
IP = s.getsockname()[0]

command = 'panel serve spinsight/spinsight.py --port {} --allow-websocket-origin={}:{}'.format(options.port, IP, 
options.port)
print('Deploying SpinSight at IP address: {} (port {})'.format(IP, options.port))

if options.url:
    command += ' --allow-websocket-origin={}:{}'.format(options.url, options.port)
    print('Deploying SpinSight at URL: http://{}:{}'.format(options.url, options.port))

# serve application
subprocess.run(command.split())