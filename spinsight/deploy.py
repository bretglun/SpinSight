# Deploy app on network

import socket
import panel as pn
import optparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from spinsight import main


def CLI():
    # parse command line
    p = optparse.OptionParser()
    p.add_option('--port', '-p', default=80,  type="int",
                    help="Port to deploy SpinSight")
    p.add_option('--network', '-n', action='store_true',
                    help="Deploy on local network")
    p.add_option('--url', '-u', default='',  type="string",
                    help="URL identifying server")
    options, arguments = p.parse_args()

    hosts = []
    if options.network: # get IP number
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        hosts = [s.getsockname()[0]] # IP number
    if options.url:
        hosts.append(options.url)

    if len(hosts)>0:
        print('Deploying SpinSight at:')
        for host in hosts:
            print('* http://{}:{}'.format(host, options.port))

    # serve application
    pn.serve(main.getApp, show=False, title='SpinSight', port=options.port, websocket_origin=['{}:{}'.format(host, options.port) for host in hosts])

if __name__ == "__main__":
    CLI()

