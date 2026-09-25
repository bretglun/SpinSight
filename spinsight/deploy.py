# Deploy app on network

import optparse
import socket
import sys
from pathlib import Path

import darkdetect
import panel as pn

sys.path.append(str(Path(__file__).parent.parent))
from datetime import datetime

from spinsight import app


def CLI():
    # parse command line
    p = optparse.OptionParser()
    p.add_option('--noBrowser', '-b',  action='store_true',
                    help="Do not open a new browser window when the application launches")
    p.add_option('--port', '-p', default=80,  type="int",
                    help="Port to deploy SpinSight")
    p.add_option('--network', '-n', action='store_true',
                    help="Deploy on local network")
    p.add_option('--url', '-u', default='',  type="string",
                    help="URL identifying server")
    p.add_option('--instances', '-i', default=1,  type="int",
                    help="Number of instances to serve to handle more users (default: 1)")
    p.add_option('--mode', '-m', default='',  type="string",
                    help="GUI mode (\"dark\"/\"light\")")
    p.add_option('--settingsFile', '-s', default='',  type="string",
                    help="settings filename (stem only)")
    p.add_option('--liveSliders', '-l',  action='store_true',
                    help="sliders update the image continuously while dragging (instead of only once released)")
    p.add_option('--slidersWithText', '-t',  action='store_true',
                    help="add textbox next to sliders for increased control over input")
    options, _ = p.parse_args()

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
            print(f'* http://{host}:{options.port}')
    
    dark_mode = darkdetect.isDark()
    if options.mode:
        if options.mode not in ['dark', 'light']:
            raise OSError('GUI mode must be either "dark" or "light"')
        dark_mode = options.mode == 'dark'

    # serve application
    try:
        startTime = datetime.now()
        def get_app(): return app.get_app(dark_mode, options.settingsFile, startTime, not options.liveSliders, options.slidersWithText) # closure function
        pn.serve(get_app, show=not options.noBrowser, title='SpinSight', port=options.port, websocket_origin=[f'{host}:{options.port}' for host in hosts], num_procs=options.instances)
    except OSError as e:
        print(e)
        print(f'Could not serve SpinSight on port {options.port}. Perhaps try another port (specify using the -p flag)')


if __name__ == '__main__':
    CLI()