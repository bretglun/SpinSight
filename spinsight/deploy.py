# Deploy app on network

import socket
import panel as pn
import darkdetect
import optparse
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from spinsight import app
from datetime import datetime


def CLI():
    # parse command line
    p = optparse.OptionParser()
    p.add_option('--port', '-p', default=80,  type="int",
                    help="Port to deploy SpinSight")
    p.add_option('--network', '-n', action='store_true',
                    help="Deploy on local network")
    p.add_option('--url', '-u', default='',  type="string",
                    help="URL identifying server")
    p.add_option('--mode', '-m', default='',  type="string",
                    help="GUI mode (\"dark\"/\"light\")")
    p.add_option('--settingsFile', '-s', default='',  type="string",
                    help="settings filename (stem only)")
    p.add_option('--liveSliders', '-l',  action='store_true',
                    help="sliders update the image continuously while dragging (instead of only once released)")
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
            print(f'* http://{host}:{options.port}')
    
    dark_mode = darkdetect.isDark()
    if options.mode:
        if options.mode not in ['dark', 'light']:
            raise IOError('GUI mode must be either "dark" or "light"')
        dark_mode = options.mode == 'dark'

    # serve application
    try:
        startTime = datetime.now()
        def get_app(): return app.get_app(dark_mode, options.settingsFile, startTime, not options.liveSliders) # closure function
        pn.serve(get_app, show=False, title='SpinSight', port=options.port, websocket_origin=[f'{host}:{options.port}' for host in hosts])
    except OSError as e:
        print(e)
        print(f'Could not serve SpinSight on port {options.port}. Perhaps try another port (specify using the -p flag)')


if __name__ == '__main__':
    CLI()