SpinSight MRI simulator
===
Run the simulator locally
---
`panel serve spinsight/spinsight.py`

This deploys the app at http://localhost:5006/app

Deploy the simulator on a network
---
`panel serve spinsight/spinsight.py --port 80 --allow-websocket-origin=`*IP-number*`:80`

This allows network users to run the app by simply entering the server computers *IP-number* in their browser.