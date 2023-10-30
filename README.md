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
The same thing is achieved by:  
`python deploy.py --port 80`

To keep the panel process running even after ending an ssh session:
* ssh into remote
* type `screen`
* start the panel process as described above
* press **Ctrl-A**, then **Ctrl-D**. This detaches the screen session with processes running
* you can now log off remote
* to resume the screen session later, login and type `screen -r` and you will see the process output