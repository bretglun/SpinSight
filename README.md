SpinSight MRI simulator
===
Running the simulator
---
See:  
`python spinsight/deploy.py -h`

To keep the panel process running even after ending an ssh session:
* ssh into remote
* type `screen`
* start the panel process as described above
* press **Ctrl-A**, then **Ctrl-D**. This detaches the screen session with processes running
* you can now log off remote
* to resume the screen session later, login and type `screen -r` and you will see the process output