MRI simulator
===
Run the simulator locally
---
`panel serve app.py`

This deploys the app at http://localhost:5006/app

Deploy the simulator on a network
---
`panel serve app.py --port 80 --allow-websocket-origin=`*IP-number*`:80`

This allows network users to run the app by simply entering the server computers *IP-number* in their browser.

Deploy on Google App Engine:  
---
`gcloud app deploy --stop-previous-version`

This publishes the app on https://mrisimulator.ey.r.appspot.com

A guide to deploying python web applications using Google App Engine can be found on https://towardsdatascience.com/deploy-a-python-visualization-panel-app-to-google-cloud-cafe558fe787