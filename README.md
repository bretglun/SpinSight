SpinSight MRI simulator
===
SpinSight is an MRI simulator written in Python and created for educational puposes. It jointly visualizes the imaging parameters, the MRI pulse sequence, the k-space data matrix, and the MR image. These are updated in near real-time when the user changes parameters. The simulator is run as a web browser dashboard. The data is simulated from computational 2D phantoms in vector graphics format (SVG).

Running the Simulator
---
Install using pip: 
```
pip install spinsight
```
Then run as a command line tool
```
spinsight
```
This serves SpinSight on the local host, so that the simulator can be run by navigating to [localhost](http://localhost) in the web browser. The same command line tool can be used to deploy the simulator on a local network, or on a web server (run `spinsight -h` for help). Be aware that several minutes are required upon loading a phantom for the first time.  

Phantom construction
--------------------
To create a new phantom, add a directory with the phantom name under [spinsight/phantoms](./spinsight/phantoms). This directory shall contain specifications in a `.toml` file with the same name (see [brain.toml](./spinsight/phantoms/brain/brain.toml) for reference). The specified `.svg` file must meet the following specifications:
* All paths must be closed
* All paths must have a fill color matching a hexcolor defined in the `TISSUES` dict in [constants.py](./spinsight/constants.py) (this defines the tissue).
* Only polygons are supported (not Bézier curves etc)

Alternatively a second `.toml` file can be specified with a list of shapes (see [Shepp-Logan_shapes.toml](./spinsight/phantoms/Shepp-Logan/Shepp-Logan_shapes.toml) for reference).

Dependencies
------------
See [pyproject.toml](./pyproject.toml) under heading **[tool.poetry.dependencies]**. 

License
-------
SpinSight is distributed under the terms of the GNU General Public License. See [LICENSE.md](./LICENSE.md).

Contact Information
-------------------
Johan Berglund, Ph.D.  
Uppsala University Hospital,  
Uppsala, Sweden  
johan.berglund@akademiska.se

![](spinsight.png)

---
Copyright © 2021–2025 Johan Berglund.