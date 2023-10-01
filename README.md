# Hacettepe-Indoor-Space-Estimation
This repository contains the source code for the article "A Reproducible Approach for Estimating Indoor Space Area on Mobile Lidar Dataset Using DBSCAN". 

The 'indoor_area.py' file contains the functions and library imports used.
The 'test_indoor_area.py' file contains the code that needs to be run to replicate the experiment.
The 'configuration.yaml' file contains the parameters used in the experiment. if the experiment is to be run with different parameters, the 'test_indoor_area.py' file can be run again by changing the parameters in this file.

When the 'test_indoor_area.py' file is executed:
the point cloud data for the time interval specified in the configuration.yaml file will appear on the screen.
![Screenshot 2023-10-01 212921](https://github.com/Heisenberk-Karabay/Hacettepe-Indoor-Space-Estimation/assets/85685449/7f2e188b-6934-4947-86c6-ffff103a152b)

After this screen the dbscan algorithm runs, the data will be displayed on the screen.
![Screenshot 2023-10-01 213000](https://github.com/Heisenberk-Karabay/Hacettepe-Indoor-Space-Estimation/assets/85685449/8df1e3fa-3af9-414c-aef9-267870bd62e4)

After this screen, the dots consisting of the most repetitive pattern will appear on the screen.
![Screenshot 2023-10-01 213104](https://github.com/Heisenberk-Karabay/Hacettepe-Indoor-Space-Estimation/assets/85685449/3cfcce3c-02f3-4ba2-b8b6-50f973e7a8f5)

Finally, histograms of x y z data will be recorded and the calculated area value will be written to the console.
![X values](https://github.com/Heisenberk-Karabay/Hacettepe-Indoor-Space-Estimation/assets/85685449/fd7e2bd8-d388-42d6-8c62-c2d87c14abde)
![Y values](https://github.com/Heisenberk-Karabay/Hacettepe-Indoor-Space-Estimation/assets/85685449/626e1765-475f-4d1d-92fe-b8b3f56c8478)
![Z values](https://github.com/Heisenberk-Karabay/Hacettepe-Indoor-Space-Estimation/assets/85685449/ee21c4e6-ad08-4336-b0b6-64f9cee41acc)
