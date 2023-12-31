import numpy as np
import pandas as pd 
import open3d as o3d
from collections import Counter
import matplotlib.pyplot as plt
import yaml
import os

def read_configuration(configuration_file):
    # Configuration is provided as a yaml file
    with open(configuration_file, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
            return conf
        except yaml.YAMLError as exc:
            print(exc)


def find_time(conf, **kwargs):
    is_completed = False
    counter = 0
    
    if('enter' in kwargs):
        rowskip = kwargs['enter'] # if we are processing the exit
        search_time = conf["time_end"] 
    else:
        rowskip = conf["rowskip"]
        search_time = conf["time_start"]


    while is_completed == False:
        if (conf["log"]):
            print("Processing rows in Laz (~csv) file: ", str(rowskip + (counter*conf["stepsize"]) ))

        point_cloud_data = pd.read_csv(conf["input_points"], 
                                       skiprows = rowskip, 
                                       nrows= conf["stepsize"], 
                                       header = 1 , 
                                       delimiter=" ")
        # Column names may change - check the corresponding csv file
        point_cloud_data.columns = ['X','Y','Z','R','G','B','Time','Intensity']
        
        if(point_cloud_data['Time'][0] > search_time):
            # our rowskip is higher than its real value
            print("Provide a smaller rowskip value")
            rowskip = -1 
            time = -1
            return (rowskip, time)

        unique_list = np.unique(point_cloud_data['Time'])
    
        for index in range(len(unique_list)):
            # the times of panaroma images may not match with point cloud exactly
            # a margin of 0.1 seconds is considered
            if  unique_list[index] - 0.1 <= search_time <= unique_list[index] + 0.1:
                time = unique_list[index]
                is_completed = True
                return (rowskip, time)
     
        rowskip += conf["stepsize"]
        del point_cloud_data

def interval(conf, skip_start, skip_end, time_start, time_end):
    
    point_cloud_data = pd.read_csv(conf['input_points'],
                                   skiprows = skip_start, 
                                   nrows = skip_end - skip_start + conf['stepsize'], 
                                   header=1 , 
                                   delimiter=" ")

    point_cloud_data.columns = ['X','Y','Z','R','G','B','Time','Intensity']

    for index in range(skip_end - skip_start + conf["stepsize"]):
        if point_cloud_data['Time'][index] == time_start:
            
            index_start = index
            print(f'index start: {index_start}')
            break

    for index in range(skip_end - skip_start + conf["stepsize"]):
        if point_cloud_data['Time'][index] == time_end:
            
            index_end = index
            print(f'index_end: {index_end}')
            break
    return index_start, index_end

def visualise_pcd(conf, rowskip_enter, rowskip_exit, time_enter, time_exit):
    
    start_index, end_index = interval(conf, 
                                      rowskip_enter, 
                                      rowskip_exit, 
                                      time_enter, 
                                      time_exit)
    
    point_cloud_data = pd.read_csv(conf["input_points"],
                                   skiprows = rowskip_enter + start_index, 
                                   nrows = end_index - start_index, 
                                   header=1 , 
                                   delimiter=" ")
    
    point_cloud_data.columns = ['X','Y','Z','R','G','B','Time','Intensity']
      
    to_numpy_xyz = point_cloud_data[['X', 'Y', 'Z']].copy()
    to_numpy_rgb = point_cloud_data[['R', 'G', 'B']].copy()
    
    df_p = pd.DataFrame(to_numpy_xyz).to_numpy()
    df_c = pd.DataFrame(to_numpy_rgb).to_numpy()
    
    xyz = np.zeros((end_index - start_index, 3))
    abc = np.zeros((end_index - start_index, 3))
    
    xyz[:,0] = df_p[:,0]
    xyz[:,1] = df_p[:,1]
    xyz[:,2] = df_p[:,2]
    abc[:,0] = df_c[:,0] / 255
    abc[:,1] = df_c[:,1] / 255
    abc[:,2] = df_c[:,2] / 255
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors  = o3d.utility.Vector3dVector(abc)

    o3d.visualization.draw_geometries([pcd])

    return pcd

def run_dbscan(conf,point_cloud):

    voxel_size = conf['dbscan']['down_sample_ratio']
    eps = conf['dbscan']['eps']
    min_points = conf['dbscan']['min_points']

    downpcd = point_cloud.voxel_down_sample(voxel_size=voxel_size)

    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(downpcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
        print(labels)
    label_counts = Counter(labels)
    most_common_label = label_counts.most_common(1)[0][0]
    print(f"The most repeated label is: {most_common_label}")

    max_label = labels.max()
    print(f"point cloud has {max_label + 1} clusters")
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    downpcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([downpcd])

    most_common_array = []
    for point in range(len(np.asarray(downpcd.points))):
        if labels[point] != most_common_label:
            pass
        elif labels[point] == most_common_label:
            most_common_array.append(np.asarray(downpcd.points)[point])

    numpy_array = np.array(most_common_array)
    del most_common_array

    pcd_cleaned = o3d.geometry.PointCloud()
    pcd_cleaned.points = o3d.utility.Vector3dVector(numpy_array)
    o3d.visualization.draw_geometries([pcd_cleaned])
    
    return pcd_cleaned

def draw_histogram(point_cloud):

    x_data = np.asarray(point_cloud.points)[:,0]    
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(x_data, 1000)
    ax.set_xlabel('X value')
    ax.set_ylabel('Number of occurrence')
    ax.set_title('Histogram of x values')
    plt.savefig('X values.png')
    print('figure saved')
    plt.show()
    
    y_data = np.asarray(point_cloud.points)[:,1]    
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(y_data, 1000)
    ax.set_xlabel('Y value')
    ax.set_ylabel('Number of occurrence')
    ax.set_title('Histogram of y values')
    plt.savefig('Y values.png')
    print('figure saved')
    plt.show()

    z_data = np.asarray(point_cloud.points)[:,2]    
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(z_data, 1000)
    ax.set_xlabel('Z value')
    ax.set_ylabel('Number of occurrence')
    ax.set_title('Histogram of z values')
    plt.savefig('Z values.png')
    print('figure saved')
    plt.show()

def calculate_area(point_cloud):

    x_diff = abs(np.amax(np.asarray(point_cloud.points),0)[0] - np.amin(np.asarray(point_cloud.points),0)[0])
    y_diff = abs(np.amax(np.asarray(point_cloud.points),0)[1] - np.amin(np.asarray(point_cloud.points),0)[1])

    print(f'Calculated area: {x_diff*y_diff}')