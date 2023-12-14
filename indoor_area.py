import numpy as np
import pandas as pd 
import open3d as o3d
from collections import Counter
import matplotlib.pyplot as plt
import yaml
import os
import csv

def read_configuration(configuration_file):
    # Configuration is provided as a yaml file
    with open(configuration_file, "r") as stream:
        try:
            conf = yaml.safe_load(stream)
            return conf
        except yaml.YAMLError as exc:
            print(exc)

def abs_time_diff(time1,time2):
    return abs(time1-time2)

def create_skiprow_data(conf):
    skiprow_data = []
    skiprows=0

    data_path = conf['input_points']
    no_of_rows = conf['stepsize']
    skiprow_data_path = conf['rowskip_data_path']

    if os.path.exists(skiprow_data_path):
        pass
    else:
        try:
            while True:
                data = pd.read_csv(data_path,skiprows=skiprows,nrows=no_of_rows,delimiter=' ',header=1)
                time_init = data.iloc[0][6]
                time_final = data.iloc[-1][6]
                skiprow_data.append((time_final-time_init,skiprows))
                print(f'creating rowskip data for {skiprows}-{skiprows + no_of_rows}')
                skiprows += no_of_rows

        except IndexError: 
            pass

        df = pd.DataFrame(skiprow_data)
        df.columns = ['time_difference(sec)','data_interval']
        df.to_csv(skiprow_data_path,index= False)

def find_time(conf, **kwargs):

    initial_time = conf["time_initial"]
    time_start= conf["time_start"]
    rowskip_data_path = conf["rowskip_data_path"]
    stepsize = conf["stepsize"]
    is_completed = False
    counter = 0

    create_skiprow_data(conf)

    data = pd.read_csv(rowskip_data_path,dtype=float)
    time_diff = abs_time_diff(initial_time,time_start)
    
    cumilative_time = 0
    
    for index in range(len(data['time_difference(sec)'])):

        if cumilative_time < time_diff:
            cumilative_time += data['time_difference(sec)'][index]

        elif cumilative_time > time_diff:
            conf["rowskip"] = (index-1) * stepsize

            break
    
    if('enter' in kwargs):
        rowskip = kwargs['enter'] # if we are processing the exit
        search_time = conf["time_end"] 
    else:
        rowskip = conf["rowskip"]
        search_time = conf["time_start"]

    while is_completed == False:
        if (conf["log"]):
            print("Processing rows in Laz (~txt) file: ", str(rowskip + (counter*conf["stepsize"]) ))

        point_cloud_data = pd.read_csv(conf["input_points"], 
                                       skiprows = rowskip, 
                                       nrows= conf["stepsize"], 
                                       header = 1 , 
                                       delimiter=" ")
        # Column names may change - check the corresponding csv file
        point_cloud_data.columns = ['X','Y','Z','R','G','B','Time','Intensity']

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

    index_start = 0
    index_end = 0

    for index in range(skip_end - skip_start + conf["stepsize"]):
        if point_cloud_data['Time'][index] == time_start:
            
            index_start = index
            break

    for index in range(skip_end - skip_start + conf["stepsize"]):
        if point_cloud_data['Time'][index] == time_end:
            
            index_end = index
            print(f'no total proccessed points: {index_end - index_start}')
            conf['no_of_points'] = index_end - index_start
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
            o3d.utility.VerbosityLevel.Error) as cm:
        labels = np.array(downpcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
        print(labels)
    label_counts = Counter(labels)
    most_common_label = label_counts.most_common(1)[0][0]

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

    print(f'no of all points: {conf["no_of_points"]}')
    print(f'no of cleaned points: {len(np.asarray(pcd_cleaned.points))}')
    print(f'ratio of cleaned/all : { conf["no_of_points"] / len(np.asarray(pcd_cleaned.points))}')
    o3d.visualization.draw_geometries([pcd_cleaned])
    
    return pcd_cleaned

def draw_histogram(conf,point_cloud):

    manual_adj = conf['manual_adjustment']

    filename_x = 'x_hist'
    filename_y = 'y_hist'

    x_data = np.asarray(point_cloud.points)[:,0]    
    fig, ax = plt.subplots()
    n, bins, patches = ax.hist(x_data, 1000)
    ax.set_xlabel('X value')
    ax.set_ylabel('Number of occurrence')
    ax.set_title('Histogram of x values')
    plt.savefig('X values.png')
    print('figure saved')
    plt.show()
    np.savetxt(filename_x,x_data)

    if manual_adj:

        print('choose the first x value')
        x_input1 = float(input())
        print('choose the second x value')
        x_input2 = float(input())

        x_data = pd.read_csv('x_hist',header=None)
        fig, ax = plt.subplots()
        ax.axvline(x=x_input1, color='r', linestyle='--')
        ax.axvline(x=x_input2, color='r', linestyle='--')
        n, bins, patches = ax.hist(x_data, 1000)
        ax.set_xlabel('X value')
        ax.set_ylabel('Number of occurrence')
        ax.set_title('Histogram of x values')
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
    np.savetxt(filename_y,y_data)
    
    if manual_adj:

        print('choose the first y value')
        y_input1 = float(input())
        print('choose the second y value')
        y_input2 = float(input())

        y_data = pd.read_csv('y_hist',header=None)
        fig, ax = plt.subplots()
        ax.axvline(x=y_input1, color='r', linestyle='--')
        ax.axvline(x=y_input2, color='r', linestyle='--')
        n, bins, patches = ax.hist(y_data, 1000)
        ax.set_xlabel('Y value')
        ax.set_ylabel('Number of occurrence')
        ax.set_title('Histogram of x values')
        plt.show()

        print(f'graph calculated area: {abs(y_input2-y_input1)*abs(x_input2-x_input1)}')
    
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
