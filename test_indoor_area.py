from indoor_area import *

configuration_file = 'configuration.yaml'
conf = read_configuration(configuration_file)

rowskip_enter, time_enter = find_time(conf) 

if(rowskip_enter != -1):
    rowskip_exit, time_exit = find_time(conf, enter = rowskip_enter)  
    
    pcd = visualise_pcd(conf, 
                        rowskip_enter, 
                        rowskip_exit, 
                        time_enter, 
                        time_exit)
    
    cleaned_pcd = run_dbscan(conf, pcd)

    draw_histogram(conf,cleaned_pcd)
    calculate_area(cleaned_pcd)

