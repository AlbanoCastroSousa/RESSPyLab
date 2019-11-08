import numpy as np
import os
from RESSPyLab import opt_multi_run

# NOTE: THIS IS JUST A SAMPLE FILE TO ILLUSTRATE THE SET-UP OF THE opt_multi_run FUNCTION AND DOES NOT ACTUALLY RUN

# Locations of data sets
# opt_multi_run will load all the .csv files in each of the data directories, there must be no other files in each directory
data_dirs = ['dir1/my_data1/', 'dir1/my_data2/', 'dir1/my_data3/']

# Should use the "filter_data" option for each data set
filter_list = [False, False, False]

# Directory where output directories will be created
out_root = 'dir4/out_root_dir/'
# Get the names and set the output directory paths
names = [os.path.basename(os.path.normpath(s)) for s in data_dirs]
output_dirs = [os.path.join(out_root, n) for n in names]

# Set the initial point and run the optimizations
x_0 = np.array([200000., 355.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
x_0_all = [x_0 for d in data_dirs]
opt_multi_run(data_dirs, output_dirs, names, filter_list, x_0_all, model_type='UVC')
