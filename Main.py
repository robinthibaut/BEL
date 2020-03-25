# %% Import
import os
from os.path import join as jp
import shutil

import numpy as np
import uuid
from multiprocessing import Process

from MyToolbox import FileOps
from Flow import my_flow
from Transport import my_transport
from Backtracking import my_backtracking
from SignedDistance import signed_distance


# Directories
cwd = os.getcwd()
exe_loc = jp(cwd, 'exe')
# EXE files directory.
exe_name_mf = jp(exe_loc, 'mf2005.exe')
exe_name_mt = jp(exe_loc, 'mt3d.exe')
exe_name_mp = jp(exe_loc, 'mp7.exe')

# TODO: check if the code does what I want when I change the discretization
# FIXME: Pumping well position in wells_data
# Define injection wells
# [ [X, Y, [r#1, r#2... r#n]] ]
# Pumping well located at [1000, 500]

# Template
# wells_data = np.array([[980, 500, [0, 24, 0]]])

# x = np.arange(850, 1100, 40)
# y = np.arange(350, 650, 40)
# len(x)*len(y)

# x = np.arange(950, 1150, 100)
# y = np.arange(450, 650, 100)
# len(x)*len(y)
# wells_xy = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])


def gen_rand_well(radius, x0, y0):
    # random angle
    alpha = 2 * np.pi * np.random.random()
    # random radius
    r = 50 + radius * np.sqrt(np.random.random())
    # calculating coordinates
    x = r * np.cos(alpha) + x0
    y = r * np.sin(alpha) + y0

    return [x, y]

# # radius of the circle
# circle_r = 100
# # center of the circle (x, y)
# circle_x = 980
# circle_y = 500
# ww = [gen_rand_well(circle_r, circle_x, circle_y) for i in range(50)]
# ww = np.array(ww)
# np.save(jp(cwd, 'grid', 'iw'), ww)
# import matplotlib.pyplot as plt
# # plt.plot(ww[:,0], ww[:,1], 'ko')
# # plt.show()
# # plt.close()

pumping_well = np.array([[1000, 500, [-1000, -1000, -1000]]])
pwxy = pumping_well[0, :2]

wells_xy = [pwxy + [-50, -50],
            pwxy + [-70, 60],
            pwxy + [-100, 5],
            pwxy + [68, 15],
            pwxy + [30, 80],
            pwxy + [50, -30]]
# wells_xy = np.load(jp(cwd, 'grid', 'iw.npy'))
wells_data = np.array([[wxy[0], wxy[1], [0, 24, 0]] for wxy in wells_xy])


def main():
    # Main results directory.
    res_dir = uuid.uuid4().hex
    results_dir = jp(cwd, 'results', res_dir)
    # Generates the result directory
    FileOps.dirmaker(results_dir)
    np.save(jp(cwd, 'grid', 'iw'), wells_data)  # Saves injection wells stress period data
    np.save(jp(results_dir, 'iw'), wells_data)  # Saves injection wells stress period data
    np.save(jp(cwd, 'grid', 'pw'), pumping_well)  #
    np.save(jp(results_dir, 'pw'), pumping_well)  # Saves pumping well stress period data
    # Run Flow
    flow_model = my_flow(exe_name=exe_name_mf, model_ws=results_dir, wells=wells_data)
    # # Run Transport
    transport_model = my_transport(modflowmodel=flow_model, exe_name=exe_name_mt)
    # Run Modpath
    end_points = my_backtracking(flow_model, exe_name_mp)
    # Compute signed distance
    signed_distance(end_points, results_dir=results_dir)

    # Deletes everything except final results
    for the_file in os.listdir(results_dir):
        if not the_file.endswith('.npy') and not the_file.endswith('.py') and not the_file.endswith('.xy'):
            file_path = os.path.join(results_dir, the_file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    jobs = []
    n_series = 250
    n_jobs = 4
    for j in range(n_series):
        for i in range(n_jobs):  # Can run max 4 instances of mt3dms at once on this computer
            process = Process(target=main)
            jobs.append(process)
            process.start()
        process.join()
        process.close()
