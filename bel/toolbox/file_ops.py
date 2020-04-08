import os
from os.path import join as jp
import shutil
import numpy as np
import flopy


class FileOps:

    def __init__(self):
        pass

    @staticmethod
    def load_res(res_dir, n=0, check=True, roots=None):
        """

        @param check: Whether to check breaktrough curves for unrealistic results or not.
        @param res_dir: main directory containing results sub-directories
        @param n: if != 0, will randomly select n sub-folders from res_dir
        @return: tp, sd
        """
        # TODO: Split this function to generalize and load one feature at a time

        bkt_files = []  # Breakthrough curves files
        sd_files = []  # Signed-distance files
        hk_files = []  # Hydraulic conductivity files
        # r=root, d=directories, f = files
        if not roots:
            roots = []
            for r, d, f in os.walk(res_dir, topdown=False):
                # Adds the data files to the lists, which will be loaded later
                if 'bkt.npy' in f and 'pz.npy' in f and 'hk0.npy' in f:
                    bkt_files.append(jp(r, 'bkt.npy'))
                    sd_files.append(jp(r, 'pz.npy'))
                    hk_files.append(jp(r, 'hk0.npy'))
                    roots.append(r)
                else:  # If one of the files is missing, deletes the sub-folder
                    try:
                        if r != res_dir:  # Make sure to not delete the main results directory !
                            shutil.rmtree(r)
                    except TypeError:
                        pass
            if n:
                folders = np.random.choice(np.arange(len(roots)), n)  # Randomly selects n folders
                bkt_files = np.array(bkt_files)[folders]
                sd_files = np.array(sd_files)[folders]
                hk_files = np.array(hk_files)[folders]
                roots = np.array(roots)[folders]
        else:
            [bkt_files.append(jp(res_dir, r, 'bkt.npy')) for r in roots]
            [sd_files.append(jp(res_dir, r, 'pz.npy')) for r in roots]
            [hk_files.append(jp(res_dir, r, 'hk0.npy')) for r in roots]

        # Load and filter results
        # We first load the breakthrough curves and check for errors.
        # If one of the concentration is > 1, it means the simulation is unrealistic and we remove it from the dataset.
        if check:  # Check whether to delete for simulation issues
            tpt = list(map(np.load, bkt_files))
            rm = []  # Will contain indices to remove
            for i in range(len(tpt)):
                for j in range(len(tpt[i])):
                    if max(tpt[i][j][:, 1]) > 1:  # Check results files whose max computed head is > 1 and removes them
                        rm.append(i)
                        break
            for index in sorted(rm, reverse=True):
                del bkt_files[index]
                del sd_files[index]
                del hk_files[index]
                del roots[index]

        # Check unpacking
        tpt = list(map(np.load, bkt_files))  # Re-load transport curves
        # hk = np.array(list(map(np.load, hk_files)))  # Load hydraulic K
        sd = np.array(list(map(np.load, sd_files)))  # Load signed distance
        return tpt, sd, roots

    @staticmethod
    def load_data(res_dir, n=0, check=True, data_flag=False):
        """

        @param check: Whether to check breaktrough curves for unrealistic results or not.
        @param res_dir: main directory containing results sub-directories
        @param n: if != 0, will randomly select n sub-folders from res_dir
        @param data_flag: if True, only loads data and not the target
        @return: tp, sd
        """
        # TODO: Split this function to generalize and load one feature at a time

        bkt_files = []  # Breakthrough curves files
        sd_files = []  # Signed-distance files
        hk_files = []  # Hydraulic conductivity files
        # r=root, d=directories, f = files
        roots = []
        for r, d, f in os.walk(res_dir, topdown=False):
            # Adds the data files to the lists, which will be loaded later
            if 'bkt.npy' in f and 'sd.npy' in f and 'hk0.npy' in f:
                bkt_files.append(jp(r, 'bkt.npy'))
                sd_files.append(jp(r, 'sd.npy'))
                hk_files.append(jp(r, 'hk0.npy'))
                roots.append(r)
            else:  # If one of the files is missing, deletes the sub-folder
                try:
                    if r != res_dir:  # Make sure to not delete the main results directory !
                        shutil.rmtree(r)
                except TypeError:
                    pass
        if n:
            folders = np.random.choice(np.arange(len(roots)), n)  # Randomly selects n folders
            bkt_files = np.array(bkt_files)[folders]
            sd_files = np.array(sd_files)[folders]
            hk_files = np.array(hk_files)[folders]
            roots = np.array(roots)[folders]

        # Load and filter results
        # We first load the breakthrough curves and check for errors.
        # If one of the concentration is > 1, it means the simulation is unrealistic and we remove it from the dataset.
        if check:  # Check whether to delete for simulation issues
            tpt = list(map(np.load, bkt_files))
            rm = []  # Will contain indices to remove
            for i in range(len(tpt)):
                for j in range(len(tpt[i])):
                    if max(tpt[i][j][:, 1]) > 1:  # Check results files whose max computed head is > 1 and removes them
                        rm.append(i)
                        break
            for index in sorted(rm, reverse=True):
                del bkt_files[index]
                del sd_files[index]
                del hk_files[index]
                del roots[index]

        # Check unpacking
        tpt = list(map(np.load, bkt_files))  # Re-load transport curves
        # hk = np.array(list(map(np.load, hk_files)))  # Load hydraulic K
        if not data_flag:
            sd = np.array(list(map(np.load, sd_files)))  # Load signed distance
            return tpt, sd, roots
        else:
            return tpt, roots

    @staticmethod
    def datread(file=None, header=0):
        """Reads space separated dat file"""
        with open(file, 'r') as fr:
            op = np.array([list(map(float, l.split())) for l in fr.readlines()[header:]])
        return op

    @staticmethod
    def dirmaker(dird):
        """
        Given a folder path, check if it exists, and if not, creates it
        :param dird: path
        :return:
        """
        try:
            if not os.path.exists(dird):
                os.makedirs(dird)
        except:
            pass

    @staticmethod
    def load_flow_model(nam_file, exe_name='', model_ws=''):
        flow_loader = flopy.modflow.mf.Modflow.load

        return flow_loader(f=nam_file, exe_name=exe_name, model_ws=model_ws)

    @staticmethod
    def load_transport_model(nam_file, modflowmodel, exe_name='', model_ws='', ftl_file='mt3d_link.ftl',
                             version='mt3d-usgs'):
        transport_loader = flopy.mt3d.Mt3dms.load
        transport_reloaded = transport_loader(f=nam_file, version=version, modflowmodel=modflowmodel,
                                              exe_name=exe_name, model_ws=model_ws)
        transport_reloaded.ftlfilename = ftl_file

        return transport_reloaded
