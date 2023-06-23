# coding:utf-8
import os,sys
import pandas as pd
import threading
import subprocess
import argparse

# Custom imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_modflow'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_modpath'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'model_seawat'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vtk_export_grid'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vtk_export_watertable'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'vtk_export_pathlines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'custom_utils'))
from model_modflow import model_modflow as modflow
from model_modpath import model_modpath as modpath
from model_seawat import model_seawat as seawat
from vtk_export_grid import vtk_export_grid as vtk_grid
from vtk_export_watertable import vtk_export_watertable as vtk_watertable
from vtk_export_pathlines import vtk_export_pathlines as vtk_pathlines
from custom_utils import helpers as utils

########################################################################################################################
#                                           GLOBAL MODEL PARAMETERS SETTINGS                                           #
########################################################################################################################

# FOLDER
folder_path = os.path.dirname(os.path.abspath(__file__)) + '/'
print("folder_path : ", folder_path)

# STUDY SITES
print("folder + string : ", folder_path + "data/study_sites.txt")
sites = pd.read_table(folder_path + "data/study_sites.txt", sep=',', header=0, index_col=0)

# Hydrological parameters
permeability = [8.64] 
theta = [0.1]   # Porosity in %
geology = [0]   # 0: homogeneous geology | 1: heterogeneous geology
time = [0]

def setting(permeability, time, geology, theta, input_file, step, ref, chronicle, approx, rate, rep, steady, site=2):
    site_number = site
    row_site = sites.loc[sites['number']==site_number]
    coordinates = [row_site.iloc[0]["xmin"], row_site.iloc[0]["xmax"],row_site.iloc[0]["ymin"], row_site.iloc[0]["ymax"]]


    # TIME DISCRETIZATION
    time_param = time

    # STRUCTURE
    geology_param = geology  
    permeability_param = permeability   # m^2/d | only if geology_param = 0
    theta_param = theta                 # Porosity in %
    thickness_param = 1                 # 0: homogeneous thickness | 1: flat bottom (heterogeneous thickness)

    # MODEL NAME
    print("site_number :", site_number)
    print("site name formule :", sites.index._data[site_number])
    site_name = sites.index._data[site_number] + '/'
    model_name = utils.generate_model_name(chronicle, approx, rate, ref, steady, site, permeability_param=permeability)
    if rep:
        model_name = model_name + "_" + str(rep)

    model_folder = model_name + '/'
    
    if (input_file is None):
        if ref:
            chronicle_file = pd.read_table(folder_path + "data/chronicles.txt", sep=',', header=0, index_col=0)
            input_file = chronicle_file.template[chronicle]

        else:
            input_file = utils.get_input_file_name(chronicle, approx, rate, ref, steady, site=None, step=None)
        
    
    # SIMULATION
    modflow_param = 1  # 0: disabled | 1: enabled
    seawat_param = 0  # 0: disabled | 1: enabled
    if time ==1:
        modpath_param = 1  # 0: disabled | 1: enabled
    else:
        modpath_param = 0
    # VTK OUTPUT
    grid = 0  # 0: disabled | 1: enabled
    watertable = 0  # 0: disabled | 1: enabled
    pathlines = 0 # 0: disabled | 1: enabled

    # CREATE AND RUN MODFLOW MODEL - SEAWAT MODEL - MODPATH MODEL
    if modflow_param == 1:
        print("file :" + input_file)
        print("site_name : ", site_name)
        print("folder_path : ", folder_path)
        print("model_folder : ", model_folder)
        modflow(input_file, file_name=folder_path, model_name=model_name, model_folder=folder_path + "outputs/" + site_name + model_folder,
                coord=coordinates, tdis=time_param, geo=geology_param, permea=permeability_param, thick=thickness_param, port=int(row_site["port_number"]), porosity=theta_param, ref=ref)
    if seawat_param == 1:
        seawat(filename=folder_path,modelfolder=folder_path + site_name + model_folder, modelname=model_name)
    if modpath_param == 1:
        modpath(filename=folder_path, modelname=model_name + '_swt', modelfolder=folder_path + site_name + model_folder)

    # CREATE OUTPUT FILES
    if not os.path.exists(folder_path + "outputs/" + site_name + model_folder + 'output_files'):
        os.makedirs(folder_path+ "outputs/" + site_name + model_folder + 'output_files')
    if grid == 1:
        vtk_grid(modelname=model_name, modelfolder=folder_path + "outputs/" + site_name + model_folder, coord=coordinates)
    if watertable == 1:
        vtk_watertable(modelname=model_name, modelfolder=folder_path + "outputs/" + site_name + model_folder, coord=coordinates)
    if pathlines == 1:
        vtk_pathlines(modelname=model_name + '_swt', modelfolder=folder_path + "outputs/" + site_name + model_folder, coord=coordinates)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputfile", type=str, required=False)
    parser.add_argument("-ts", "--timestep", type=int, required=False)
    parser.add_argument("-ref", "--reference", action='store_true')
    parser.add_argument("-chr", "--chronicle", type=int, required=True)
    parser.add_argument("-site", "--site", type=int, required=False)
    parser.add_argument("-rate", "--rate", type=float, required=False)
    parser.add_argument("-approx", '--approximation', type=int, required=False)
    parser.add_argument("-rep", '--rep', type=int, required=False)
    parser.add_argument("-perm", "--permeability", type=float, required=False)
    parser.add_argument("-sd", "--steady", type=int, required=False)
    args = parser.parse_args()

    input_file = args.inputfile
    step = args.timestep
    rate = args.rate
    reference = args.reference
    site=args.site
    chronicle = args.chronicle
    approx = args.approximation
    rep=args.rep
    perm = args.permeability
    steady=args.steady 

    if rep==0:
        rep=None

    if site:
        if perm:
            setting(perm, time[0], geology[0], theta[0], input_file, step, reference, chronicle, approx, rate, rep, steady, site=site)
        else:
            setting(permeability[0], time[0], geology[0], theta[0], input_file, step, reference, chronicle, approx, rate, rep, steady, site=site)
    else:
        if perm:
            setting(perm, time[0], geology[0], theta[0], input_file, step, reference, chronicle, approx, rate, rep, steady)
        else:
            setting(permeability[0], time[0], geology[0], theta[0], input_file, step, reference, chronicle, approx, rate, rep, steady)
