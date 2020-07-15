
import helpers
import flopy.utils.binaryfile as fpu
import argparse
import numpy as np


def get_evolution_of_head_level_for_a_cell(folder, site_number, chronicle, approx, rate, row, col, ref):
    simu_name = helpers.get_model_name(site_number, chronicle, approx, rate, ref=ref, perm=False)
    site_name = helpers.get_site_name_from_site_number(site_number)
    repo_simu = folder + site_name + "/" + simu_name
    topoSimu = helpers.get_soil_surface_values_for_a_simulation(repo_simu, simu_name)
    simu_hds = fpu.HeadFile(repo_simu + '/' + simu_name + '.hds')


    mask_array, mask_ncol, mask_nrow = helpers.get_mask_data_for_a_site(site_number)
    if np.isnan(mask_array[row][col]):
        print("cell row " + str(row) + " and col " + str(col) + " is not in BVE.")
        return

    endTime = int(simu_hds.get_times()[-1])#15341
    #print(endTime)


    evol_head_cell = []
    evol_depth_cell = []

    ztopo = topoSimu[row][col]
    if ref:
        for day in range(endTime):
            simu_head = simu_hds.get_data(kstpkper=(0, day)) # 0 because timestep = 1 and 1-1 = 0 (0-based numbers)
            h = helpers.get_non_dry_cell_hds_value(simu_head, row, col, simu_head.shape[0])
            evol_depth_cell.append(h - ztopo)
            evol_head_cell.append(h)
        np.save(repo_simu + "/Evol_Depth_Cell_Row" + str(row) + "_Col" + str(col) + "_Site_" + str(site_name) + "_Chronicle" + str(chronicle) + ".npy", evol_depth_cell)
        print("Repo: ", repo_simu)
    else:
        print("The function has not been implemented for alternative simulations yet!")
    

def create_map_cell_localisation(folder, site_number, chronicle, approx, rate, ref, row, col):
    mask_array, mask_ncol, mask_nrow = helpers.get_mask_data_for_a_site(site_number)
    values = np.zeros(shape=(mask_nrow, mask_ncol))
    values[row][col] = 1
    site_name = helpers.get_site_name_from_site_number(site_number)
    helpers.save_clip_dem(folder, site_number, chronicle, approx=None, rate=None, ref=True, values=values, tif_name="Evol_Depth_Cell_Row" + str(row) + "_Col" + str(col) + "_Site_" + str(site_name) + "_Chronicle" + str(chronicle) + ".tif")
    print("Map created")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-site", "--sitenumber", type=int, required=True)
    parser.add_argument("-chr", "--chronicle", type=int, required=True)
    parser.add_argument("-approx", "--approx", type=int, required=False)
    parser.add_argument("-rate", "--rate", type=int, required=False)
    parser.add_argument("-f", "--folder", type=str, required=True)
    parser.add_argument("-r", "--row", type=int, required=True)
    parser.add_argument("-c", "--col", type=int, required=True)
    parser.add_argument("-m", "--map", action='store_true')
    parser.add_argument("-ref", "-ref", action='store_true')
    args = parser.parse_args()

    site_number = args.sitenumber
    chronicle = args.chronicle
    approx = args.approx
    rate = args.rate
    folder = args.folder
    row = args.row
    col = args.col
    ref = args.ref
    m = args.map

    
    if m:
        create_map_cell_localisation(folder, site_number, chronicle, approx, rate, ref, row, col)
    else:
        get_evolution_of_head_level_for_a_cell(folder, site_number, chronicle, approx, rate, row, col, ref)