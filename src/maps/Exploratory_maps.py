from helpers import helpers
import numpy as np
import flopy.utils.binaryfile as fpu
import argparse

"""
TODO
    Note : For now it is written for reference simulations only!

"""

NB_YEARS = 42
PERIODS = [365, 365, 366, 365]
CYCLE = len(PERIODS)

def compute_depth_with_stationary_state(folder, site_number, chronicle, permeability):
    
    ref_name = helpers.get_model_name(site_number, chronicle, approx=None, rate=None, ref=True, steady=False, permeability=86.4)
    ss_name = helpers.get_model_name(site_number, chronicle, approx=None, rate=None, ref=True, steady=True, permeability=permeability)
    site_name = helpers.get_site_name_from_site_number(site_number)
    repo_ref = folder + site_name + "/" + ref_name
    repo_ss = folder + site_name + "/" + ss_name

    # Topography file
    topo_ref = np.load(repo_ref + "/soil_surface_topo_"+ ref_name + ".npy")
    # Watertable altitude for stationary state
    ss_hds = fpu.HeadFile(repo_ss + '/' + ss_name + '.hds')

    #kstp = ss_hds.get_kstpkper()
    
    ss_head = ss_hds.get_data(kstpkper=(0, 0))
    #print(ss_head)

    nbrowtot = ss_head.shape[1]
    nbcoltot = ss_head.shape[2]

    depth_values = np.full((nbrowtot, nbcoltot), -9)

        # mask to only get the data of the "equivalent watershed"
    mask_array, mask_ncol, mask_nrow = helpers.get_mask_data_for_a_site(site_number)
    if mask_nrow == nbrowtot:
        print("same number of rows")
    else:
        print("Not same number of rows!")
    if mask_ncol == nbcoltot:
        print("Same number of cols")
    else:
        print("Not same number of columns!")

    for nrow in range(nbrowtot):
        for ncol in range(nbcoltot):
            # Mask of BVE
            if np.isnan(mask_array[nrow][ncol]):
                continue
            s = helpers.get_non_dry_cell_hds_value(ss_head, nrow, ncol, ss_head.shape[0])
            d = topo_ref[nrow][ncol] - s
            depth_values[nrow][ncol] = d

    np.save(repo_ref + "/Depth_Map_StationaryState_Site_" + str(site_name) + "_Chronicle" + str(chronicle) + "_Permeability" + str(permeability) + ".npy", depth_values)




def create_depth_map(folder, site_number, chronicle, permeability):
    site_name = helpers.get_site_name_from_site_number(site_number)
    npy_name = "Depth_Map_StationaryState_Site_" + str(site_name) + "_Chronicle" + str(chronicle) + "_Permeability" + str(permeability) + ".npy"
    tif_name = "Depth_Map_StationaryState_Site_" + str(site_name) + "_Chronicle" + str(chronicle) + "_Permeability" + str(permeability) + "_MNT.tif"
    helpers.save_clip_dem(folder, site_number, chronicle, approx=None, rate=None, ref=True, npy_name=npy_name, tif_name=tif_name, permeability=permeability)
    print("Map created")



def compute_and_create_depth_map(folder, site_number, chronicle, permeability):
    compute_depth_with_stationary_state(folder, site_number, chronicle, permeability)
    create_depth_map(folder, site_number, chronicle, permeability)


def compute_amplitude(folder, site_number, chronicle):
    
    ref_name = helpers.get_model_name(site_number, chronicle, approx=None, rate=None, ref=True, steady=False, permeability=None)
    site_name = helpers.get_site_name_from_site_number(site_number)
    print("site_name: ", site_name)
    repo_ref = folder + site_name + "/" + ref_name

    ref_hds = fpu.HeadFile(repo_ref + '/' + ref_name + '.hds')
    refHead_init = ref_hds.get_data(kstpkper=(0, 0))
    nbrowtot = refHead_init.shape[1]
    nbcoltot = refHead_init.shape[2]

    endTime = int(ref_hds.get_times()[-1])
    #print(endTime)

    hmin_values = np.full((nbrowtot, nbcoltot), -9)
    hmax_values = np.full((nbrowtot, nbcoltot), -9)
    dh = np.zeros(shape=(nbrowtot, nbcoltot))
    mask_array, mask_ncol, mask_nrow = helpers.get_mask_data_for_a_site(site_number)

    compt_year = 1
    compt_days = 0

    for day in range(endTime):
        
        if day == 0:
            ref_head = refHead_init
            for nrow in range(nbrowtot):
                for ncol in range(nbcoltot):
                    # Mask of BVE
                    if np.isnan(mask_array[nrow][ncol]):
                        continue
                    h = helpers.get_non_dry_cell_hds_value(ref_head, nrow, ncol, ref_head.shape[0])
                    hmin_values[nrow][ncol] = h
                    hmax_values[nrow][ncol] = h
        else:
            ref_head = ref_hds.get_data(kstpkper=(0, day))
            for nrow in range(nbrowtot):
                for ncol in range(nbcoltot):
                    # Mask of BVE
                    if np.isnan(mask_array[nrow][ncol]):
                        continue
                    h = helpers.get_non_dry_cell_hds_value(ref_head, nrow, ncol, ref_head.shape[0])
                    #print("h", h)
                    if h < hmin_values[nrow][ncol]:
                        hmin_values[nrow][ncol] = h
                    elif h > hmax_values[nrow][ncol]:
                        hmax_values[nrow][ncol] = h
        ind = compt_year%CYCLE
        #print("ind", ind)
        if ind == 0:
            if day == (compt_days  + PERIODS[CYCLE-1] - 1): # the first day is 0
                print("day: ", day)
                for nrow in range(nbrowtot):
                    for ncol in range(nbcoltot):
                        dh[nrow][ncol] += (hmax_values[nrow][ncol] - hmin_values[nrow][ncol])
                        #print("hmax_values[nrow][ncol]: ", hmax_values[nrow][ncol], "hmin_values[nrow][ncol]", hmin_values[nrow][ncol])

                compt_days += PERIODS[CYCLE-1]
                compt_year += 1
        else:
            if day == (compt_days  + PERIODS[ind-1] - 1):
                print("day: ", day)
                for nrow in range(nbrowtot):
                    for ncol in range(nbcoltot):
                        dh[nrow][ncol] += (hmax_values[nrow][ncol] - hmin_values[nrow][ncol])
                        #print("hmax_values[nrow][ncol]: ", hmax_values[nrow][ncol], "hmin_values[nrow][ncol]", hmin_values[nrow][ncol])
                
                compt_days += PERIODS[ind-1]
                compt_year += 1
            
        if day == endTime-1:
            #print("last day, division")
            dh /= NB_YEARS
        
    np.save(repo_ref + "/Amp_Map_Site_" + str(site_name) + "_Chronicle" + str(chronicle) + ".npy", dh)


def create_amp_map(folder, site_number, chronicle, permeability):
    site_name = helpers.get_site_name_from_site_number(site_number)
    npy_name = "Amp_Map_Site_" + str(site_name) + "_Chronicle" + str(chronicle) + ".npy"
    tif_name = "Amp_Map_Site_" + str(site_name) + "_Chronicle" + str(chronicle) + "_MNT.tif"
    helpers.save_clip_dem(folder, site_number, chronicle, approx=None, rate=None, ref=True, npy_name=npy_name, tif_name=tif_name, permeability=permeability)
    print("Map created")


def compute_and_create_amp_map(folder, site_number, chronicle, permeability):
    compute_amplitude(folder, site_number, chronicle)
    create_amp_map(folder, site_number, chronicle, permeability)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-site", "--sitenumber", type=int, required=True)
    parser.add_argument("-chr", "--chronicle", type=int, required=True)
    parser.add_argument("-f", "--folder", type=str, required=True)
    parser.add_argument("-d", "--depth", action='store_true')
    parser.add_argument("-a", "--amp", action='store_true')
    parser.add_argument("-perm", "--permeability", type=float, required=False)
    args = parser.parse_args()

    site_number = args.sitenumber
    chronicle = args.chronicle
    folder = args.folder
    depth = args.depth
    amp = args.amp
    perm = args.permeability


    if depth:
        compute_and_create_depth_map(folder, site_number, chronicle, perm)
    elif amp:
        compute_and_create_amp_map(folder, site_number, chronicle, perm)
    # 