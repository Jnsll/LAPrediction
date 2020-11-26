import os
import gdal
import numpy as np
import argparse
import csv
import pandas as pd


def get_site_name_from_site_number(site_number):
    sites = pd.read_csv(
        "/DATA/These/Projects/modflops/docker-simulation/modflow/"
        + "data/study_sites.txt",
        sep=",",
        header=0,
        index_col=0,
    )  # \\s+
    site_name = sites.index._data[site_number] + "/"
    return site_name


def compute_feature_CV_HV(site_number):

    mask_array, mask_ncol, mask_nrow = get_mask_data_for_a_site(site_number)
    coastal_vulne = 0
    hydro_vulne = 0
    for nrow in range(mask_nrow):
        for ncol in range(mask_ncol):
            if mask_array[nrow][ncol] == 1:
                coastal_vulne += 1
            elif mask_array[nrow][ncol] == 2:
                hydro_vulne += 1
    print(coastal_vulne)
    print(hydro_vulne)

    site_name = get_site_name_from_site_number(site_number)

    with open(
        "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/"
        + str(site_name)
        + "/"
        + "Feature_CV_HV_Site_"
        + str(site_number)
        + "_BVE.csv",
        "w",
    ) as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(["Site", "Coastal Vulnerability", "Hydrological Vulnerability"])
        writer.writerow([site_number, coastal_vulne, hydro_vulne])


def get_mask_data_for_a_site(site_number):
    mask_file = os.path.join(
        "/DATA/These/Projects/modflops/docker-simulation/modflow/docker-simulation/modflow/data/Masks/", str(site_number) + "_basins.tif"
    )   #"/DATA/These/OSUR/Extract_BV_june/", str(site_number) + "_ZV_C1_H2_Mask.tif"
    print(mask_file)
    ds = gdal.Open(mask_file)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    mask_array = np.array(ds.GetRasterBand(1).ReadAsArray())
    print("cols mask:", cols)
    print("rows mask:", rows)
    return mask_array, cols, rows


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-site", "--sitenumber", type=int, required=True)

    args = parser.parse_args()

    site_number = args.sitenumber

    compute_feature_CV_HV(site_number)
