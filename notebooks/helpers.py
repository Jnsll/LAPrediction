import pandas as pd
import os
import gdal
import numpy as np
import math
from osgeo import osr, ogr
import flopy

def get_site_name_from_site_number(site_number):
    sites = pd.read_csv("/DATA/These/Projects/modflops/docker-simulation/modflow/" + 'data/study_sites.txt',
                        sep=',', header=0, index_col=0) #\\s+
    site_name = sites.index._data[site_number]
    return site_name

def get_model_name(site_number, chronicle, approx, rate, ref, perm):
    model_name = "model_time_0_geo_0_thick_1_K_86.4_Sy_0.1_Step1_site" + str(site_number) + "_Chronicle" + str(chronicle)
    if perm:
        model_name += "_SteadyState"
    elif not ref:
        model_name += "_Approx" + str(approx)
        if approx == 0:
            model_name += "_Period" + str(rate)
        elif approx==1:
            model_name += "_RechThreshold" + str(rate)
    return model_name


def get_mask_data_for_a_site(site_number):
    mask_file = os.path.join("/DATA/These/OSUR/Extract_BV_june/", str(site_number) + "_Mask.tif")
    ds = gdal.Open(mask_file)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    mask_array = np.array(ds.GetRasterBand(1).ReadAsArray())
    print("cols mask:", cols)
    print("rows mask:", rows)
    return mask_array, cols, rows


def get_non_dry_cell_hds_value(hds, nrow, ncol, nlayer):
    layer = 0
    h = hds[layer][nrow][ncol]
    while (math.isclose(abs(h)/1e+30, 1, rel_tol=1e-3)) and layer < nlayer:
        if layer == nlayer-1:
            print("cell completely dry")
        else:
            h = hds[layer+1][nrow][ncol]
            layer += 1
    return h

def getWeightToSurface(zs, h, dc, alpha):
    """
        zs : value of soil surface at the point x
        h : head value for watertable at the x point
        dc : critical depth
        alpha : ratio of critical depth for calculating the width of transition zone
    """
    ddc = alpha * dc  # Alpha must be not null

    borneInf = zs - (dc + (ddc / 2))
    borneSup = zs - (dc - (ddc / 2))

    if h <= borneInf:
        Ws = 0
    elif h >= borneSup:
        Ws = 1
    else:
        Ws = math.sin((math.pi * (h - borneInf)) / (2*ddc))

    return Ws

def get_soil_surface_values_for_a_simulation(repo_simu, model_name):
    """
        Retrieve the matrix of the topographical altitude.
    """
    mf = flopy.modflow.Modflow.load(repo_simu + "/" + model_name + '.nam')
    dis = flopy.modflow.ModflowDis.load(
        repo_simu + "/" + model_name + '.dis', mf)
    topo = dis.top._array
    return topo


def get_model_size(coord):
    r_dem = "/DATA/These/Projects/modflops/docker-simulation/modflow/" + "/data/MNT_TOPO_BATH_75m.tif"
    xmin = coord[0]
    xmax = coord[1]
    ymin = coord[2]
    ymax = coord[3]
    dem = gdal.Open(r_dem)
    dem_geot = dem.GetGeoTransform()
    dem_Xpos = np.ones((dem.RasterXSize))
    dem_Ypos = np.ones((dem.RasterYSize))
    for i in range(0, dem.RasterYSize):
        yp = dem_geot[3] + (dem_geot[5] * i)
        dem_Ypos[i] = yp
    for j in range(0, dem.RasterXSize):
        xp = dem_geot[0] + (dem_geot[1] * j)
        dem_Xpos[j] = xp
    ulX = (np.abs(dem_Xpos - xmin)).argmin()
    lrX = (np.abs(dem_Xpos - xmax)).argmin()
    ulY = (np.abs(dem_Ypos - ymax)).argmin()
    lrY = (np.abs(dem_Ypos - ymin)).argmin()
    clip_dem_x = dem_Xpos[ulX:lrX]
    clip_dem_y = dem_Ypos[ulY:lrY]
    return ulY, lrY, ulX, lrX, clip_dem_x, clip_dem_y


def get_clip_dem(coord):
    r_dem = "/DATA/These/Projects/modflops/docker-simulation/modflow/" + "/data/MNT_TOPO_BATH_75m.tif"
    ulY, lrY, ulX, lrX, clip_dem_x, clip_dem_y = get_model_size(coord)
    dem = gdal.Open(r_dem)
    dem_geot = dem.GetGeoTransform()
    dem_data = dem.GetRasterBand(1).ReadAsArray()
    clip_dem = dem_data[ulY:lrY, ulX:lrX]
    return dem_geot, clip_dem_x, clip_dem_y, clip_dem


def save_clip_dem(folder, site_number, chronicle, approx, rate, ref, npy_name, tif_name):
    sites = pd.read_csv("/DATA/These/Projects/modflops/docker-simulation/modflow/" + "/data/study_sites.txt", sep=',', header=0, index_col=0)
    model_name = get_model_name(site_number, chronicle, approx, rate, ref, perm=False)
    site_name = get_site_name_from_site_number(site_number)
    repo_simu = folder + site_name + "/" + model_name

    coord = sites._get_values[site_number, 1:5]
    print(coord)
    geot, geotx, geoty, demData = get_clip_dem(coord)
    drv = gdal.GetDriverByName("GTiff")
    ds = drv.Create(repo_simu + "/" + tif_name, demData.shape[1], demData.shape[0], 1, gdal.GDT_Float32)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2154)
    ds.SetProjection(srs.ExportToWkt())
    gt = [geotx[0], geot[1], 0, geoty[1], 0, geot[5]]
    ds.SetGeoTransform(gt)
    values = np.load(repo_simu + "/" + npy_name)
    ds.GetRasterBand(1).WriteArray(values)