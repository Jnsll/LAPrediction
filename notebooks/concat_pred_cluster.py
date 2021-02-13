import pandas as pd
import matplotlib.pyplot as plt
import argparse

#Global Variables
MYDIR = "output/"
SITES_NUMBERS = range(1,41)
SUB_NUMBERS = range(1,31)

def concat_prediction_HError_results_with_subcatchment_for_all_sites_by_cluster(nb_clusters, std, approx=0, chronicle=0, permeability=27.32):
    
    frames = []
    for site_number in SITES_NUMBERS:
        for subcatch_number in SUB_NUMBERS:

            file_name = "Prediction_HErrorValues_SubCatch" + str(subcatch_number) + "_by_cluster" + str(nb_clusters) + "_Chronicle" + str(chronicle) + "_Approx" + str(approx) + "_K" + str(permeability)+ "_Slope_Elevation_LC_SAR_Area_CV_HV"
            if std:
                file_name += "_std.csv"
            else:
                file_name += ".csv"

            file = (
                MYDIR
                + "Approx"
                + str(approx)
                + "/Chronicle"
                + str(chronicle)
                + "/SiteTest"
                + str(site_number)
                + "/"
                + file_name
            )
            #print("file", file)
            try:
                dfp = pd.read_csv(file, sep=";")
            except:
                continue
            frames.append(dfp)
            df = pd.concat(frames)
    file_save = "Prediction_HErrorValues_SubCatch_by_cluster" + str(nb_clusters) + "_Chronicle" + str(chronicle) + "_Approx" + str(approx) + "_K" + str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV"
    if std:
        file_save += "_std.csv"
    else:
        file_save += ".csv"
    df.to_csv(MYDIR + file_save, index=False)
    print(
        "File '"
        + file_save
        + "' has been created."
        )

def concat_prediction_pmax_results_with_subcatchment_for_all_sites_by_cluster(nb_clusters, std, approx=0, chronicle=0, permeability=27.32):
    
    frames = []
    for site_number in SITES_NUMBERS:
        for subcatch_number in SUB_NUMBERS:
            file_name = "Prediction_PMax_SubCatch" + str(subcatch_number) + "_by_cluster" + str(nb_clusters) + "_Chronicle" + str(chronicle) + "_Approx" + str(approx) + "_K" + str(permeability) + "_Slope_Elevation_LC_SAR_Area_CV_HV"
            if std:
                file_name += "_std.csv"
            else:
                file_name += ".csv"
            file = (
                MYDIR
                + "Approx"
                + str(approx)
                + "/Chronicle"
                + str(chronicle)
                + "/SiteTest"
                + str(site_number)
                + "/"
                + file_name
            )
            #print("file", file)
            try:
                dfp = pd.read_csv(file, sep=";")
                #print("dfp", dfp)
            except:
                continue
            frames.append(dfp)
            #print("frames", frames)
            df = pd.concat(frames)
    file_save = "Prediction_PMax_SubCatch_by_cluster" + str(nb_clusters) + "_Chronicle" + str(chronicle) + "_Approx" + str(approx) + "_K"+ str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV"
    if std:
        file_save += "_std.csv"
    else:
        file_save += ".csv"
    df.to_csv(MYDIR + file_save, index=False)
    print(
        "File '"
        + file_save
        + "' has been created."
        )







# def concat_prediction_pmax_results_for_all_sites(nb_clusters):
#     approximations = [0]
#     chronicles = [0]
#     permeability=27.32
#     # df = pd.DataFrame(columns=["Site", "ExecTimeSum"])
#     frames = []
#     for approx in approximations:
#         for chronicle in chronicles:
#             for test_site in sites:
#                 file = (
#                     MYDIR
#                     + "Approx"
#                     + str(approx)
#                     + "/Chronicle"
#                     + str(chronicle)
#                     + "/SiteTest"
#                     + str(test_site)
#                     + "/"
#                     + "Prediction_HError_Basic_Chronicle"
#                     + str(chronicle)
#                     + "_Approx"
#                     + str(approx)
#                     + "_K"
#                     + str(permeability)
#                     + "_BVE_CVHV_Geomorph_Sat_Extend.csv"
#                 )
#                 try:
#                     dfp = pd.read_csv(file, sep=";")
#                 except:
#                     continue
#                 frames.append(dfp)
#                 df = pd.concat(frames)
#     df.to_csv(MYDIR + "Pred_Chronicle" + str(chronicle) + "_Approx"
#                     + str(approx)
#                     + "_K"
#                     + str(permeability) + "_AllSites__BVE_CVHV_Geomorph_Sat_Extend.csv", index=False)
#     print(
#         "File '"
#         + "Pred_Chronicle" 
#         + str(chronicle) 
#         + "_Approx"
#         + str(approx)
#         + "_K"
#         + str(permeability) + "_AllSites__BVE_CVHV_Geomorph_Sat_Extend.csv"
#         + "' has been created."
#     )

# df_sort=df.sort_values(by=['R2 Test Test Site'])
# print(df_sort)
# df.plot(kind="scatter", x="Test Site", y="R2 Test")
# plt.savefig(
#     MYDIR + "/All_Relation_Site_Deter_coef_Chronicle" + str(chronicle) 
#     + "_Approx"
#     + str(approx)
#     + "_K"
#     + str(permeability) 
#     + "_BVE_CVHV_Geomorph_Sat_Extend.png"
# )
# plt.clf()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-clusters", "--clusters", type=int, required=True)
    parser.add_argument("-std", action="store_true")
    args = parser.parse_args()
    
    nb_clusters = args.clusters
    std = args.std

    concat_prediction_pmax_results_with_subcatchment_for_all_sites_by_cluster(nb_clusters,std)
    concat_prediction_HError_results_with_subcatchment_for_all_sites_by_cluster(nb_clusters,std)