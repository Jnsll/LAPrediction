import pandas as pd
import matplotlib.pyplot as plt


#Global Variables
MYDIR = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/ZLearning/"
SITES_NUMBERS = range(1,41)

def concat_prediction_HError_results_with_subcatchment_for_all_sites(approx=0, chronicle=0, permeability=27.32):
    
    frames = []
    for site_number in SITES_NUMBERS:
        file = (
            MYDIR
            + "Approx"
            + str(approx)
            + "/Chronicle"
            + str(chronicle)
            + "/SiteTest"
            + str(site_number)
            + "/"
           + "Prediction_HErrorValues_SubCatch_Chronicle"
            + str(chronicle)
            + "_Approx"
            + str(approx)
            + "_K"
            + str(permeability)
            + "_Slope_Elevation_LC_SAR_Area_CV_HV.csv"
        )
        try:
            dfp = pd.read_csv(file, sep=";")
        except:
            continue
        frames.append(dfp)
        df = pd.concat(frames)
    df.to_csv(MYDIR + "Prediction_HErrorValues_SubCatch_Chronicle" + str(chronicle) + "_Approx"
                    + str(approx)
                    + "_K"
                    + str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV.csv", index=False)
    print(
        "File '"
        + "Prediction_HErrorValues_SubCatch_Chronicle" + str(chronicle) + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV.csv"
        + "' has been created."
    )

def concat_prediction_pmax_results_with_subcatchment_for_all_sites(approx=0, chronicle=0, permeability=27.32):
    
    frames = []
    for site_number in SITES_NUMBERS:
        file = (
            MYDIR
            + "Approx"
            + str(approx)
            + "/Chronicle"
            + str(chronicle)
            + "/SiteTest"
            + str(site_number)
            + "/"
           + "Prediction_PMax_SubCatch_Chronicle"
            + str(chronicle)
            + "_Approx"
            + str(approx)
            + "_K"
            + str(permeability)
            + "_Slope_Elevation_LC_SAR_Area_CV_HV.csv"
        )
        try:
            dfp = pd.read_csv(file, sep=";")
        except:
            continue
        frames.append(dfp)
        df = pd.concat(frames)
    df.to_csv(MYDIR + "Prediction_PMax_SubCatch_Chronicle" + str(chronicle) + "_Approx"
                    + str(approx)
                    + "_K"
                    + str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV.csv", index=False)
    print(
        "File '"
        + "Prediction_PMax_SubCatch_Chronicle" + str(chronicle) + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability) + "_AllSites_Slope_Elevation_LC_SAR_Area_CV_HV.csv"
        + "' has been created."
    )







def concat_prediction_pmax_results_for_all_sites():
    approximations = [0]
    chronicles = [0]
    permeability=27.32
    # df = pd.DataFrame(columns=["Site", "ExecTimeSum"])
    frames = []
    for approx in approximations:
        for chronicle in chronicles:
            for test_site in sites:
                file = (
                    MYDIR
                    + "Approx"
                    + str(approx)
                    + "/Chronicle"
                    + str(chronicle)
                    + "/SiteTest"
                    + str(test_site)
                    + "/"
                    + "Prediction_HError_Basic_Chronicle"
                    + str(chronicle)
                    + "_Approx"
                    + str(approx)
                    + "_K"
                    + str(permeability)
                    + "_BVE_CVHV_Geomorph_Sat_Extend.csv"
                )
                try:
                    dfp = pd.read_csv(file, sep=";")
                except:
                    continue
                frames.append(dfp)
                df = pd.concat(frames)
    df.to_csv(MYDIR + "Pred_Chronicle" + str(chronicle) + "_Approx"
                    + str(approx)
                    + "_K"
                    + str(permeability) + "_AllSites__BVE_CVHV_Geomorph_Sat_Extend.csv", index=False)
    print(
        "File '"
        + "Pred_Chronicle" 
        + str(chronicle) 
        + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability) + "_AllSites__BVE_CVHV_Geomorph_Sat_Extend.csv"
        + "' has been created."
    )

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
    concat_prediction_pmax_results_with_subcatchment_for_all_sites()
    concat_prediction_pmax_results_with_subcatchment_for_all_sites()