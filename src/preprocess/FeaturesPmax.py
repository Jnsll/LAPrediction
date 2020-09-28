import pandas as pd
import csv
import argparse

# Variables
# permeability=27.32
# site=1
# chronicle=0
# approx=0
folder = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results"

# Utils 
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

def get_model_name(site_number, chronicle, approx, rate, ref, steady, permeability=86.4):
    model_name = "model_time_0_geo_0_thick_1_K_"+ str(permeability) + "_Sy_0.1_Step1_site" + str(site_number) + "_Chronicle" + str(chronicle)
    if steady:
        model_name += "_SteadyState"
    elif not ref:
        model_name += "_Approx" + str(approx)
        if approx == 0 or approx == 2:
            model_name += "_Period" + str(rate)
        elif approx==1:
            model_name += "_RechThreshold" + str(rate)
    return model_name



# File with features about Saturation
def get_df_saturation_features(site_number, chronicle, permeability):
    model_name = "model_time_0_geo_0_thick_1_K_" + str(permeability) + "_Sy_0.1_Step1_site" + str(site_number) + "_Chronicle" + str(chronicle) + "_SteadyState"
    file_name = model_name + "_extracted_features.csv"
    site_name = get_site_name_from_site_number(site_number)

    df_sat_feats = pd.read_csv(folder + "/" + site_name + "/" + model_name  + "/" + file_name, sep=";")
    return df_sat_feats


# Features about geomorphy
def get_df_geomorph_features(site_number):
    df_geomorph = pd.DataFrame(columns=["Site", "Slope", "Elevation", "LC", "CW", "Area"])

    with open(
        "/DATA/These/OSUR/Extract_BV_june/" + str(site_number) + "_slope_elevation_Lc_Cw_A",
        newline="",
    ) as f:
        reader = csv.reader(f)
        geomorph = list(reader)

    slope = float(geomorph[0][0])
    elev = float(geomorph[0][1])
    Lc = float(geomorph[0][2])
    Cw = float(geomorph[0][3])
    A = float(geomorph[0][4])

    df_geomorph = df_geomorph.append(
        {
            "Site": int(site_number),
            "Slope": slope,
            "Elevation": elev,
            "LC": Lc,
            "CW": Cw,
            "Area": A,
        },
        ignore_index=True,
    )

    return df_geomorph

# File with Cv & HV
def get_df_vulnerability_features(site_number):
    site_name = get_site_name_from_site_number(site_number)

    df_vulne = pd.read_csv(
    "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/"
            + str(site_name)
            + "/"
            + "Feature_CV_HV_Site_"
            + str(site_number)
            + "_BVE.csv",
        sep=";"
    )
    return df_vulne




def get_H_Errors_for_a_site(site_number, chronicle, approx, permeability):
    site_name = get_site_name_from_site_number(site_number)
    ref_name = get_model_name(site_number, chronicle, None, None, ref=True, steady=False, permeability=permeability)
    indicator = "H"
    mainRepo = folder + "/" + site_name + '/'

    if approx == 0 or approx == 2:
        rates = [1.0, 2.0, 7.0, 30.0, 90.0, 182.0, 365.0, 730.0, 3652.0]
        nb_lines = [15341, 7671, 2193, 513, 172, 86, 44, 23, 6]
    else:
        rates = [0, 0.0002, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 0.8, 1.0, 2.0]
        nb_lines = [15341, 6597, 239, 122, 62, 50, 32, 26, 17, 14, 8]

    modelnames = [ref_name]
    for p in range(1, len(rates)):
        modelnames.append(get_model_name(site_number, chronicle, approx, rates[p], ref=False, steady=False, permeability=permeability))


    dfglob = pd.DataFrame()

    for ind in range(len(modelnames)):
        simuRepo = modelnames[ind]
        filename = simuRepo + "_Ref_" + ref_name + "_errorsresult_" + indicator +"_BVE.csv"
        try:
            df = pd.read_csv(mainRepo + simuRepo + "/" + filename, sep=";")
        except FileNotFoundError:
            print("File not Found: ", filename)
            continue
        dfglob = pd.concat([dfglob,df])
    dfglob.to_csv(mainRepo + "H_Errors_Chronicle" + str(chronicle) + "_Approx" + str(approx) + "_K" + str(permeability) + "_BVE.csv", index=False)
    return dfglob


# Get H ind values to determine the value of P (real)
def get_pmax(df_Herrors):
    liste_y_test_HError = df_Herrors["H Error"].tolist()
    H_limit = 0.1
    rates = [1, 2, 7, 30, 90, 182, 365, 730, 3652]
    for i in range(len(liste_y_test_HError)):
        if liste_y_test_HError[i] > H_limit:
            p_test = rates[i - 1]
            break
        else:
            if i == len(liste_y_test_HError) - 1:
                p_test = rates[-1]

    return(p_test)

def create_file_with_features_pmax(site_number, chronicle, approx, permeability):
    print("Site ", site_number, " :")
    df_sat_feats = get_df_saturation_features(site_number, chronicle, permeability)
    df_geomorph = get_df_geomorph_features(site_number)
    df_vulne = get_df_vulnerability_features(site_number)
    df_features = pd.concat([df_geomorph, df_vulne[['Coastal Vulnerability', 'Hydrological Vulnerability']], df_sat_feats], axis=1, join='inner')
    df_Herrors = get_H_Errors_for_a_site(site_number, chronicle, approx, permeability)
    p_max = get_pmax(df_Herrors)
    df_features["Pmax"] = p_max
    site_name = get_site_name_from_site_number(site_number)
    df_features.to_csv(folder + "/" + site_name + "/" + "Features_Prediction_Pmax_Chronicle" + str(chronicle) + "_Approx" + str(approx) + "_K" + str(permeability) + "_BVE.csv", index=False)
    print("File '" + "Features_Prediction_Pmax_Chronicle" + str(chronicle) + "_Approx" + str(approx) + "_K" + str(permeability) + "_BVE.csv" + "' created!")




if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-chr", "--chronicle", type=int, required=True)
    parser.add_argument("-approx", "--approximation", type=int, required=True)
    parser.add_argument("-site", "--sitenumber", type=int, required=True)
    #parser.add_argument("-f", "--folder", type=str, required=False)
    parser.add_argument("-perm", "--permeability", type=float, required=False)

    args = parser.parse_args()

    approx = args.approximation
    chronicle = args.chronicle
    site = args.sitenumber
    #folder = args.folder
    perm = args.permeability

    create_file_with_features_pmax(site, chronicle, approx, perm)
