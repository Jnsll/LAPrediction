import pandas as pd
import prediction_of_H_indicator_with_subCatchmentData_by_cluster as prediction
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import os
import csv

RESULT_FOLDER = "/DATA/These/Projects/LAPrediction/notebooks/output"
OUTPUT_FOLDER = "/DATA/These/Projects/LAPrediction/notebooks/output"
NB_CLUSTERS=10




def save_Hind_results_into_file(site_number, subcatch_number, nb_clusters, subCatchment_numbers, rates, liste_y_test_HError, y_test_pred, approx=0, chronicle=0, permeability=27.32):

    MYDIR = (
    OUTPUT_FOLDER
    + "/"
    + "Approx"
    + str(approx)
    + "/Chronicle"
    + str(chronicle)
    + "/SiteTest"
    + str(site_number)
    )

    # Checking if the path and directory where to store the prediction files exists, if not it is created
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("The directory did not exist. It has been created here: " + MYDIR)

    with open(
        MYDIR
        + "/"
        + "Prediction_HErrorValues_SubCatch"
        + str(subcatch_number)
        + "_by_cluster" + str(nb_clusters)
        + "_Chronicle"
        + str(chronicle)
        + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability)
        + "_Slope_Elevation_LC_SAR_Area_CV_HV.csv",
        "w",
    ) as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "Approx",
                "Chronicle",
                "Test Site",
                "SubCatchment",
                "Rate",
                "H Error Real",
                "H Error Predict",
            ]
        )
        for i in range(len(liste_y_test_HError)):
            writer.writerow(
                [
                    approx,
                    chronicle,
                    site_number,
                    subCatchment_numbers[i],
                    rates[i],
                    liste_y_test_HError[i],
                    y_test_pred[i],
                ]
            )
    print("File created here: ", MYDIR
        + "/"
        + "Prediction_HErrorValues_SubCatch"
        + str(subcatch_number) + "_by_cluster" + str(nb_clusters)
        + "_Chronicle"
        + str(chronicle)
        + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability)
        + "_Slope_Elevation_LC_SAR_Area_CV_HV.csv")

def save_Pmax_results_into_file(site_number, subcatch_number, nb_clusters, p_test, p_pred, mse_test, r2_test, approx=0, chronicle=0, permeability=27.32):
    MYDIR = (
    OUTPUT_FOLDER
    + "/"
    + "Approx"
    + str(approx)
    + "/Chronicle"
    + str(chronicle)
    + "/SiteTest"
    + str(site_number)
    )
    
    # Checking if the path and directory where to store the prediction files exists, if not it is created
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("The directory did not exist. It has been created here: " + MYDIR)
    
    with open(
        MYDIR
        + "/"
        + "Prediction_PMax_SubCatch"
        + str(subcatch_number)
        + "_by_cluster" + str(nb_clusters)
        + "_Chronicle"
        + str(chronicle)
        + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability)
        + "_Slope_Elevation_LC_SAR_Area_CV_HV.csv",
        "w",
    ) as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "Approx",
                "Chronicle",
                "Test Site",
                "SubCatchment",
                "MSE Test",
                "R2 Test",
                "P Real",
                "P pred",
            ]
        )
        for subCatch in p_test:
            writer.writerow(
                [
                    approx,
                    chronicle,
                    site_number,
                    subCatch,
                    mse_test[subCatch],
                    r2_test[subCatch],
                    p_test[subCatch],
                    p_pred[subCatch],
                ]
            )
    print("File created here: " + MYDIR+ "/"+ "Prediction_PMax_SubCatch" + str(subcatch_number) + "_by_cluster" + str(nb_clusters)
        + "_Chronicle"+ str(chronicle)+ "_Approx" + str(approx)+ "_K"+ str(permeability)+ "_Slope_Elevation_LC_SAR_Area_CV_HV.csv")



def get_cluster_for_site_subcath(input_data_cluster, site_number, subcatch_number):
    return int(input_data_cluster.loc[(input_data_cluster.Site == site_number) & (input_data_cluster.SubCatch == subcatch_number)]["clusters"])

def get_train_and_test_data(X, y, test_site, test_subcatch):
    y_test = y.loc[(y.Site == test_site) & (y.SubCatch == test_subcatch)]
        ## We do not want to take the site number into account for the prediction
    del y_test["Site"]
    del y_test["SubCatch"]

        # Removing the data for the site we want to predict  =>>> /!\ TO DO : remove only the test subcatch of the test site ?
    y_train = y.drop(y[y.Site == test_site].index)
        ## We do not want to take the site number into account for the prediction
    del y_train["Site"]
    del y_train["SubCatch"]

        # Splitting the x (features) into training and testing data
    X_test = X.loc[(X.Site == test_site) & (X.SubCatch == test_subcatch)]
        ## We do not want to take the site number into account for the prediction
    del X_test["Site"]
    del X_test["SubCatch"]
    del X_test["Category"]
    del X_test["clusters"]
        # Removing the data for the site we want to predict
    X_train = X.drop(X[X.Site == test_site].index)
        ## We do not want to take the site number into account for the prediction
    del X_train["Site"]
    del X_train["SubCatch"]
    del X_train["Category"]
    del X_train["clusters"]
    
    return X_train, X_test, y_train, y_test

def train_random_forest_model(X_train, y_train):
    forest = RandomForestRegressor(
        n_estimators=1000, criterion="mse", random_state=1, n_jobs=-1, oob_score = True, bootstrap = True
    )
    forest.fit(X_train, y_train.values.ravel())
    return forest

def predict_with_trained_model(forest, X_train, X_test):
    # Predicting results
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    return y_train_pred, y_test_pred


# subcatch_in_clusters = {}
# for cluster in range(NB_CLUSTERS):
#     subcatch_in_clusters[cluster] = {}
#     data_for_cluster = input_data_cluster[input_data_cluster["clusters"]==cluster]
#     #print(data_for_cluster)
#     for index in range(len(data_for_cluster)):
#         site = data_for_cluster.iloc[index, 0]
#         sub = data_for_cluster.iloc[index, 1]
#         if site not in subcatch_in_clusters[cluster]:
#             subcatch_in_clusters[cluster][site] = [sub]
#         else:
#             subcatch_in_clusters[cluster][site].append(sub)


input_data_cluster = pd.read_csv(RESULT_FOLDER + "/" + "clusters_categories" + str(NB_CLUSTERS) + "_cut.csv", sep=",")

input_data = prediction.import_input_data()
input_data_cleaned = prediction.clean_data(input_data)
result = pd.merge(input_data_cleaned, input_data_cluster, how="left",on=["Site", "SubCatch"])


for site_number in range(1,40):
    for test_subcatch in range(1,30):
        print(site_number)
        print(test_subcatch)
        try :
            cluster_number = get_cluster_for_site_subcath(input_data_cluster, site_number, test_subcatch)
        except:
            continue
        data_cluster = result.loc[result["clusters"]==cluster_number]

        features, variable_to_predict = prediction.split_dataset_into_features_and_variable_to_predict(data_cluster)
        features_train, features_test, variable_train, variable_test = get_train_and_test_data(features, variable_to_predict, site_number, test_subcatch)
        prediction_model = train_random_forest_model(features_train, variable_train)
        variable_train_pred, variable_test_pred = predict_with_trained_model(prediction_model, features_train, features_test)
        liste_variable_test_HError = prediction.get_list_variable_test_Hind_Values(variable_test)
        subCatchment_numbers = [test_subcatch] * len((variable_test))
        liste_variable_test_pred_HError = variable_test_pred
        mse_test, r2_test = prediction.get_standard_quality_metrics(subCatchment_numbers, liste_variable_test_HError, liste_variable_test_pred_HError)
        rates = prediction.get_rates_for_a_site(site_number, features)
        pmax_test, pmax_pred = prediction.get_real_and_pred_pmax(subCatchment_numbers, rates, liste_variable_test_HError, variable_test_pred)
        save_Hind_results_into_file(site_number, test_subcatch, NB_CLUSTERS, subCatchment_numbers, rates, liste_variable_test_HError, variable_test_pred, approx=0, chronicle=0, permeability=27.32)
        save_Pmax_results_into_file(site_number, test_subcatch, NB_CLUSTERS, pmax_test, pmax_pred, mse_test, r2_test, approx=0, chronicle=0, permeability=27.32)