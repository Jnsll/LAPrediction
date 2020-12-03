import pandas as pd
import numpy as np
import csv
import os
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Global Variables
RESULT_FOLDER = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results"


def predict_H_ind_for_a_site_with_subCatchment_data(site_number, chronicle=0, approx=0, permeability=27.32):
    input_data = import_input_data()
    input_data_cleaned = clean_data(input_data)
    features, variable_to_predict = split_dataset_into_features_and_variable_to_predict(input_data_cleaned)
    features_train, features_test, variable_train, variable_test = get_train_and_test_data(features, variable_to_predict, site_number)
    prediction_model = train_random_forest_model(features_train, variable_train)
    variable_train_pred, variable_test_pred = predict_with_trained_model(prediction_model, features_train, features_test)
    subCatchment_numbers = get_subcatchment_numbers_for_a_site(site_number, variable_to_predict)
    liste_variable_test_HError = get_list_variable_test_Hind_Values(variable_test)
    liste_variable_test_pred_HError = variable_test_pred
    mse_test, r2_test = get_standard_quality_metrics(subCatchment_numbers, liste_variable_test_HError, liste_variable_test_pred_HError)
    rates = get_rates_for_a_site(site_number, features)
    pmax_test, pmax_pred = get_real_and_pred_pmax(subCatchment_numbers, rates, liste_variable_test_HError, variable_test_pred)
    save_Hind_results_into_file(site_number, subCatchment_numbers, rates, liste_variable_test_HError, variable_test_pred, approx=0, chronicle=0, permeability=27.32)
    save_Pmax_results_into_file(site_number, pmax_test, pmax_pred, mse_test, r2_test, approx=0, chronicle=0, permeability=27.32)


def get_list_variable_test_Hind_Values(y_test):
    liste_y_test_HError = y_test["HError"].tolist()
    return liste_y_test_HError

def get_rates_for_a_site(site_number, data):
    rates = data[data["Site"]==site_number]["Rate"].to_list()
    return rates

def get_subcatchment_numbers_for_a_site(site_number, data):
    subCatchment_numbers = data[data["Site"]==site_number]["SubCatch"].to_list()
    return subCatchment_numbers

def import_input_data():
    # Importing the dataset and storing it inside a dataframe
    input_data = pd.read_csv(RESULT_FOLDER + "/" + "DataInputPred_SUB.csv", sep=",")
    return input_data


def clean_data(input_data):
    # Removing rows with missing data
    input_data_no_nan = input_data.dropna()
    return input_data_no_nan

def split_dataset_into_features_and_variable_to_predict(input_data_no_nan):
    y = input_data_no_nan.filter(["Site", "SubCatch", "HError"], axis=1)
    X = input_data_no_nan.drop("HError", axis=1)
    return X,y


def get_train_and_test_data(X, y, test_site):
    y_test = y[y.Site == test_site]
        ## We do not want to take the site number into account for the prediction
    del y_test["Site"]
    del y_test["SubCatch"]

        # Removing the data for the site we want to predict
    y_train = y.drop(y[y.Site == test_site].index)
        ## We do not want to take the site number into account for the prediction
    del y_train["Site"]
    del y_train["SubCatch"]

        # Splitting the x (features) into training and testing data
    X_test = X[X.Site == test_site]
        ## We do not want to take the site number into account for the prediction
    del X_test["Site"]
    del X_test["SubCatch"]

        # Removing the data for the site we want to predict
    X_train = X.drop(X[X.Site == test_site].index)
        ## We do not want to take the site number into account for the prediction
    del X_train["Site"]
    del X_train["SubCatch"]
    
    return X_train, X_test, y_train, y_test



def train_random_forest_model(X_train, y_train):
    forest = RandomForestRegressor(
        n_estimators=1000, criterion="mse", random_state=1, n_jobs=-1
    )
    forest.fit(X_train, y_train.values.ravel())
    return forest

def predict_with_trained_model(forest, X_train, X_test):
    # Predicting results
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)
    return y_train_pred, y_test_pred


def get_standard_quality_metrics(subCatchment_numbers, liste_y_test_HError, liste_y_test_pred_HError):
    subcatch = 0
    mse_test = {}
    r2_test = {}
    y_test_by_subcatch = {}
    y_test_pred_by_subcatch = {}


    for index_sub in range(len(subCatchment_numbers)):
        if subCatchment_numbers[index_sub] != subcatch:
            subcatch = subCatchment_numbers[index_sub]
            y_test_by_subcatch[subcatch] = []
            y_test_pred_by_subcatch[subcatch] = []
        y_test_by_subcatch[subcatch].append(liste_y_test_HError[index_sub])
        y_test_pred_by_subcatch[subcatch].append(liste_y_test_pred_HError[index_sub])
        

    for sub in y_test_by_subcatch:
        mse_test[sub] = mean_squared_error(y_test_by_subcatch[sub], y_test_pred_by_subcatch[sub])
        r2_test[sub] = r2_score(y_test_by_subcatch[sub], y_test_pred_by_subcatch[sub])

    # print('MSE train: %.3f, test: %.3f' % (mse_train, mse_test))
    # print('R^2 train: %.3f, test: %.3f' % (r2_train,r2_test))
    return mse_test, r2_test

# def get_pmax_from_variable_predictions(subCatchment_numbers, rates, y_test, y_test_pred):
#     p_test, p_pred = get_real_and_pred_pmax(subCatchment_numbers, rates, liste_y_test_HError, y_test_pred)
#     return p_test, p_pred

def get_real_and_pred_pmax(subCatchment_numbers, rates, liste_y_test_HError, y_test_pred, H_limit=0.1):
    subcatch = 0
    p_test = {}
    p_pred = {}
    for index_sub in range(len(subCatchment_numbers)):
        if subCatchment_numbers[index_sub] != subcatch:
            subcatch = subCatchment_numbers[index_sub]
            pmaxTest_found = False
            pmaxPred_found = False
            
        if pmaxTest_found is False and liste_y_test_HError[index_sub] > H_limit:
            p_test[subcatch] = rates[index_sub -1]
            pmaxTest_found = True
        elif pmaxTest_found is False and index_sub == len(subCatchment_numbers)-1:
                p_test[subcatch] = rates[-1]
        elif pmaxTest_found is False and subCatchment_numbers[index_sub+1] != subcatch:
                p_test[subcatch] = rates[-1]
        
        
        if pmaxPred_found is False and y_test_pred[index_sub] > H_limit:
            p_pred[subcatch] = rates[index_sub -1]
            pmaxPred_found = True
        elif pmaxPred_found is False and index_sub == len(subCatchment_numbers)-1:
                p_pred[subcatch] = rates[-1]
        elif pmaxPred_found is False and subCatchment_numbers[index_sub+1] != subcatch:
                p_pred[subcatch] = rates[-1]
    
    print("Real value of p: ", p_test)
    print("Predicted value of p: ", p_pred)
    return p_test, p_pred


def save_Hind_results_into_file(site_number, subCatchment_numbers, rates, liste_y_test_HError, y_test_pred, approx=0, chronicle=0, permeability=27.32):

    MYDIR = (
    RESULT_FOLDER
    + "/ZLearning/"
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
        + "Prediction_HErrorValues_SubCatch_Chronicle"
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


def save_Pmax_results_into_file(site_number, p_test, p_pred, mse_test, r2_test, approx=0, chronicle=0, permeability=27.32):
    MYDIR = (
    RESULT_FOLDER
    + "/ZLearning/"
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
        + "Prediction_PMax_SubCatch_Chronicle"
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-site", "--sitenumber", type=int, required=True)
    args = parser.parse_args()

    site_number = args.sitenumber
    
    print("Init...")
    predict_H_ind_for_a_site_with_subCatchment_data(site_number, chronicle=0, approx=0, permeability=27.32)
    print("...Done")