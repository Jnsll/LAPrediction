# Lib imports
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import argparse
import csv
from numpy import mean
from numpy import std
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import GradientBoostingRegressor


# Prediction variables
chronicle = 0
approx = 0
permeability = 27.32
# Global variables
result_folder = (
    "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results"
)
dataset_filename = (
    "Exps_H_Indicator_Chronicle"
    + str(chronicle)
    + "_Approx"
    + str(approx)
    + "_K"
    + str(permeability)
    + "_All_Sites"
    + "_BVE_CVHV_Geomorph_Sat_Extend.csv"
)

# test_site = 1 # belongs to [1,...]
# learn_size = 0.3

rates = [
        1.0,
        2.0,
        7.0,
        15.0,
        21.0,
        30.0,
        45.0,
        50.0,
        60.0,
        75.0,
        90.0,
        100.0,
        125.0,
        150.0,
        182.0,
        200.0,
        250.0,
        300.0,
        330.0,
        365.0,
        550.0,
        640.0,
        730.0,
        1000.0,
        1500.0,
        2000.0,
        2250.0,
        3000.0,
        3182.0,
        3652.0,
    ]



def gbt_pred(test_site, chronicle=0, approx=0, permeability=27.32):
    print("test site: ", test_site)
    # Path to store the files created during the prediction process
    MYDIR = (
        result_folder
        + "/ZLearning/"
        + "Approx"
        + str(approx)
        + "/Chronicle"
        + str(chronicle)
        + "/SiteTest"
        + str(test_site)
    )

    # Checking if the path and directory where to store the prediction files exists, if not it is created
    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("The directory did not exist. It has been created here: " + MYDIR)

    # Importing the dataset and storing it inside a dataframe
    df = pd.read_csv(result_folder + "/" + dataset_filename)
    # print(df)
    # print(result_folder + "/" + dataset_filename)

    # Only selecting the data corresponding to the chronicle and approximation chosen
    df_chr = df[df["Chronicle"] == chronicle]
    df_Chr_Approx = df_chr[df_chr["Approx"] == approx]

    # Removing the dataframe columns which are not to be used for the prediction
    del df_Chr_Approx["Approx"]
    del df_Chr_Approx["Chronicle"]
    del df_Chr_Approx["Execution Time"]
    del df_Chr_Approx["Number of Lines"]

    # print(df_Chr_Approx)

    # Variable to predict
    y = df_Chr_Approx.filter(["Site_number", "H Error"], axis=1)
    # y = pd.concat([df_Chr_Approx["Site_number"], df_Chr_Approx["H Error"]], axis=1)
    # Features used to predict
    X = df_Chr_Approx.drop("H Error", axis=1)

## CALL TO GET STATS
# get_global_stats_of_dataset(df_Chr_Approx,y)


    X_train, X_test, y_train, y_test = get_train_and_test_data(X, y, test_site)

    # n_estimators :
    # creiterion :
    # random_state :
    # n_jobs :
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train.values.ravel())


    # Predicting results
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # print(y_test_pred, X_test)

    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    # print('MSE train: %.3f, test: %.3f' % (mse_train, mse_test))
    # print('R^2 train: %.3f, test: %.3f' % (r2_train,r2_test))

    liste_y_test_HError = y_test["H Error"].tolist()
    p_test, p_pred = get_real_and_pred_pmax(liste_y_test_HError, y_test_pred)

    with open(
        MYDIR
        + "/"
        + "Prediction_HError_GBT_Chronicle"
        + str(chronicle)
        + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability)
        + "_BVE_CVHV_Geomorph_Sat_H_values_Extend.csv",
        "w",
    ) as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "Approx",
                "Chronicle",
                "Test Site",
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
                    test_site,
                    rates[i],
                    liste_y_test_HError[i],
                    y_test_pred[i],
                ]
            )

    with open(
        MYDIR
        + "/"
        + "Prediction_HError_GBT_Chronicle"
        + str(chronicle)
        + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability)
        + "_BVE_CVHV_Geomorph_Sat_Extend.csv",
        "w",
    ) as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "Approx",
                "Chronicle",
                "Test Site",
                "MSE Train",
                "MSE Test",
                "R2 Train",
                "R2 Test",
                "P Real",
                "P pred",
            ]
        )
        writer.writerow(
            [
                approx,
                chronicle,
                test_site,
                mse_train,
                mse_test,
                r2_train,
                r2_test,
                p_test,
                p_pred,
            ]
        )

    get_plot_comparison_HError_values(X_test, y_test, y_test_pred, MYDIR)

    return mse_train, mse_test, r2_train, r2_test, p_test, p_pred

def get_real_and_pred_pmax(liste_y_test_HError, y_test_pred, H_limit=0.1):
    for i in range(len(liste_y_test_HError)):
        if liste_y_test_HError[i] > H_limit:
            p_test = rates[i - 1]
            break
        else:
            if i == len(liste_y_test_HError) - 1:
                p_test = rates[-1]
    for i in range(len(y_test_pred)):
        if y_test_pred[i] > H_limit:
            p_pred = rates[i - 1]
            break
        else:
            if i == len(y_test_pred) - 1:
                p_pred = rates[-1]
    print("Real value of p: ", p_test)
    print("Predicted value of p: ", p_pred)
    return p_test, p_pred

def get_train_and_test_data(X, y, test_site):
    y_test = y[y.Site_number == test_site]
        ## We do not want to take the site number into account for the prediction
    del y_test["Site_number"]

        # Removing the data for the site we want to predict
    y_train = y.drop(y[y.Site_number == test_site].index)
        ## We do not want to take the site number into account for the prediction
    del y_train["Site_number"]

        # Splitting the x (features) into training and testing data
    X_test = X[X.Site_number == test_site]
        ## We do not want to take the site number into account for the prediction
    del X_test["Site_number"]

        # Removing the data for the site we want to predict
    X_train = X.drop(X[X.Site_number == test_site].index)
        ## We do not want to take the site number into account for the prediction
    del X_train["Site_number"]
    
    return X_train, X_test, y_train, y_test

def get_plot_comparison_HError_values(X_test, y_test, y_test_pred, MYDIR):
    dfp = pd.concat([X_test["Rate"], y_test], axis=1)
    # dfp = pd.concat([dfp, cat])
    plt.rcParams.update({"font.size": 18})
    # print(dfp)
    indicator = "H"
    marker = ["o", "^", "P", ">", "d", "p", "s", "h", "D"]
    # hue_kws=dict()
    a = sns.relplot(x="Rate", y=indicator + " Error", s=70, markers=marker, data=dfp)
    a.set(xlabel="Rate", ylabel=indicator + " Indicator (m)")
    axes = plt.gca()
    # axes.set_ylim([ymin,ymax])
    plt.plot(dfp["Rate"], dfp[indicator + " Error"], linewidth=2, alpha=0.2)
    plt.xticks(dfp["Rate"], rotation=70)
    # a.fig.suptitle("Evolution of " + indicator + " indicator value according to the execution time \n Approximation with " + approximation + "\n" + site_name + " site")
    plt.subplots_adjust(top=0.8)

    max_exec_time = 3652

    H_threshold = 0.1

    plt.plot(
        [0, max_exec_time],
        [H_threshold, H_threshold],
        linewidth=3,
        alpha=0.7,
        color="Red",
        dashes=[6, 2],
    )

    # print(dfp['Rate'])
    # print(y_test_pred)
    plt.plot(
        dfp["Rate"], y_test_pred, linewidth=3, alpha=0.7, color="Green", marker="o"
    )

    a.savefig(
        MYDIR
        + "/Comparison_HError_GBTPredict_real_pred"
        + "_Chronicle"
        + str(chronicle)
        + "_Approx"
        + str(approx)
        + "_K"
        + str(permeability)
        + "_BVE_CVHV_Geomorph_H_Values_Extend.png"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-site", "--sitenumber", type=int, required=True)

    args = parser.parse_args()

    site_number = args.sitenumber

    gbt_pred(site_number, chronicle=0, approx=0, permeability=27.32)
