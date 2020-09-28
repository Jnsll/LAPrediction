import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import argparse


# global var
folder = "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results"

# Using pandas to import the dataset
df = pd.read_csv(folder + "/" + "Exps_H_Indicator_All_Sites.csv")


def all_random_forest(approx, chronicle, test_site):
    MYDIR = (
        "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/"
        + "ZLearning/"
        + "Approx"
        + str(approx)
        + "/Chronicle"
        + str(chronicle)
        + "/SiteTest"
        + str(test_site)
    )

    (
        df_Chr_Approx,
        df_Chr_Approx_SiteTest,
    ) = get_data_with_chronicle_number_and_site_number_as_test(
        df, approx, chronicle, test_site
    )
    get_simple_scatter_plot_variables(df_Chr_Approx, approx, chronicle, test_site)
    X, y, minimum_H, maximum_H, mean_H, median_H, std_H = get_stats_from_dataset(
        df_Chr_Approx, approx, chronicle, test_site
    )
    get_pairplots(df_Chr_Approx, approx, chronicle, test_site)
    get_correlation_matrix(df_Chr_Approx, approx, chronicle, test_site)

    (
        forest,
        mse_train,
        mse_test,
        r2_train,
        r2_test,
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_pred,
    ) = get_randomForest_model(X, y)

    X_testSite, y_testSite = get_testSite_datasets(df_Chr_Approx_SiteTest)
    (
        mse_testSite_train,
        mse_testSite_test,
        r2_testSite_train,
        r2_testSite_test,
    ) = get_scores_testSite(forest, X_testSite, y_testSite, y_train, y_train_pred)

    with open(MYDIR + "/" + "Stats.csv", "w") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow(
            [
                "Approx",
                "Chronicle",
                "Test Site",
                "Minimum H",
                "Maximum H",
                "Mean H",
                "Median H",
                "Standard Dev H",
                "MSE Train",
                "MSE Test",
                "R2 Train",
                "R2 Test",
                "MSE Train Test Site",
                "MSE Test Test Site",
                "R2 Train Test Site",
                "R2 Test Test Site",
            ]
        )
        writer.writerow(
            [
                approx,
                chronicle,
                test_site,
                minimum_H,
                maximum_H,
                mean_H,
                median_H,
                std_H,
                mse_train,
                mse_test,
                r2_train,
                r2_test,
                mse_testSite_train,
                mse_testSite_test,
                r2_testSite_train,
                r2_testSite_test,
            ]
        )


def get_scores_testSite(forest, X_testSite, y_testSite, y_train, y_train_pred):
    y_testSite_test = forest.predict(X_testSite)
    mse_testSite_train = mean_squared_error(y_train, y_train_pred)
    mse_testSite_test = mean_squared_error(y_testSite, y_testSite_test)
    r2_testSite_train = r2_score(y_train, y_train_pred)
    r2_testSite_test = r2_score(y_testSite, y_testSite_test)
    print(
        "MSE test Site train : %.3f, test: %.3f"
        % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_testSite, y_testSite_test),
        )
    )
    print(
        "R^2 test Site train: %.3f, test: %.3f"
        % (r2_score(y_train, y_train_pred), r2_score(y_testSite, y_testSite_test))
    )
    return mse_testSite_train, mse_testSite_test, r2_testSite_train, r2_testSite_test


def get_testSite_datasets(df_Chr_Approx_SiteTest):
    y_testSite = df_Chr_Approx_SiteTest[
        "H Error"
    ]  # df_Chr0_Approx0.filter(["H Error"], axis=1)

    X_testSite = df_Chr_Approx_SiteTest.drop("H Error", axis=1)

    return X_testSite, y_testSite


def get_scores(y_train, y_test, y_train_pred, y_test_pred):
    mse_train = mean_squared_error(y_train, y_train_pred)
    mse_test = mean_squared_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    print(
        "MSE train: %.3f, test: %.3f"
        % (
            mean_squared_error(y_train, y_train_pred),
            mean_squared_error(y_test, y_test_pred),
        )
    )
    print(
        "R^2 train: %.3f, test: %.3f"
        % (r2_score(y_train, y_train_pred), r2_score(y_test, y_test_pred))
    )
    return mse_train, mse_test, r2_train, r2_test


def get_randomForest_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1
    )
    # Success
    print("Training and testing split was successful.")

    forest = RandomForestRegressor(
        n_estimators=1000, criterion="mse", random_state=1, n_jobs=-1
    )

    forest.fit(X_train, y_train)
    y_train_pred = forest.predict(X_train)
    y_test_pred = forest.predict(X_test)

    mse_train, mse_test, r2_train, r2_test = get_scores(
        y_train, y_test, y_train_pred, y_test_pred
    )

    return (
        forest,
        mse_train,
        mse_test,
        r2_train,
        r2_test,
        X_train,
        X_test,
        y_train,
        y_test,
        y_train_pred,
    )


def get_correlation_matrix(df_Chr_Approx, approx, chronicle, test_site):
    MYDIR = (
        "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/"
        + "ZLearning/"
        + "Approx"
        + str(approx)
        + "/Chronicle"
        + str(chronicle)
        + "/SiteTest"
        + str(test_site)
    )
    # Calculate and show correlation matrix
    cm = np.corrcoef(df_Chr_Approx.values.T)
    sns.set(font_scale=1.5)
    hm = sns.heatmap(
        cm,
        cbar=True,
        annot=True,
        square=True,
        fmt=".2f",
        annot_kws={"size": 15},
        yticklabels=list(df_Chr_Approx),
        xticklabels=list(df_Chr_Approx),
    )
    plt.savefig(MYDIR + "/CorrelationMatrix_HError.png")
    plt.clf()


# Calculate and show pairplot
def get_pairplots(df_Chr_Approx, approx, chronicle, test_site):
    MYDIR = (
        "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/"
        + "ZLearning/"
        + "Approx"
        + str(approx)
        + "/Chronicle"
        + str(chronicle)
        + "/SiteTest"
        + str(test_site)
    )

    sns.pairplot(df_Chr_Approx, height=2.5)
    # plt.show()
    plt.savefig(MYDIR + "/Pairplots_HError.png")
    plt.clf()


def get_stats_from_dataset(df_Chr_Approx, approx, chronicle, test_site):
    y = df_Chr_Approx["H Error"]  # df_Chr0_Approx0.filter(["H Error"], axis=1)
    X = df_Chr_Approx.drop("H Error", axis=1)
    # new = old.filter(['A','B','D'], axis=1)
    print(
        "Dataset has {} data points with {} variables each.".format(
            *df_Chr_Approx.shape
        )
    )

    # Minimum value of the data
    minimum_H = np.amin(y)
    # Maximum value of the data
    maximum_H = np.amax(y)

    # Mean value of the data
    mean_H = np.mean(y)

    # Median value of the data
    median_H = np.median(y)

    # Standard deviation of values of the data
    std_H = np.std(y)

    # Show the calculated statistics
    print("Statistics for dataset:\n")
    print("Minimum H: {}".format(minimum_H))
    print("Maximum H: {}".format(maximum_H))
    print("Mean H: {}".format(mean_H))
    print("Median H {}".format(median_H))
    print("Standard deviation of H: {}".format(std_H))

    return X, y, minimum_H, maximum_H, mean_H, median_H, std_H


def get_simple_scatter_plot_variables(df_Chr_Approx, approx, chronicle, test_site):
    MYDIR = (
        "/run/media/jnsll/b0417344-c572-4bf5-ac10-c2021d205749/exps_modflops/results/"
        + "ZLearning/"
        + "Approx"
        + str(approx)
        + "/Chronicle"
        + str(chronicle)
        + "/SiteTest"
        + str(test_site)
    )
    features = [
        "Rate",
        "Satured Zone Area",
        "Vulnerability Sum",
        "Vulnerability Rate",
        "Global Area of site (in cells)",
        "Saturation Rate",
    ]
    colors = ["black", "blue", "orange", "grey", "green", "purple"]

    CHECK_FOLDER = os.path.isdir(MYDIR)
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)

    for i in range(len(features)):
        plt.scatter(
            df_Chr_Approx[features[i]], df_Chr_Approx["H Error"], color=colors[i]
        )
        plt.title("H Error Vs " + features[i], fontsize=14)
        plt.xlabel(features[i], fontsize=14)
        plt.ylabel("H Error", fontsize=14)
        plt.grid(True)
        # plt.show()

        plt.savefig(
            MYDIR + "/simple_scatter_plot_HError_" + features[i] + ".png"
        )  # , bbox_inches='tight'
        plt.clf()


def get_data_with_chronicle_number_and_site_number_as_test(
    df, approx, chronicle, test_site
):
    """
    TODO
    """

    df_chr = df[df["Chronicle"] == chronicle]
    df_Chr_Approx = df_chr[df_chr["Approx"] == approx]

    del df_Chr_Approx["Approx"]
    del df_Chr_Approx["Chronicle"]
    del df_Chr_Approx["Execution Time"]
    del df_Chr_Approx["Number of Lines"]
    df_Chr_Approx_SiteTest = df_Chr_Approx[df_Chr_Approx.Site_number == test_site]
    df_Chr_Approx = df_Chr_Approx.drop(
        df_Chr_Approx[df_Chr_Approx.Site_number == test_site].index
    )
    del df_Chr_Approx["Site_number"]
    del df_Chr_Approx_SiteTest["Site_number"]
    # print(df_Chr_Approx)
    return df_Chr_Approx, df_Chr_Approx_SiteTest


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-chr", "--chronicle", type=int, required=True)
    parser.add_argument("-approx", "--approximation", type=int, required=False)
    parser.add_argument("-site", "--sitenumber", type=int, required=True)

    args = parser.parse_args()

    approx = args.approximation
    chronicle = args.chronicle
    site_number = args.sitenumber

    all_random_forest(approx, chronicle, site_number)
