import pandas as pd

class Data(object):

    DATA_FOLDER = "/DATA/These/Projects/LAPrediction/notebooks/data"
    file_name = "DataInputPred_SUB_clean.csv"  

    def load_data(self):
        input_data = pd.read_csv(self.DATA_FOLDER + "/" + self.file_name, sep=",")
        return input_data

    def clean_data(self, input_data):
        input_data_no_nan = input_data.dropna()
        return input_data_no_nan

    def split_data_into_features_and_variable_to_predict(self, input_data_cleaned):
        y = input_data_cleaned.filter(["Site", "SubCatch", "HError"], axis=1)
        X = input_data_cleaned.drop("HError", axis=1)
        return X,y

    def get_train_and_test_data_for_a_subcatch(self, X, y, test_site, test_subcatch):
        y_test = y.loc[(y.Site == test_site) & (y.SubCatch == test_subcatch)]
            ## We do not want to take the site number into account for the prediction
        del y_test["Site"]
        del y_test["SubCatch"]

            # Removing the data for the site we want to predict =>>> DROP the whole site because any simulation has been run for the whole site.
        y_train = y.drop(y.loc[(y.Site == test_site)].index)
            ## We do not want to take the site number into account for the prediction
        del y_train["Site"]
        del y_train["SubCatch"]

            # Splitting the x (features) into training and testing data
        X_test = X.loc[(X.Site == test_site) & (X.SubCatch == test_subcatch)]
            ## We do not want to take the site number into account for the prediction
        del X_test["Site"]
        del X_test["SubCatch"]

            # Removing the data for the site we want to predict
        X_train = X.drop(X.loc[(X.Site == test_site)].index)
            ## We do not want to take the site number into account for the prediction
        del X_train["Site"]
        del X_train["SubCatch"]
        
        return X_train, X_test, y_train, y_test

    def get_list_variable_test_Hind_Values(self, y_test):
        liste_y_test_HError = y_test["HError"].tolist()
        return liste_y_test_HError

    def get_subcatchment_numbers_for_a_subcatch(self, y, site_number, subcatch_number):
        subCatchment_numbers = y[y["Site"]==site_number]["SubCatch"].to_list()
        subs = [subcatch for subcatch in subCatchment_numbers if subcatch == subcatch_number]
        return subs

    def get_rates_for_a_subcatch(self, features, site_number, subcatch_number):
        rates = features.loc[(features["Site"]==site_number) & (features["SubCatch"] == subcatch_number)]["Rate"].to_list()
        return rates