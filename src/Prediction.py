from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

class Prediction:


    def predict_with_trained_model(self, learningModel, X_train, X_test):
        # Predicting results
        y_train_pred = learningModel.predict(X_train)
        y_test_pred = learningModel.predict(X_test)
        return y_train_pred, y_test_pred

    def get_mse_and_nse_metrics(self, subCatchment_numbers, liste_y_test_HError, liste_y_test_pred_HError):
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

        self.mse = mse_test
        self.nse = r2_test

        return mse_test, r2_test

    
    def get_real_and_pred_pmax_for_a_subcatch(self, rates, subcatch_number, liste_variable_test_HError, liste_variable_test_pred_HError, H_limit=0.1):
        p_test = 0
        p_pred = 0
        pmaxTest_found = False
        pmaxPred_found = False

        if len(rates) != len(liste_variable_test_HError):
            print("Not same size for rates and liste_variable_test_HError")
            return False
        if len(rates) != len(liste_variable_test_pred_HError):
            print("Not same size for rates and liste_variable_test_pred_HError")
            return False
        if len(liste_variable_test_HError) != len(liste_variable_test_pred_HError):
            print("Listes of H values are not the same size!")
            return False

        for index_sub in range(len(liste_variable_test_HError)):
            if pmaxTest_found is False and liste_variable_test_HError[index_sub] > H_limit:
                p_test = rates[index_sub -1]
                pmaxTest_found = True
            elif pmaxTest_found is False and index_sub == len(liste_variable_test_HError)-1:
                if rates[-1] == float(3652):
                    p_test = 3652
            elif pmaxTest_found:
                print(index_sub)
                break
                    
        for index_sub in range(len(liste_variable_test_pred_HError)):    
            if pmaxPred_found is False and liste_variable_test_pred_HError[index_sub] > H_limit:
                p_pred = rates[index_sub -1]
                pmaxPred_found = True
            elif pmaxPred_found is False and index_sub == len(liste_variable_test_pred_HError)-1:
                if rates[-1] == float(3652):
                    p_pred = rates[-1]
            elif pmaxPred_found:
                print(index_sub)
                break
        
        print("Real value of p: ", p_test)
        print("Predicted value of p: ", p_pred)
        return p_test, p_pred