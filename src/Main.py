import Data
import LearningModel
import Prediction

data = Data.Data()
input_data = data.load_data()
input_data_cleaned = data.clean_data(input_data)
#print(input_data_cleaned)
features, variable_to_predict = data.split_data_into_features_and_variable_to_predict(input_data_cleaned)
# print(features)
# print(variable_to_predict)
features_train, features_test, variable_train, variable_test = data.get_train_and_test_data_for_a_subcatch(features, variable_to_predict, test_site=5, test_subcatch=4)


model = LearningModel.LearningModel()
model.set_model_type("RandomForest")
prediction_model = model.train_model(features_train, variable_train)
#print(prediction_model)

prediction = Prediction.Prediction()
variable_train_pred, variable_test_pred = prediction.predict_with_trained_model(prediction_model, features_train, features_test)
rates = data.get_rates_for_a_subcatch(features, site_number=5, subcatch_number=4)
liste_variable_test_HError = data.get_list_variable_test_Hind_Values(variable_test)
liste_variable_test_pred_HError = variable_test_pred
subcatch_number = 4
pmax_test, pmax_pred = prediction.get_real_and_pred_pmax(rates, subcatch_number, liste_variable_test_HError, liste_variable_test_pred_HError)