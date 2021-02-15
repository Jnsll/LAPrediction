from sklearn.ensemble import RandomForestRegressor

class LearningModel:

    def __init__(self):
        pass

    def get_model(self):
        return self.model

    def set_model_type(self, type):
        if self.is_model_type_random_forest(type):
            self.model = RandomForestRegressor( n_estimators=1000, criterion="mse", random_state=1, 
            n_jobs=-1, oob_score = True, bootstrap = True)
        return self.model

    def is_model_type_random_forest(self, type):
        if type == "RandomForest":
            return True
        else:
            return False
    
    
    def train_model(self, X_train, y_train):
        self.model.fit(X_train, y_train.values.ravel())
        return self.model
