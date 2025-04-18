from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
import joblib

def train_model(X, y, save_path):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=42)

    model = LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=64, class_weight='balanced')
    model.fit(X_train, y_train)

    joblib.dump(model, save_path)
    return model, X_test, y_test
