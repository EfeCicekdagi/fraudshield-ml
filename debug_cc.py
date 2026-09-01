import sys
sys.path.insert(0, 'src')
import joblib
import pandas as pd
from fraudshield.inference.predictor import FraudPredictor
from sklearn.base import BaseEstimator
from fraudshield.pipelines.finalization_pipeline import MLPPipeline
from sklearn.utils._response import _get_response_values
from sklearn.preprocessing import LabelEncoder
import numpy as np

class Dummy(BaseEstimator):
    pass

MLPPipeline.__sklearn_tags__ = Dummy.__sklearn_tags__

c = joblib.load('artifacts/final/calibrator.joblib')
df = pd.DataFrame({'step':[1],'type':['PAYMENT'],'amount':[100],'oldbalanceOrg':[100],'nameOrig':['C'],'nameDest':['M']})
p = FraudPredictor()
X = p._build_features(df)

cc = c.calibrated_classifiers_[0]

predictions, _ = _get_response_values(
    cc.estimator,
    X,
    response_method=["decision_function", "predict_proba"],
)
if predictions.ndim == 1:
    predictions = predictions.reshape(-1, 1)

n_classes = cc.classes.shape[0]
proba = np.zeros((X.shape[0], n_classes))

label_encoder = LabelEncoder().fit(cc.classes)
pos_class_indices = label_encoder.transform(cc.estimator.classes_)
print(f"pos_class_indices={pos_class_indices}")
print(f"predictions={predictions}")

for class_idx, this_pred, calibrator in zip(
    pos_class_indices, predictions.T, cc.calibrators
):
    print(f"class_idx before={class_idx}")
    if n_classes == 2:
        class_idx += 1
    print(f"class_idx after={class_idx}")
    print(f"this_pred={this_pred}")
    calib_pred = calibrator.predict(this_pred)
    print(f"calib_pred={calib_pred}")
    proba[:, class_idx] = calib_pred

if n_classes == 2:
    proba[:, 0] = 1.0 - proba[:, 1]
    
print("final proba:")
print(proba)
