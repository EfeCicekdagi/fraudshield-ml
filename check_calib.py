import sys
sys.path.insert(0, 'src')
import joblib
import pandas as pd
from fraudshield.inference.predictor import FraudPredictor
from sklearn.base import BaseEstimator
from fraudshield.pipelines.finalization_pipeline import MLPPipeline

class Dummy(BaseEstimator):
    pass

MLPPipeline.__sklearn_tags__ = Dummy.__sklearn_tags__

c = joblib.load('artifacts/final/calibrator.joblib')
df = pd.DataFrame({'step':[1],'type':['PAYMENT'],'amount':[100],'oldbalanceOrg':[100],'nameOrig':['C'],'nameDest':['M']})
p = FraudPredictor()
df_f = p._build_features(df)

print(c.predict_proba(df_f))
