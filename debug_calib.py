import numpy as np
import joblib

probs = np.array([[3.417e-10]])
p_stacked = np.vstack((1 - probs, probs)).T
print('stacked shape:', p_stacked.shape)
print('stacked:', p_stacked)

c = joblib.load('artifacts/final/calibrator.joblib')

class Dummy:
    def __init__(self):
        self.classes_ = np.array([0, 1])
    def predict_proba(self, X):
        return p_stacked

# Patch calibrator
for cc in c.calibrated_classifiers_:
    cc.estimator = Dummy()

c.estimator = Dummy()
c.classes_ = np.array([0, 1])

res = c.predict_proba([[0]])
print('calib proba:', res)

# Try directly
regressor = c.calibrated_classifiers_[0].calibrators[0]
print('direct regressor predict:', regressor.predict([probs[0, 0]]))
