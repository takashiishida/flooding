import numpy as np
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking.client import MlflowClient


class MLFlowAnalysis:
    def __init__(self, run_id, tracking_uri):
        self.run_id = run_id
        self.tracking_uri = tracking_uri

    def prepare_data(self):
        trLss = self.get_history(metric='trLss')
        vaLss = self.get_history(metric='vaLss')
        teLss = self.get_history(metric='teLss')
        trAcc = self.get_history(metric='trAcc')
        vaAcc = self.get_history(metric='vaAcc')
        teAcc = self.get_history(metric='teAcc')
        return trLss, vaLss, teLss, trAcc, vaAcc, teAcc
        
    def prepare_clean_data(self):
        trclLss = self.get_history(metric='trclLss')
        vaclLss = self.get_history(metric='vaclLss')
        teclLss = self.get_history(metric='teclLss')
        trclAcc = self.get_history(metric='trclAcc')
        vaclAcc = self.get_history(metric='vaclAcc')
        teclAcc = self.get_history(metric='teclAcc')
        return trclLss, vaclLss, teclLss, trclAcc, vaclAcc, teclAcc    
        
    def get_history(self, metric):
        """Return metric history"""
        # mltrack = mlflow.tracking.MlflowClient(tracking_uri='sqlite:///mlflow.db')
        mltrack = mlflow.tracking.MlflowClient(tracking_uri=self.tracking_uri)
        hist = mltrack.get_metric_history(run_id=self.run_id, key=metric)
        mhist = []
        for h in hist:
            mhist.append(h.value)
        return mhist

    def get_best(self, vaMet, teMet, minmax, earlyStopping=True):
        """
        vaMet (list): validation metric
        teMet (list): test metric
        minmax (str): choose from 'max' or 'min'.  Usually you would choose max for validation accuracy and 'min' for validation loss.
        earlyStopping (bool, optional): True will get test score from epoch of the best validation score, else just get the last test epoch
        """
        if earlyStopping:
            if minmax == 'max':
                return np.max(vaMet), teMet[np.argmax(vaMet)]
            if minmax == 'min':
                return np.min(vaMet), teMet[np.argmin(vaMet)]
        else:
            return vaMet[-1], teMet[-1]