import os
from azureml.core import Workspace
from azureml.core.model import Model 

ws = Workspace.from_config()

model = Model.register(workspace = ws, model_path = 'logistic_model.pkl',
                       model_name = 'CreditRiskLogReg')