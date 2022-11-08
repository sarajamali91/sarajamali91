import matplotlib.pyplot as plt
import pandas as pd
import time
import datetime
#import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
#import tensorflow_datasets as tfds
import openml
from openml.datasets import get_dataset

# OpenML Dataset ID

metrics = []
Dataset = input('Please enter Dataset ID:')  # provide dataset id
dataset = openml.datasets.get_dataset(Dataset)

# Print a summary
print(
    f"This is dataset '{dataset.name}', the target feature is "
    f"'{dataset.default_target_attribute}'" 
     )
print(f"URL: {dataset.url}")
DATASET_NAME={dataset.name}

print(dataset.description)

# LabelEncoded Target
X, y, categorical_indicator, attribute_names = dataset.get_data(
    dataset_format="array", target=dataset.default_target_attribute)

dataset = pd.DataFrame(X, columns=attribute_names)
dataset["target"] = y

# training and test data split
data = dataset.sample(frac=0.80, random_state=1234)

data_unseen = dataset.drop(data.index)
data.reset_index(inplace=True, drop=True)
data_unseen.reset_index(inplace=True, drop=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))
#----------------------------------------------------------------------------------#
print("--------------------------------------------------------------------")

#-------------------------------------Benchmark--------------------------------------------#
#print("please check your Automl is avalible, automl(system)")

import h2o
from h2o.automl import H2OAutoML
       
MODEL_NAME = 'h2o'
            
print("-------------------H2O MODEL----------------------------------")
            
h2o.init(max_mem_size="1G")
            # Convert to h2o dataframe
hf = h2o.H2OFrame(dataset)
            # Change the column type to a factor:
hf['target'] = hf['target'].asfactor()
            # Data Transform - Split train : test datasets
train, valid = hf.split_frame(ratios = [.85], seed = 1234)

print("Training Dataset", train.shape)
print("valid Dataset", valid.shape)
            # Identify predictors and response
feature_columns = train.columns
target_column   = "target"
feature_columns.remove(target_column)
start_experiment = time.time()
print()
            # Run AutoML for YY base models (limited to 1 hour max runtime by default)
aml = H2OAutoML(max_models=5, seed=1, include_algos = ["DeepLearning"], nfolds=0,

     )

aml.train(x=feature_columns, y=target_column, training_frame = train, validation_frame = valid)

end_expriment = time.time()
print()

lb = aml.leaderboard
model = aml.leader 
print(lb.head(rows = lb.nrows))

import scikitplot as skplt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import cohen_kappa_score, confusion_matrix
import pandas as pd

hf = h2o.H2OFrame(data_unseen)

# Predict with the best model.
predicted_y_ = model.predict(hf[feature_columns])
predicted_data_ = predicted_y_.as_data_frame()

# Evaluate the skill of the Trained model
acc_                 = accuracy_score(data_unseen[target_column], predicted_data_['predict'])
classReport_         = classification_report(data_unseen[target_column], predicted_data_['predict'])
confMatrix_          = confusion_matrix(data_unseen[target_column], predicted_data_['predict']) 
        
print(); print('Testing Results of the trained model: ')
print(); print('Accuracy : ', acc_)
#print(); print('Confusion Matrix :\n', confMatrix)
print(); print('Classification Report :\n',classReport_)


cls_report_h2o= pd.DataFrame(classification_report(data_unseen[target_column], predicted_data_['predict'], output_dict=True)).transpose()
cls_report_h2o.to_csv(f'{DATASET_NAME}_H2O_Classification.csv', index= True)
print(); print('Testing Results of the trained model: ')
print(); print('Accuracy : ', acc_)
print(); print('Confusion Matrix :\n', confMatrix_)
print(); print('Classification Report :\n',classReport_)

metrics.append({
'Accuracy': round(accuracy_score(data_unseen[target_column], predicted_data_['predict']),4),
'Time_sec': round(end_expriment - start_experiment),
'Time': datetime.datetime.now(),
 })


pd.DataFrame(metrics).to_csv(f'{DATASET_NAME}_{MODEL_NAME}_metrics.csv', index=False,)

            # Confusion matrix
skplt.metrics.plot_confusion_matrix(data_unseen[target_column], predicted_data_['predict'], figsize=(9,9)); plt.show()
print(); print("Model is complete ... ... ...")
      #return  None
