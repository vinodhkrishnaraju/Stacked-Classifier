from sklearn.cross_validation import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.externals import joblib
from model_pipeline import getModelInstance, getTransformInstance
import numpy as np
import pandas as pd

'''
Model Mgmt Platform MMP gets instance of the Composite Model implementation.
Does validation and stores metrics
Learns model
Serializes the learned model
'''



def validation(X, y, model):
    """Run stratified k fold validation and print metrics"""

    skf = StratifiedKFold(y, n_folds=3)
    for train_index, test_index in skf:
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train.copy(), y_train.copy())
        model_prob = model.predict(X_test.copy())
        y_pred = model_prob
        y_test = y_test[y_pred.index]
        print(classification_report(y_test, y_pred))
        print('Accuracy: {:.2%}'.format(accuracy_score(y_test, y_pred)))
        print(confusion_matrix(y_test, y_pred))
        model.predict_proba(X_test.copy()).to_csv('tmp/phase3.csv',header=True)


def fit_model(x_train, y_train, apply_model, dump_model=False, metrics=False, model_name='ALL'):
    """Fits model and dump with metrics"""

    apply_model.fit(x_train.copy(), y_train.copy())
    if dump_model:
        joblib.dump(apply_model, 'fitted_model/mmp_phase1_D2.clf')

    if metrics:
        model_prob = apply_model.predict(x_train.copy())
        y_pred = model_prob
        y_test = y_train.loc[y_pred.index]
        y_test = model_type(y_test, model_name)
        print(classification_report(y_test, y_pred))
        print('Accuracy: {:.2%}'.format(accuracy_score(y_test, y_pred)))
        print(confusion_matrix(y_test, y_pred))


def predict_model():
    """Load model and predict"""

    fitted_model = joblib.load('fitted_model/mmp_phase1_D2.clf')
    x_predict = pd.read_csv('data/X_train_v2.csv', index_col='msisdn').replace('\N', np.nan)
    x_predict = x_predict.convert_objects(convert_numeric=True)
    x_predict = x_predict.query('segment in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)')
    df = fitted_model.predict(x_predict)
    fitted_model.predict_proba(x_predict.copy()).to_csv('tmp/x_predict.csv', header=True)
    print(df.shape)


def data_transform(x_train):
    """Data Transform steps"""

    transform = getTransformInstance()
    x_train_transformed = transform.fit_transform(x_train)
    return x_train_transformed


def data_import():    
    x_train = pd.read_csv('data/Input_data.csv', index_col='sno', low_memory=False).replace('\N', np.nan)
    print('Imported Data shape:---->', x_train.shape)
    duplicate_index = x_train.index.duplicated()
    x_train = x_train.loc[~duplicate_index]

    
    xy_axis = pd.read_csv('data/Second_data.csv', index_col='sno').replace('\N', np.nan)
    xy_axis = xy_axis.convert_objects(convert_numeric=True)
    print('Imported Data shape:---->', xy_axis.shape)
    x_train = xy_axis.join(x_train, how='inner')
    print('Imported Data shape:---->', x_train.shape)

    duplicate_index = x_train.index.duplicated()
    x_train = x_train.loc[~duplicate_index]

    y_label = x_train.apply(
            lambda x:
            1 if x['segment'] < 4.0 and x['Y_axis'] > 8 and x['X_axis'] > 8  
            else 2 if x['segment'] < 4.0 and x['Y_axis'] > 8 >= x['X_axis']  
            else 3 if x['segment'] < 4.0 and x['Y_axis'] <= 8 < x['X_axis']  
            else 4 if x['segment'] < 4.0 and x['Y_axis'] <= 8 and x['X_axis'] <= 8  
            else 5 if x['segment'] >= 4.0 and x['Y_axis'] > 8 and x['X_axis'] > 8  
            else 6 if x['segment'] >= 4.0 and x['Y_axis'] > 8 >= x['X_axis']  
            else 7 if x['segment'] >= 4.0 and x['Y_axis'] <= 8 < x['X_axis']  
            else 8,  
            axis=1
    )

    xonitor_columns = ['Y_axis', 'X_axis']
    x_train.drop(xonitor_columns, axis=1, inplace=True)
    x_train.drop('segment', axis=1, inplace=True)


    with open('tmp/datatypes.txt','w') as fout:
        for item in x_train.dtypes:
            fout.write("%s\n" % item)
    print x_train.dtypes
    return x_train, y_label




if __name__ == '__main__':
    model = getModelInstance()
    train, label = data_import()
    # model.fit(train, label)
    # joblib.dump(model, 'model_dump/model_with_ensure_cols_reduced_estimator.clf')
    # from testing import partition_parser
    # with open('data/pahe3_part.csv') as fin:
    #     data_unknown = partition_parser(fin)
    #
    # model.predict_proba(data_unknown)

    # joblib.dump(model, 'model_dump/phasethree_model.clf')

    validation(train, label, model)
    # predict_model()
    # train_transformed = data_transform(train)
