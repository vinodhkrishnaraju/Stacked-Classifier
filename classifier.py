from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
import pandas as pd
from sklearn.cross_validation import StratifiedKFold
from stub import TaggingModel
from configuration_generator import ConfigGenerator


class CompositeModel(TaggingModel):
    def __init__(self):
        """ Composite model with two stack implementation and age model """

        self.fitted_columns = None
        self.y_clfstack1 = {
            'clf1': LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.01,
                                       fit_intercept=True, intercept_scaling=1,
                                       class_weight=None, random_state=None,
                                       solver='liblinear', max_iter=4,
                                       verbose=0, warm_start=False, n_jobs=-1),

            'clf2': RandomForestClassifier(n_estimators=8, criterion='gini', max_depth=12,
                                           min_samples_split=500, min_samples_leaf=300, min_weight_fraction_leaf=0.0,
                                           max_features=250, max_leaf_nodes=None, bootstrap=False,
                                           oob_score=False, n_jobs=-1, random_state=None, verbose=0, warm_start=False,
                                           class_weight=None),

            'clf3': ExtraTreesClassifier(bootstrap=False, class_weight=None, criterion='entropy',
                                         max_depth=10, max_features=150, max_leaf_nodes=None,
                                         min_samples_leaf=100, min_samples_split=2,
                                         min_weight_fraction_leaf=0.0, n_estimators=8, n_jobs=-1,
                                         oob_score=False, random_state=None, verbose=0, warm_start=False),

            'clf4': GradientBoostingClassifier(init=None, learning_rate=0.01, loss='exponential',
                                               max_depth=15, max_features=250, max_leaf_nodes=None,
                                               min_samples_leaf=100, min_samples_split=500,
                                               min_weight_fraction_leaf=0.0, n_estimators=8,
                                               presort='auto', random_state=None, subsample=1, verbose=0,
                                               warm_start=False)
        }

        self.y_clfstack2 = {
            'clf1': GradientBoostingClassifier(loss='deviance', learning_rate=0.03, n_estimators=4,
                                               subsample=1.0, min_samples_split=2, min_samples_leaf=1,
                                               min_weight_fraction_leaf=0.0, max_depth=5, init=None,
                                               random_state=None, max_features=100, verbose=0,
                                               max_leaf_nodes=None, warm_start=False, presort='auto'),

            'clf2': RandomForestClassifier(n_estimators=4, criterion='entropy',
                                           max_depth=10, min_samples_split=2,
                                           min_samples_leaf=25, min_weight_fraction_leaf=0.0,
                                           max_features=50, bootstrap=True, max_leaf_nodes=None,
                                           oob_score=False, n_jobs=-1, random_state=None,
                                           verbose=0, warm_start=False, class_weight=None),

            'clf3': LogisticRegression(penalty='l1', dual=False, tol=0.0001, C=0.01,
                                       fit_intercept=True, intercept_scaling=1,
                                       class_weight=None, random_state=None,
                                       solver='liblinear', max_iter=4,
                                       verbose=0, warm_start=False, n_jobs=1)

        }

    def validate_predictors(self, data, labels=None):
        """ Check if all predictors are in training dataset """

        # validate should be done based on fitted columns not just on age_model
        columns = set(self.age_model_columns)
        if len(columns.difference(data.columns)) > 0:
            raise ValueError("Following columns missing in training dataset: %s" % columns.difference(data.columns))
        if labels is not None:
            if any(data.index != labels.index):
                raise ValueError("Row count mismatch in predictors and response")

    def fit(self, X, y):
        """
        fit age_mode, adult_model, and youth_model
        :param X: DataFrame, containing all the columns mentioned in age_model_column_list, adult_model_columns,
            youth_model_columns
        :param y: predicted segments (A1, A2, Y1, Y2, ...) in the same order as X
        :return: self
        """

        if X.index.difference(y.index).shape[0] > 0:
            raise ValueError('Train set index mismatch label index')

        y = y.loc[X.index]
        self.validate_predictors(X, y)
        age_label = y.apply(lambda x: 1 if x > 4 else 0)
        y_axis_label = y.apply(lambda x: 1 if x in [1, 2, 5, 6] else 0)
        x_axis_label = y.apply(lambda x: 1 if x in [1, 3, 5, 7] else 0)
        self.fitted_columns = X.columns

        # Fit Function for Y axis
        counter = 1
        first_valid_y = pd.DataFrame()
        skf = StratifiedKFold(y_axis_label, n_folds=5)
        for train_index, test_index in skf:
            print 'Y axis iteration: ', counter, ':'
            counter += 1
            x_train, x_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y_axis_label.iloc[train_index], y_axis_label.iloc[test_index]

            first_test_y = pd.DataFrame(index=x_test.index)

            for nm, clf in self.y_clfstack1.items():
                clf.fit(x_train, y_train)
                column_list = [nm + '_' + class_label for class_label in ['Down', 'Top']]
                x_valid = pd.DataFrame(clf.predict_proba(x_test), index=x_test.index, columns=column_list)
                first_test_y = pd.concat([first_test_y, x_valid.iloc[:, 1:2]], axis=1)

            first_valid_y = pd.concat([first_valid_y, first_test_y], axis=0)

        first_valid_y['avg'] = first_valid_y.mean(axis=1)
        append_columns = self.append_columns + self.dummy_coded_attributes
        first_valid_y = pd.concat([first_valid_y, X.loc[first_valid_y.index, append_columns]], axis=1)

        for nm, clf in self.y_clfstack2.items():
            clf.fit(first_valid_y, y_axis_label)

       
        return self

    def ensure_columns(self, X):
        missing_columns = self.fitted_columns.difference(X.columns)
        for col in missing_columns:
            X[missing_columns] = 0
        return X[self.fitted_columns]

    def predict_proba(self, X):
        """
        Predict segments for the data points in given data set which belong to any one of the SubGroup.
        Data points not belonging to any SubGroup is 'UNKNOWN' segment.
        """

        X = self.ensure_columns(X)
        self.validate_predictors(X)

        # Predict_proba for Top/Down axis #
        first_valid_y = pd.DataFrame(index=X.index)
        for nm, clf in self.y_clfstack1.items():
            column_list = [nm + '_' + class_label for class_label in ['Down', 'Top']]
            x_valid = pd.DataFrame(clf.predict_proba(X), index=X.index, columns=column_list)
            first_valid_y = pd.concat([first_valid_y, x_valid.iloc[:, 1:2]], axis=1)

        first_valid_y['avg'] = first_valid_y.mean(axis=1)
        append_columns = self.append_columns + self.dummy_coded_attributes
        first_valid_y = pd.concat([first_valid_y, X.loc[first_valid_y.index, append_columns]], axis=1)

        second_valid_y = pd.DataFrame(index=first_valid_y.index)
        for nm, clf in self.y_clfstack2.items():
            column_list = [nm + '_' + class_label for class_label in ['Down', 'Top']]
            x_valid = pd.DataFrame(clf.predict_proba(first_valid_y), index=first_valid_y.index, columns=column_list)
            second_valid_y = pd.concat([second_valid_y, x_valid.iloc[:, 1:2]], axis=1)


        segment_proba_df = pd.DataFrame(index=X.index)
        segment_proba_df['top_prob'] = (second_valid_y['clf1_Top'] + second_valid_y['clf2_Top'] + second_valid_y[
            'clf3_Top']) / 3
        segment_proba_df = segment_proba_df.fillna(0)
        return segment_proba_df
        

    def export_csv(self, segment_df):
        """ Export probabilities and predicted segment into csv file """

        def sort_proba(row):
            """ Returns column name of rowwise sorted probabilities """

            if row.isnull().sum().sum() > 0:
                return None
            else:
                return row.sort_values(ascending=False).index[0]

        if len(segment_df.index):
            segment_df['PREDICTED_SEG'] = segment_df[['A1', 'Y1', 'UNKNOWN']].apply(sort_proba, axis=1)
        else:
            segment_df['PREDICTED_SEG'] = None

        segment_df_export = segment_df[['A1', 'Y1', 'UNKNOWN', 'PREDICTED_SEG']]
        segment_df_export['fold'] = pd.Series(segment_df.shape[0], index=segment_df.index)
        segment_df_export.to_csv('tmp/segment_df_export.csv', header=False, mode='a')


    def predict(self, X):
        """ Predicts the final segment labels """

        def sort_proba(row):
            """ Returns column name of row wise sorted probabilities """
            return row.sort_values(ascending=False).index[0]

        segment_proba_df = self.predict_proba(X)


        segment_proba_df['actual_x_bin'] = segment_proba_df['right_prob'].apply(
                lambda x:
                'Bin8' if x > 0.7109
                else 'Bin7' if x > 0.66819
                else 'Bin6' if x > 0.61036
                else 'Bin5' if x > 0.54228
                else 'Bin4' if x > 0.4010
                else 'Bin3' if x > 0.36098
                else 'Bin2' if x > 0.3171
                else 'Bin1' if x >= 0
                else 'NULL'
        )

        segment_proba_df['actual_y_bin'] = segment_proba_df['top_prob'].apply(
                lambda x:
                'Bin8' if x > 0.74967
                else 'Bin7' if x > 0.69224
                else 'Bin6' if x > 0.63199
                else 'Bin5' if x > 0.5449
                else 'Bin4' if x > 0.35556
                else 'Bin3' if x > 0.30243
                else 'Bin2' if x > 0.25694
                else 'Bin1' if x >= 0
                else 'NULL'
        )

        segment_proba_df['segment'] = segment_proba_df.apply(lambda x: self.getSegment(x), axis=1)
        return segment_proba_df['segment']
