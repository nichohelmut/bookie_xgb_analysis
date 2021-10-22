import datetime
import os
import time
import warnings

import google.auth
import numpy as np
import xgboost as xgb
from google.cloud import storage
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from .helpers import read_bigquery, write

warnings.filterwarnings('ignore')


class XGBAnalysis:
    def __init__(self):

        self.X_with_columns = read_bigquery('pp_X').sort_values('index1').reset_index(drop=True)
        self.X_with_columns = self.X_with_columns.loc[:, self.X_with_columns.columns != 'row_added']
        self.Z_with_columns = read_bigquery('pp_Z').sort_values('index1').reset_index(drop=True)
        self.Z_with_columns = self.Z_with_columns.loc[:, self.Z_with_columns.columns != 'row_added']

        columns_to_drop = ['index1']

        self.X_with_columns.drop(columns_to_drop, axis=1, inplace=True)
        self.Z_with_columns.drop(columns_to_drop, axis=1, inplace=True)

        self.X = np.array(self.X_with_columns)
        self.Y = read_bigquery('pp_Y').sort_values('index1').reset_index(drop=True)
        self.Y = np.array(self.Y.drop(['row_added', 'index1'], axis=1))

        self.Z = np.array(self.Z_with_columns)
        self.df_next_games = read_bigquery('pp_next_games_teams')

        self.credentials, self.project_id = google.auth.default()

    def upper_limits(self):
        print(' ')
        print('______________________________________________________________')

    def under_limits(self):
        print('______________________________________________________________')
        print(' ')

    def k_fold(self):
        kf = KFold(n_splits=4, random_state=0, shuffle=True)
        kf.get_n_splits(self.X)

        for train_index, test_index in kf.split(self.X):
            print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.Y[train_index], self.Y[test_index]

        return X_train, X_test, y_train, y_test

    def xgb_model(self):
        XGB_model = xgb.XGBClassifier(silent=False,
                                      learning_rate=0.005,
                                      colsample_bytree=0.5,
                                      subsample=0.8,
                                      objective='multi:softprob',
                                      n_estimators=1000,
                                      reg_alpha=0.2,
                                      reg_lambda=.5,
                                      max_depth=5,
                                      gamma=5,
                                      seed=82)

        return XGB_model

    def feature_importance(self, model):
        features_names = list(self.X_with_columns.columns)

        importance = np.round(model.feature_importances_, 4)
        dictionary = dict(zip(features_names, importance))
        sorted_dictionary = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
        names = []
        values = []

        self.upper_limits()
        for i in range(0, len(importance)):
            print('Feature Importance: {:35} {}%'.format(
                sorted_dictionary[i][0], np.round(sorted_dictionary[i][1] * 100, 4))
            )
            names.append(sorted_dictionary[i][0])
            values.append(np.round(sorted_dictionary[i][1] * 100, 4))
        self.under_limits()

    def xgb_predict(self, model):
        z_pred = model.predict(self.Z)
        xgb_df_next_games = self.df_next_games.copy()
        xgb_df_next_games['predicted_result'] = z_pred
        xgb_df_next_games['real_result'] = False

        return xgb_df_next_games

    def save_to_db(self, df):
        write(df, self.project_id, 'statistics', 'xgb_next_games_pred', self.credentials)
        write(df, self.project_id, 'statistics', 'xgb_total_pred', self.credentials, "append")

    def save_to_storage(self, df):
        client = storage.Client()
        bucket = client.get_bucket('xgb_next_games_pred')

        bucket.blob(f'xgb_next_games_pred_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv').upload_from_string(
            df.to_csv(index=False), 'text/csv')

    def xgb_fit_and_predict(self):
        X_train, X_test, y_train, y_test = self.k_fold()
        eval_set = [(X_train, y_train), (X_test, y_test)]
        XGB_model = self.xgb_model()

        XGB_model.fit(X_train, y_train, eval_metric=["merror", "mlogloss"], eval_set=eval_set, verbose=True)
        y_pred = XGB_model.predict(X_test)
        y_pred_train = XGB_model.predict(X_train)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy_train = accuracy_score(y_train, y_pred_train)

        self.upper_limits()
        print("XGB train Accuracy: %.2f%%" % (accuracy_train * 100.0))
        print("XGB Accuracy: %.2f%%" % (accuracy * 100.0))
        self.under_limits()

        self.feature_importance(XGB_model)
        xgb_df_next_games = self.xgb_predict(XGB_model)
        xgb_df_next_games = xgb_df_next_games.iloc[::-1].reset_index(drop=True)

        df_all = read_bigquery('df_all_sorted')
        df_all = df_all.iloc[::-1].reset_index(drop=True)

        for index, row in xgb_df_next_games.iterrows():
            odds_home = df_all.loc[index]['odds_ft_home_team_win']
            odds_draw = df_all.loc[index]['odds_ft_draw']
            odds_away = df_all.loc[index]['odds_ft_away_team_win']

            xgb_df_next_games.at[index, 'odds_ft_home_team_win'] = odds_home
            xgb_df_next_games.at[index, 'odds_ft_draw'] = odds_draw
            xgb_df_next_games.at[index, 'odds_ft_away_team_win'] = odds_away

        print(xgb_df_next_games)
        os.environ['TZ'] = 'Europe/Amsterdam'
        time.tzset()
        xgb_df_next_games["date_time"] = time.strftime('%X %x %Z')
        xgb_df_next_games['id'] = xgb_df_next_games.index
        xgb_df_next_games['index'] = xgb_df_next_games.index
        xgb_df_next_games['real_result'] = xgb_df_next_games['real_result'].astype('Int64')

        self.upper_limits()
        print(xgb_df_next_games)
        self.under_limits()
        self.save_to_db(xgb_df_next_games)
        self.save_to_storage(xgb_df_next_games)
        print(xgb_df_next_games)
        return xgb_df_next_games
