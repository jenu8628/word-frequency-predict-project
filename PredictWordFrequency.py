import os.path
from builtins import enumerate
import csv
import os
from urllib.parse import unquote

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from xgboost import XGBRFRegressor


# 데이터 파싱 함수
def data_parsing():
    # uxlog_20211004_20211010 폴더안에 모든 로깅파일이 있음
    folder_list = os.listdir('uxlog_20211004_20211010')
    # 파일이름들을 받을 리스트
    file_list = {}
    for folder in folder_list:
        file = os.listdir('uxlog_20211004_20211010/{}'.format(folder))
        if folder not in file_list:
            file_list[folder] = file
        else:
            file_list[folder].append(file)

    column = ['bid', 'URL', 'IP', 'Time', 'Uxid', 'spec', 'referer']
    cnt = 0
    for key, val in file_list.items():
        for file in val:
            cnt += 1
            if key == '2021_10_04' and file == file_list["2021_10_04"][0]:
                f = open(f'uxlog_20211004_20211010/2021_10_04/{file_list["2021_10_04"][0]}', mode="r", encoding="UTF-8")
                read = csv.reader(f, delimiter=',', quotechar="'")
                df = pd.DataFrame(read)
                df = df.iloc[:, :7]
                df.columns = column

                # 날짜, 시간으로 넣기
                time = df['Time'].str.split('-')
                day_hour = time.str.get(2).str.split(' ')
                day = day_hour.str.get(0)
                hour_temp = day_hour.str.get(1).str.split(':')
                hour = hour_temp.str.get(0)
                df['day'] = day
                df['hour'] = hour
                df = pd.concat([df['URL'], df['day'], df['hour']], axis=1)

                df_fas = df['URL'].str.contains('keyword')
                df['True'] = df_fas
                df = df[df['True'] == True]
                df = df.drop(['True'], axis=1)
                keyword = df['URL'].str.split('keyword=')
                keyword_split = keyword.str.get(1).str.split('&')
                keyword_encoding = keyword_split.str.get(0)
                df['URL'] = keyword_encoding.apply(lambda x: unquote(x))

                df.to_csv('data/dateparsing.csv', mode='w', index=False)
            else:
                f = open(f'uxlog_20211004_20211010/{key}/{file}', mode="r", encoding="UTF-8")
                read = csv.reader(f, delimiter=',', quotechar="'")
                df = pd.DataFrame(read)
                df = df.iloc[:, :7]
                df.columns = column
                url = df['URL']

                # 날짜, 시간으로 넣기
                time = df['Time'].str.split('-')
                day_hour = time.str.get(2).str.split(' ')
                day = day_hour.str.get(0)
                hour_temp = day_hour.str.get(1).str.split(':')
                hour = hour_temp.str.get(0)
                df['day'] = day
                df['hour'] = hour
                df = pd.concat([df['URL'], df['day'], df['hour']], axis=1)
                # URL keyword있는 값만 추출 및 디코딩
                df_fas = df['URL'].str.contains('keyword')
                df['True'] = df_fas
                df = df[df['True'] == True]
                df = df.drop(['True'], axis=1)
                keyword = df['URL'].str.split('keyword=')
                keyword_split = keyword.str.get(1).str.split('&')
                keyword_encoding = keyword_split.str.get(0)
                keyword_encoding = keyword_encoding.apply(lambda x: str(x))
                df['URL'] = keyword_encoding.apply(lambda x: unquote(x))

                with open('data/dateparsing.csv', 'a', encoding="UTF-8") as s:
                    df.to_csv(s, mode='a', header=False, index=False)


# 데이터 전처리 클래스
class PreProcessor:
    def __init__(self, data):
        self.data = data

    # 전처리
    def conversion(self):
        # 라벨 인코딩
        data = self.data.copy()
        # 1. 결측치 제거
        data = data.dropna()
        # 컬럼이름 변경
        data.rename(columns={'URL': 'keyword'}, inplace=True)
        # day가 4(월)부터 시작하므로 1(월)부터 시작하도록 변경
        data['day'] = data['day'].map(lambda x: x - 3)

        # 디코딩안된 키워드 및 한글 깨짐 제거
        def delete_keyword(x):
            if '%' in x or '�' in x or '𵵸' in x:
                return None
            elif 'www' in x:
                return None
            else:
                return x
        # 키워드의 + 를 공백으로 변경
        def plus_keyword(x):
            data = x.replace('+', ' ')
            return data
        data['keyword'] = data['keyword'].apply(lambda x: delete_keyword(x))
        data = data.dropna()
        data['keyword'] = data['keyword'].apply(lambda x: plus_keyword(x))

        keyword_dict = data['keyword'].value_counts(ascending=True).to_dict()

        # frequency(빈도수) 컬럼 생성
        # day, time, keyword 그룹화를 통한 빈도수 구하기
        groups = data.groupby(['day', 'hour', 'keyword'])
        # group_dict : {(날짜, 시간, 단어) : 빈도수} 로 이루어진 딕셔너리
        group_dict = groups.size().to_dict()
        # 빈도수 담을 리스트생성
        frequency = []
        for key, val in group_dict.items():
            frequency.append(val)
        temp = list(group_dict)
        # zip함수를 이용해 day, time, keyword 추출
        day, time, keyword = zip(*temp)
        day = list(day)
        time = list(time)
        keyword = list(keyword)
        new_dict = {'day': day, 'time': time, 'keyword': keyword, 'frequency': frequency}
        # 새로운 데이터 프레임 생성
        data = pd.DataFrame(new_dict)
        data = data.drop_duplicates()

        # 키워드를 빈도수 오름차순으로 정렬
        # {키워드 : 빈도수에 따른 순서}로 딕셔너리 만들기
        # 키워드를 정수로 변환할 때 빈도수가 높으면 키워드의 숫자가 높다는 것을  통해
        # 키워드와 빈도수 간의 상관관계를 주기 위함.
        # 해당 딕셔너리를 통해 keyword를 정수로 변환
        def apply_func(x):
            return keyword_dict[x]
        data['keyword'] = data['keyword'].map(lambda x: apply_func(x))
        data = pd.get_dummies(columns=['day'], data=data)
        data.to_csv('data/preprocessor.csv', mode='w', index=False)
        # with open('keyword_dict.pickle', 'wb') as f:
        #     pickle.dump(keyword_dict, f)

    # train데이터와 test데이터 분해 및 저장
    def distribute(self):
        # target과 data로 분해
        target = self.data['frequency'].copy()
        data = self.data.drop('frequency', axis=1)
        # sklearn의 train_test_split을 통한 데이터 분해
        train_input, test_input, train_target, test_target = train_test_split(data, target, test_size=0.01, shuffle=True)
        # train, tes데이터 생성 및 저장
        train = pd.concat([train_input, train_target], axis=1)
        test = pd.concat([test_input, test_target], axis=1)
        train.to_csv('data/train.csv', mode='w', index=False)
        test.to_csv('data/test.csv', mode='w', index=False)


# 데이터 분석 및 시각화 클래스
class DataAnalyzer:
    def __init__(self, data):
        self.data = data

    plt.rcParams['font.family'] = 'Malgun Gothic'

    # 요일 별 검색 빈도수
    def day_hist(self):
        # 요일 별 검색 빈도수는 큰 차이가 없습니다.
        # print(data['day'].value_counts())
        plt.hist(self.data['day'], bins=7)
        plt.show()

    # 시간대별 키워드들 빈도수
    def hour_keyword_hist(self):
        data = self.data
        # 빈도수 별 키워드 정렬
        data = data.sort_values('frequency', ascending=False)
        keyword_index = data['keyword'].index
        arr = []
        idx = 0
        i = 0
        # 빈도수 보기
        while len(arr) < 12:
            plt.subplot(4, 3, i+1)
            if data['keyword'][keyword_index[idx]] in arr:
                idx += 1
                continue
            else:
                # group_url = data[data['keyword'] == keyword_index[idx]]
                group_keyword = data[data['keyword'] == data['keyword'][keyword_index[idx]]]
                arr.append(data['keyword'][keyword_index[idx]])
                plt.bar(group_keyword['time'], group_keyword['frequency'])
                plt.xlabel('시간')
                plt.ylabel(data['keyword'][keyword_index[idx]])
                idx += 1
                i += 1


        # 상관관계 그래프
        # sns.heatmap(data=self.data.corr(), annot=True)
        plt.show()

    # 키워드 + 시간대 별 빈도수 - 요일 별로 시각화
    def keyword_time_day_hist(self, keyword):
        # 0~23시 까지 keyword 빈도수 그래프
        # 들어온 keyword에 해당하는 행만 추출
        data = self.data[self.data['keyword'] == keyword].copy()
        # keyword 제외
        data = data.drop('keyword', axis=1)
        # day로 묶고, 해당 시간의 빈도수를 보여줘!
        # 1~8(월~금 까지 7개를 보여줌)
        day_dict = ['Mon', 'Tue', 'Wen', 'Thu', 'Fri', 'Sat', 'Sun']
        for i in range(1, 8):
            day_data = data[data['day'] == i].copy()
            day_data = day_data.drop('day', axis=1)
            plt.plot(day_data['time'], day_data['frequency'])
            plt.xlabel('시간')
            plt.ylabel(day_dict[i-1])
            plt.savefig('static/image/' + keyword + str(i) + '.png')
            plt.clf()
        # plt.show()

    # 단어와 빈도수의 분포도
    def keyword_frequency(self):
        data = self.data.copy()
        data = data.drop(['day', 'time'], axis=1)
        # print(data)
        data = data.groupby('keyword')
        print(data.size())

    # 시간대 별 최고 빈도수 단어
    def time_keyword(self):
        except_keyword = ['bitway', 'BITWAY', 'buyus']
        for i in range(1, 8):
            plt.rcParams['font.family'] = 'Malgun Gothic'
            day_dict = ['Mon', 'Tue', 'Wen', 'Thu', 'Fri', 'Sat', 'Sun']
            day_data = self.data.copy()
            day_data = day_data[day_data['day'] == i].drop(['day'], axis=1)
            x = []
            y = []
            if i == 7:
                plt.subplot(3, 3, 8)
            else:
                plt.subplot(3, 3, i)
            for j in range(24):
                time_data = day_data.copy()
                time_data = time_data[time_data['time'] == j].drop(['time'], axis=1)
                time_data = time_data.sort_values(by=['frequency'], ascending=False)
                for k in range(4):
                    if time_data.iloc[k, :]['keyword'] not in except_keyword:
                        x.append(str(j) + time_data.iloc[k, :]['keyword'])
                        y.append(time_data.iloc[k, :]['frequency'])
                        break
            plt.plot(x, y)
            plt.title(day_dict[i-1])
            plt.xticks(rotation=90)
            plt.ylabel('빈도수')
        plt.show()



class ModelFactory:
    def __init__(self, train_input, train_target):
        self.train_input = train_input
        self.train_target = train_target

    def RandomForestRegressor(self, model_obj=None):
        if model_obj == None:
            model = RandomForestRegressor()
        else:
            model = model_obj
        model.fit(self.train_input, self.train_target)
        return model

    def XGBRFRegressor(self, model_obj=None):
        if model_obj == None:
            model = XGBRFRegressor()
        else:
            model = model_obj
        model.fit(self.train_input, self.train_target)
        return model

    def LinearRegression(self, model_obj=None):
        if model_obj == None:
            model = LinearRegression()
        else:
            model = model_obj
        model.fit(self.train_input, self.train_target)
        return model

# chunk를 통해 데이터를 나눠서 학습!
def chunk_model(data):
    train_target = data['frequency'].copy()
    train_input = data.drop('frequency', axis=1)
    ml = ModelFactory(train_input, train_target)
    # 랜덤 포레스트
    if os.path.isfile('pickle/random_forest.pickle'):
        with open('pickle/random_forest.pickle', 'rb') as f:
            rf_model = pickle.load(f)
        rf = ml.RandomForestRegressor(rf_model)
    else:
        rf = ml.RandomForestRegressor()
    with open('pickle/random_forest.pickle', 'wb') as f:
        pickle.dump(rf, f)

    # xgbm
    if os.path.isfile('pickle/XGBM.pickle'):
        with open('pickle/XGBM.pickle', 'rb') as f:
            xgb_model = pickle.load(f)
        xgb = ml.XGBRFRegressor(xgb_model)
    else:
        xgb = ml.XGBRFRegressor()
    with open('pickle/XGBM.pickle', 'wb') as f:
        pickle.dump(xgb, f)

    # 선형회귀
    if os.path.isfile('pickle/line_regressor.pickle'):
        with open('pickle/line_regressor.pickle', 'rb') as f:
            lr_model = pickle.load(f)
        lr = ml.LinearRegression(lr_model)
    else:
        lr = ml.LinearRegression()
    with open('pickle/line_regressor.pickle', 'wb') as f:
        pickle.dump(lr, f)

# 예측함수
def predict(test_input):
    with open('pickle/random_forest.pickle', 'rb') as f:
        rf = pickle.load(f)
    with open('pickle/XGBM.pickle', 'rb') as f:
        xgb = pickle.load(f)
    with open('pickle/line_regressor.pickle', 'rb') as f:
        lr = pickle.load(f)

    rf_predict = rf.predict(test_input)
    xgb_predict = xgb.predict(test_input)
    lr_predict = lr.predict(test_input)
    # 평균으로 통합
    predicted = (rf_predict + xgb_predict + lr_predict) / 3
    predicted = np.round(predicted, 0)
    # 예측 값
    return np.array(predicted)


if __name__ == '__main__':
    # 7524314
    # 데이터 전처리
    # data = pd.read_csv('data/dateparsing.csv')
    # pre = PreProcessor(data)
    # pre.conversion()
    # #  train, test 나누기
    # data = pd.read_csv('data/preprocessor.csv')
    # pre = PreProcessor(data)
    # pre.distribute()

    # data = pd.read_csv('data/preprocessor_analysis.csv')
    # with open('pickle/keyword_graph.pickle', 'wb') as f:
    #     pickle.dump(data, f)
    # da = DataAnalyzer(data)


    # for index, data in enumerate(pd.read_csv('data/train.csv', chunksize=10000)):
    #     if index % 20 == 0:
    #         print('index : ', index)
    #     chunk_model(data)
    # print('모델 학습 종료')


    # # 테스트 파일 불러오기
    # test = pd.read_csv('data/test.csv')
    # test_target = test['frequency'].copy()
    # test_input = test.drop('frequency', axis=1)
    # # 예측 값 구하기
    # predicted = predict(test_input)
    # # 실제 값
    # test_target = np.array(test_target)
    # # 평균제곱오차 구하기
    # mse = (np.square(test_target - predicted).sum()) / len(test_target)
    # print(mse)
    # accuracy = r2_score(test_target, predicted, sample_weight=None, multioutput='uniform_average')

    # with open('flask/pickle/keyword_dict.pickle', 'rb') as f:
    #     keyword_dict = pickle.load(f)
    #
    # plt.scatter(keyword_dict.keys(), keyword_dict.values())
    # plt.show()
    # data = pd.read_csv('data/preprocessor_analysis.csv')
    # da = DataAnalyzer(data)
    # da.time_keyword()
