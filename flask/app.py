from flask import Flask
from flask import request, render_template
import pickle
import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import base64

app = Flask(__name__)

# 모델 가져오기
with open('pickle/random_forest.pickle', 'rb') as f:
    rf = pickle.load(f)
with open('pickle/XGBM.pickle', 'rb') as f:
    xgb = pickle.load(f)
with open('pickle/line_regressor.pickle', 'rb') as f:
    lr = pickle.load(f)

# 기존의 데이터의 모든 keyword: 빈도수 dict
with open('pickle/keyword_dict.pickle', 'rb') as f:
    keyword_dict = pickle.load(f)


@app.route('/')
def main():
    return render_template('main.html')


# 예측에 필요한 데이터 반환
def parse_search_request(request):
    # time, day, keyword 가져오기
    time = int(request.form.get('time'))
    hour = int(request.form.get('hour'))
    hour = hour + (12 * time)
    day = request.form.get('day')
    keyword = request.form.get('keyword')
    return time, hour, day, keyword


# 데이터 프레임 생성
def create_dataframe(day, time, keyword_num):
    test_input = {
        'day': [int(day)],
        'time': [int(time)],
        'keyword': [keyword_num]
        }
    column = list(test_input.keys())
    df = pd.DataFrame(test_input, columns=column)
    infologger.info(df)
    return df


# 예측값 반환 함수
def predict(df):
    rf_predict = rf.predict(df)
    xgb_predict = xgb.predict(df)
    lr_predict = lr.predict(df)
    predicted = (rf_predict + xgb_predict + lr_predict) / 3
    return int(np.round(predicted, 0)[0])


# 검색 get요청시
@app.route(rule='/getsearch/', methods=['GET'])
def get_searchpage():
    return render_template('getSearch.html')


# 검색 post요청시
@app.route(rule='/postsearch/', methods=['POST'])
def post_searchpage():
    # time, day, keyword
    time, hour, day, keyword = parse_search_request(request)
    # 키워드의 인트형 변환
    keyword_num = keyword_dict[keyword] if keyword in keyword_dict else 0
    
    # 예측할 데이터프레임 생성
    df = create_dataframe(hour, day, keyword_num)

    # 모델 예측
    predicted = predict(df)
    infologger.info('predict : '+ str(predicted))

    # web에서 요일 한글로 보여주기 위한 list
    day_list = ['월', '화', '수', '목', '금', '토', '일']
    day_dived = ['오전', '오후']

    return render_template('postSearch.html',
    predict=predicted,
    keyword=keyword,
    day = day_list[int(day)],
    time = day_dived[time],
    hour = hour
    )


if __name__ == '__main__':
    # DEBUG < INFO < WARNING < ERROR < CRITICAL
    infologger = logging.getLogger()
    infologger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_hander = logging.StreamHandler()
    stream_hander.setFormatter(formatter)
    infologger.addHandler(stream_hander)

    file_handler = logging.FileHandler('my.log')
    file_handler.setFormatter(formatter)
    infologger.addHandler(file_handler)
    infologger.info('start')

    app.run(host='0.0.0.0', port=5000, debug=True)