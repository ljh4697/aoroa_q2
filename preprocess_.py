import pandas as pd
import numpy as np
import holidays
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
import matplotlib.pyplot as plt

def preprocess(data_path):
    data_set = pd.read_excel(data_path)
    # 중복값 제거
    data_set.drop_duplicates(inplace=True)
    data_set.shape
    data_set.info()
    selected_for_x = ['HSCODE', 'is_coc', 'cargo_weight', 'year', 'month', 'day', 'teu']
    target = ['paid_amount']
    data_set['port_of_loading'].value_counts()
    data_set['port_of_discharge'].value_counts()
    data_set['expected_time_of_departure'] = pd.to_datetime(data_set['expected_time_of_departure'])

    def is_holi(x):
        kr_holi = holidays.KR()

        return 1 if x in kr_holi else 0
    def ec_coc(x):
        return 1 if x else 0

    selected_for_x.append('kr_holiday')

    print('preprocessing process .....')
    tqdm.pandas()
    data_set['kr_holiday'] = data_set['expected_time_of_departure'].progress_map(lambda x : is_holi(x))
    tqdm.pandas()
    data_set['is_coc'] = data_set['is_coc'].progress_map(lambda x : ec_coc(x))
    print('preprocessing Done!')


    # 년 월 일 매핑으로 전처리
    data_set['year'] = data_set['expected_time_of_departure'].dt.year
    data_set['month'] = data_set['expected_time_of_departure'].dt.month
    data_set['day'] = data_set['expected_time_of_departure'].dt.day

    # target variable plot (이상치 검증)
    data_set['paid_amount'].plot.box()
    plt.show()

    ohe_encoders = {}

    for c in ['port_of_loading', 'port_of_discharge']:
        ohe_encoders[c] = OneHotEncoder(sparse_output=False)
        data_set = pd.concat([data_set, pd.DataFrame(ohe_encoders[c].fit_transform(data_set[[c]]), columns = [c+'_ohe_' + str(i) for i in range(len(data_set[c].value_counts().keys()))])], axis=1)
        selected_for_x += [c+'_ohe_' + str(i) for i in range(len(data_set[c].value_counts().keys()))]
        
    return data_set, selected_for_x, target
