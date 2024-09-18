import glob
import os
import numpy as np
import pandas as pd
from lib.algorithm import IterativeShrinkage

def main():
    #学習済みパラメータ読込み
    parms = np.load(file="./sparsemodeling-cox-regression/parms.npz")
    x = parms["arr_0"]
    ave_A = parms["arr_1"]
    l2_norm_A = parms["arr_2"]

    #データ読み込み
    data = pd.read_csv("./sparsemodeling-cox-regression/csv/pbc_data.csv", encoding='utf-8')


    #不要なカラム削除/カラム値変更
    data.drop(columns=['Unnamed: 0','id'], inplace=True)
    data.replace({'sex': {'m': 0, 'f': 1}},inplace=True)
    data.replace({'status': {1: 0, 2: 1}},inplace=True)
    
    #欠損値の削除
    data = data.dropna()
    oo = data['status']
    data.drop(columns=['status','time'], inplace=True)
    
    #説明変数の行列A（nx17 データ数xパラメータ次元数
    A = data.to_numpy()    

    #標準化（列方向の平均0、2乗和平均1 学習済みパラメータを使用）
    A = (A - ave_A)/l2_norm_A

    #良否判定
    y = np.exp(np.dot(A, x))

    #csv出力
    df = pd.DataFrame(y, columns=['output'])
    df['status'] = oo
    df.to_csv('./score_logistic.csv', header=True)


if __name__ == "__main__":
    main()