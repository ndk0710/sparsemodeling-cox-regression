import glob
import os
import pandas as pd
import numpy as np
from lib.algorithm_v2 import IterativeShrinkage
import matplotlib.pyplot as plt


#標準化（列方向の平均0、2乗和平均1）
def normalization(A):
    ave_A = np.mean(A, axis=0, keepdims=True)
    B = A - ave_A
    l2_norm_A = np.linalg.norm(B, axis=0, keepdims=True)
    Norm_A = B/l2_norm_A
    return Norm_A, ave_A, l2_norm_A

#標準化（列方向の平均0、標準偏差1）
def normalization2(A):
    ave_A = np.mean(A, axis=0, keepdims=True)
    std_A = np.std(A, axis=0, keepdims=True)
    Norm_A = (A - ave_A)/std_A
    return Norm_A, ave_A, std_A

def main():
    #データ読み込み
    data = pd.read_csv("./sparsemodeling-cox-regression/csv/pbc_data.csv", encoding='utf-8')

    #不要なカラム削除/カラム値変更
    data.drop(columns=['Unnamed: 0','id'], inplace=True)
    data.replace({'sex': {'m': 0, 'f': 1}},inplace=True)
    data.replace({'status': {1: 0, 2: 1}},inplace=True)
    
    #欠損値の削除
    data = data.dropna()

    #カラムソート
    data.sort_values('time',inplace=True)

    print(data['status'].value_counts())
    """print(data['status'].value_counts())"""

    delta = data['status'].to_numpy()

    #目的変数のベクトルb取得（17　パラメータ次元数）
    b = data['time'].to_numpy()
    
    data.drop(columns=['status','time'], inplace=True)
    
    #説明変数の行列A（nx17 データ数xパラメータ次元数
    A = data.to_numpy()    
    
    #標準化
    A, ave_A, std_A = normalization(A)

    lams = np.logspace(-5, -1, 10)
    x_lams = []

    for i, lam in enumerate(lams):
        #反復縮小アルゴリズム
        niter = 51
        lam = 0.00083
        iterative_shrinkage = IterativeShrinkage(A, b, delta, lam)
        x, _ = iterative_shrinkage.LR_CD(niter=niter)
        x_lams.append(x)

        # パラメータ次元数17に対するゼロの数
        #zero_counts = (x == 0.).sum()
        # 学習結果保存
        np.savez("./sparsemodeling-cox-regression/parms.npz", x, ave_A, std_A)
    
    x_lams = np.array(x_lams)
    
    #描画（経路）
    for pix in x_lams.T:
        plt.semilogx(lams, pix)
    plt.xlim(lams[0], lams[-1])
    plt.ylim(-1.5, 1.5)
    """ndx = np.where(ne2.x != 0)[0]
    for col in ndx:
        plt.axhline(ne2.x[col], ls='--', c='k')"""
    plt.ylabel('Regression coefficients')
    plt.xlabel('$\lambda$')
    plt.savefig('path.png', dpi=220)
    

if __name__ == "__main__":
    main()