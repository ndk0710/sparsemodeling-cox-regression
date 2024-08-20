import glob
import os
#from PIL import Image
import numpy as np
import pandas as pd
from lib.algorithm import IterativeShrinkage

def main():
    #パラメータ
    patch_size = 8
    patch_num = 2     
    file_dir = "./Dataset/test_tone"        #評価データのフォルダ

    #学習済みパラメータ読込み
    parms = np.load(file="./parms.npz")
    x0 = parms["arr_0"]
    x = parms["arr_1"]
    ave_A = parms["arr_2"]
    l2_norm_A = parms["arr_3"]

    #説明変数の行列A（nx1024 データ数xパラメータ次元数）
    A = []
    img_files_pass = glob.glob(f"{file_dir}/pass/*.jpg")
    img_files_fail = glob.glob(f"{file_dir}/fail/*.jpg")
    img_files = img_files_pass + img_files_fail

    #画像データ分処理
    for index, fname in enumerate(img_files):

        if os.path.isfile(fname):
            imgPIL = Image.open(fname)
            A.append((np.array(imgPIL.crop((patch_size * patch_num - 1, patch_size * patch_num - 1, imgPIL.width - patch_size * patch_num - 1, imgPIL.height - patch_size * patch_num - 1)))).flatten())

    A = np.array(A)

    #標準化（列方向の平均0、2乗和平均1 学習済みパラメータを使用）
    A = (A - ave_A)/l2_norm_A

    #良否判定
    y = np.exp(x0 + np.dot(A, x)) / (1 + np.exp(x0 + np.dot(A, x)))

    #csv出力
    df = pd.DataFrame(y, index=img_files)
    df.to_csv('./score_logistic.csv', header=True)


if __name__ == "__main__":
    main()