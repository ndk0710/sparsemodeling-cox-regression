import glob
import os
import pandas as pd
#from PIL import Image
import numpy as np
from lib.algorithm_v2 import IterativeShrinkage

#標準化（列方向の平均0、2乗和平均1）
def normalization(A):
    ave_A = np.mean(A, axis=0, keepdims=True)
    B = A - ave_A
    l2_norm_A = np.linalg.norm(B, axis=0, keepdims=True)
    Norm_A = B/l2_norm_A
    return Norm_A, ave_A, l2_norm_A

# スパース画像の表示
def show_sparse(x):
    # 非ゼロのインデックス取得 
    non_zero_ndx = np.where(x != 0.)[0]

    # ゼロ（不要な情報：黒）と非ゼロ（重要な情報：白）の画像
    zero_one = np.zeros(x.shape)
    zero_one[non_zero_ndx] = 255.
    zero_one_pilImg = Image.fromarray(np.uint8(zero_one.reshape(32,32)))
    zero_one_pilImg.save("./zero_one.png")

    # 絶対値に変換、Min-Maxの正規化画像（不要な情報：黒、重要な情報：白に近い）
    abs_x = np.abs(x)
    max = np.max(abs_x)
    min = np.min(abs_x)
    intensity = (abs_x - min)/(max - min) * 255.
    intensity_pilImg = Image.fromarray(np.uint8(intensity.reshape(32,32)))
    intensity_pilImg.save("./intensity.png")

def main():
    #パラメータ
    """patch_size = 8
    patch_num = 2
    file_dir = "./Dataset/pickup1_tone"         #学習データのフォルダ"""

    #データ読み込み
    data = pd.read_csv("./sparsemodeling-cox-regression/csv/pbc_data.csv", encoding='utf-8')

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
    
    #標準化
    A, ave_A, l2_norm_A = normalization(A)
    

    #目的変数のベクトルb取得（1024　パラメータ次元数）
    b_pass = np.zeros(len(img_files_pass))
    b_fail = np.ones(len(img_files_fail))
    b = np.hstack([np.array(b_pass), np.array(b_fail)])
    
    
    #反復縮小アルゴリズム
    niter = 51
    lam = 0.075
    iterative_shrinkage = IterativeShrinkage(A, b, lam)
    x0, x, log = iterative_shrinkage.LR_CD(niter=niter)

    # パラメータ次元数1024に対するゼロの数
    zero_counts = (x == 0.).sum()

    # スパース画像保存
    show_sparse(x)

    # 学習結果保存
    np.savez("parms", x0, x, ave_A, l2_norm_A)
    

    

if __name__ == "__main__":
    main()