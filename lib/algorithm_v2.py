import numpy as np

class IterativeShrinkage(object):
    """ 反復縮小アルゴリズム """
    def __init__(self, A, b, delta, lam):
        self.A = A                              # nxmの観測行列A（n：学習に使用するデータ数、m：パラメータの次元数（1024 = 32*32））
        self.b = b                              # n次元の正解ベクトル（生存時間）
        self.delta = delta                      # n次元のステータスベクトル（0：打ち切り、1：移植、2：死亡）
        self.eps = 1e-10                        # 更新誤差の閾値
        self.lam = lam                          # 正則化項（ラッソL1ノルム）のパラメータ
        self.x = np.zeros(A.shape[1])           # コックス回帰モデルのパラメータx（m次元のベクトル）の初期値
        self.z = np.zeros(A.shape[0])           # 正則化対数尤度関数に含まれるパラメータz(n次元のベクトル)の初期化
        self.log = []
        self.omega = np.zeros(A.shape[0])
    # 変換1
    def tranformation1(self):
        """
        lamdaの値が小さいとき全変数を使って過学習するため、パラメータの値が大きくなる？
        その結果、np.expの中身が大きくなりオーバーフローする。この場合、学習終了条件③に合致するため、更新は続かないはず。。
        根本解決はできていない。ソフトマックス関数のように最大値で正規化などできないか？
        """
        """check = np.dot(self.A, self.x)
        ooooo=np.where(check > 300, 300, check)
        return np.exp(ooooo)"""
        return np.exp(np.dot(self.A, self.x))
    
        #return np.exp(self.x0 + np.dot(self.A, self.x)) / (1 + np.exp(self.x0 + np.dot(self.A, self.x)))
    
    # 変換2
    def tranformation2(self):
        
        matrix = IterativeShrinkage.tranformation1(self)
        
        Ci = []
        for index in range(len(self.b)):
            if self.delta[index] == 1:
                tmp_Ci=self.b[0:index+1]
            Ci.append(tmp_Ci)
        
        for ci_index, ci_element in enumerate(Ci):
            total=0
            total2=0
            count=0
            for ci in ci_element:
                ndx = np.where(self.b >= ci)[0]
                Rj_sum = np.sum(matrix[ndx])
                #value = (Rj_sum * matrix[ci_index] - matrix[ci_index] * matrix[ci_index])/(Rj_sum ** 2)
                tmp=matrix[ci_index]/Rj_sum
                value = matrix[ci_index]/Rj_sum - tmp**2
                value2 = matrix[ci_index]/Rj_sum
                total += value
                total2 += value2
                count += 1
            self.omega[ci_index] = total/count
             
        return self.omega
    
    # 変換3
    def tranformation3(self):
        
        term = np.zeros(self.A.shape[0])
        matrix = IterativeShrinkage.tranformation1(self)

        Ci = []
        for index in range(len(self.b)):
            if self.delta[index] == 1:
                tmp_Ci=self.b[0:index+1]
            Ci.append(tmp_Ci)
        
        for ci_index, ci_element in enumerate(Ci):
            total=0
            count=0
            for ci in ci_element:
                ndx = np.where(self.b >= ci)[0]
                Rj_sum = np.sum(matrix[ndx])
                value = matrix[ci_index]/Rj_sum
                total += value
                count += 1
            term[ci_index] = self.delta[ci_index] - (total/count)

        return term

    # 軟閾値作用素
    def soft_threshold(self, x):
        threshold = self.lam
        return np.where(np.abs(x) > threshold, (np.abs(x) - threshold) * np.sign(x), 0)

    def soft_threshold_new(self, x, num):
        threshold = self.lam
        return np.where(np.abs(x) > threshold, (np.abs(x*num) - num*threshold) * np.sign(x*num), 0)
        #return np.sign(x*num)*np.maximum(np.abs(x*num)-num*threshold, 0)

    def LR_CD(self, niter=50):
        """ 
        コックス回帰による座標降下法(logistic regression coordinate descent method; LR_CD) 

        self.old_x0：更新前のコックス回帰モデルのパラメータx0（定数項）
        self.old_x：更新前のコックス回帰モデルのパラメータx（m次元のベクトル）
        self.pi：正則化対数尤度関数に含まれるパラメータπ（n次元のベクトル）
        　　　→事後確率（正常の確率0、異常の確率1に近づくように学習が進むはず）
        self.omega：正則化対数尤度関数に含まれるパラメータω（n次元のベクトル）
        ndx：self.omega = 0 の観測データを除外するための情報（self.omega != 0のインデックス情報）
        　→除外しないとself.z更新時に第3項が0で割る形となる（除外して更新して良い？）
        　　self.omega = 0となる条件は、事後確率がself.pi = 0（正常） or 1（異常）と断定された場合
        """

        for k in range(niter):
            #（ω）を更新
            self.pi = IterativeShrinkage.tranformation1(self)
            self.omega = IterativeShrinkage.tranformation2(self)
            self.log.append(self.pi)
            #self.omega = self.pi * (1 - self.pi)

            #（z）を更新
            term = IterativeShrinkage.tranformation3(self)
            ndx = np.where(self.omega != 0.)[0]
            oo = np.dot(self.A, self.x)[ndx]
            self.z[ndx] = np.dot(self.A, self.x)[ndx] + term[ndx] / self.omega[ndx]
            #self.z[ndx] = self.x0 + np.dot(self.A[ndx], self.x) + (self.b[ndx] - self.pi[ndx]) / self.omega[ndx]

            # 固定した（x）で（x）を推定
            omega_a = self.A[ndx] * self.omega[ndx].reshape(-1, 1)
            const = (self.z[ndx] - np.dot(self.A[ndx], self.x)).reshape(1, -1)
            a_x = self.A[ndx].T * self.x.reshape(-1, 1)
            e = np.sum(omega_a.T * (const + a_x), axis=1)/self.A[ndx].shape[0]
            e_s = IterativeShrinkage.soft_threshold(self, e)
            #e_s = IterativeShrinkage.soft_threshold_new(self, e, self.A[ndx].shape[0])

            # 学習終了条件②（汎化性能上げ過ぎ？パラメータxの全要素が0）
            if np.all(e_s == 0.):
                return self.x, self.log
                
            
            B = self.A[ndx] * self.A[ndx]
            omega_x2 = np.sum(B.T * self.omega[ndx].reshape(1, -1), axis=1)
            tmp_x = e_s/omega_x2

            # 更新前パラメータ保存（self.old_x）
            self.old_x = self.x

            # パラメータ更新（self.x）
            self.x = tmp_x

            # 1エポック目は飛ばす
            if k != 0:
                # 学習終了条件①（パラメータx0,xの更新誤差の確認）
                if np.dot(self.x - self.old_x, self.x - self.old_x) < self.eps:
                    return self.x, self.log

        return self.x, self.log

        """# 学習終了条件③（過学習？学習データを完全に間違う。self.b = 0（正常）なのにself.pi = 1（異常）。その逆も同様）
            if (np.abs(self.b - self.pi) == 1.).sum() != 0:
                return self.old_x0, self.old_x, self.log
            else:
                #（z）を更新
                ndx = np.where(self.omega != 0.)[0]
                self.z[ndx] = self.x0 + np.dot(self.A[ndx], self.x) + (self.b[ndx] - self.pi[ndx]) / self.omega[ndx]

                # 固定した（x0,x）で（x0）を推定
                tmp_x0 = np.array([(np.dot(self.omega[ndx], self.z[ndx] - np.dot(self.A[ndx], self.x)))/np.sum(self.omega[ndx])])

                # 固定した（x0,x）で（x）を推定
                omega_a = self.A[ndx] * self.omega[ndx].reshape(-1, 1)
                const = (self.z[ndx] - self.x0 - np.dot(self.A[ndx], self.x)).reshape(1, -1)
                a_x = self.A[ndx].T * self.x.reshape(-1, 1)
                e = np.sum(omega_a.T * (const + a_x), axis=1)/self.A[ndx].shape[0]
                #e_s = IterativeShrinkage.soft_threshold(self, e)
                e_s = IterativeShrinkage.soft_threshold_new(self, e, self.A[ndx].shape[0])

                # 学習終了条件②（汎化性能上げ過ぎ？パラメータxの全要素が0）
                if np.all(e_s == 0.):
                    return self.x0, self.x, self.log
                    
                B = self.A[ndx] * self.A[ndx]
                omega_x2 = np.sum(B.T * self.omega[ndx].reshape(1, -1), axis=1)
                tmp_x = e_s/omega_x2

                # 更新前パラメータ保存（self.old_x0, self.old_x）
                self.old_x0 = self.x0
                self.old_x = self.x

                # パラメータ更新（self.xo, self.x）
                self.x0 = tmp_x0
                self.x = tmp_x

                # 1エポック目は飛ばす
                if k != 0:
                    # 学習終了条件①（パラメータx0,xの更新誤差の確認）
                    if np.dot(self.x - self.old_x, self.x - self.old_x) + np.dot(self.x0 - self.old_x0, self.x0 - self.old_x0) < self.eps:
                        return self.x0, self.x, self.log
            
        return self.x0, self.x, self.log"""