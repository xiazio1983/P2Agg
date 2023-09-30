import gmpy2
import random
import numpy as np
import time


class T_paillier(object):
    def __init__(self):


        self.p_1 = 5
        self.q_1 = 9

        self.p = 2*self.p_1+1
        self.q = 2*self.q_1+1
        self.N = self.p * self.q #p*q
        self.lam_l = np.lcm((self.p-1), (self.q-1))
        self.lam = int(self.lam_l)

        self.mu = gmpy2.mod(pow(self.lam, -1), self.N)

        self.g = self.N+1

        self.sita = 6480

        self.sk2_p = random.randint(1, 6480)

        self.sk1_p = self.sita - self.sk2_p


    def L(self, x):
        return (x - 1) / self.N

    def encrypt_pai(self, m):
        r = random.randint(1, 30)
        print('生成随机数完成')
        N_2 = pow(self.N, 2)
        print('N的值为：',self.N)
        print('计算N_2完成:', N_2)
        r_N = pow(r, self.N)
        print('计算r_N完成')
        enc_m = gmpy2.mod((1+m*self.N) * r_N, N_2)
        print('加密完成')
        print('测试2：', gmpy2.mod(pow(r_N, self.lam), N_2))
        print('lam的值为：', self.lam)
        print('mu的值为：', self.mu)
        return enc_m

    def decrypt_pai(self, enc_m):


        denc_m = gmpy2.mod((self.L(gmpy2.mod(pow(enc_m, self.lam), pow(self.N, 2))))*self.mu, self.N)
        print('解密测试1:', gmpy2.mod(pow(enc_m, self.lam), pow(self.N, 2)))
        print('解密测试2', gmpy2.mod(1+self.lam * 2 *self.N, pow(self.N, 2)))
        print('测试:',self.L(gmpy2.mod(pow(enc_m, self.lam), pow(self.N, 2))))
        return denc_m

    def decrypt_JL(self, enc_m):

        int_H = random.randint(1, pow(self.N,2))
        sk_JL = -(self.sita)
        denc_m_JL = gmpy2.mod(self.L(pow(int_H, sk_JL)*enc_m), self.N)
        return denc_m_JL

    def decrypt_pai_s(self, enc_m):
        print('sita:', self.sita)
        print('sk1_p:', self.sk1_p)
        print('sk2_p:', self.sk2_p)
        denc_m_s1 = gmpy2.mod(pow(enc_m, self.sk1_p), pow(self.N, 2))
        denc_m_s2 = gmpy2.mod(pow(enc_m, self.sk2_p), pow(self.N, 2))
        return denc_m_s1, denc_m_s2

    def decrypt_pai_T(self, denc_m1, denc_m2):
        denc_m = self.L(gmpy2.mod(denc_m1*denc_m2, pow(self.N, 2)))
        return denc_m


test = T_paillier()
print('测试明文为：', 2)
#测试加密
time_start_enc = time.perf_counter()
text_enc_m = test.encrypt_pai(2)
time_cost_enc = time.perf_counter() - time_start_enc
print('加密结果为：', text_enc_m)
print('加密所耗费的时间为：', time_cost_enc)

#测试加密累乘
mul_enc = 1
time_start_enc_mul = time.perf_counter()
for i in range(150):
    mul_enc *= text_enc_m
time_cost_enc_mul = time.perf_counter() - time_start_enc_mul
print('加密结果累乘结果：', time_cost_enc_mul)#直接可以忽略不计5.4*10^（-6）

#测试JL解密
time_start_denc_JL = time.perf_counter()
text_denc_m_JL = test.decrypt_JL(text_enc_m)
print('JL解密所耗费的时间为：', (time.perf_counter()-time_start_denc_JL))


text_enc_m = pow(text_enc_m, 1)
#测试解密
text_denc_m = test.decrypt_pai(text_enc_m)
print('直接解密结果为：', text_denc_m)

#测试分段解密
time_start_denc = time.perf_counter()
text_denc_m1, text_denc_m2 = test.decrypt_pai_s(text_enc_m)
time_cost_denc = time.perf_counter() - time_start_denc
print('解密两次所耗费的时间为：', time_cost_denc)
print('分段解密结果m1为：\n')
print(text_denc_m1)
print('分段解密结果m2为：\n')
print(text_denc_m2)

#测试分段总体解密结果

text_denc_m_T = test.decrypt_pai_T(text_denc_m1, text_denc_m2)
print('分段总解密结果为：', text_denc_m_T)
































