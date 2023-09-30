import gmpy2
import numpy as np

p = 11  # 大素数
q = 19  # 大素数
n = 209  # 乘积
lam = 90  # 最小公倍数
g = 147  # 随机整数
mu = 153

def L(x,n):
    return (x - 1) / n

# 明文m 以及 随机数r已经定义
m = 80
r = 3
n_square = pow(n, 2)  # n_square = 43681
c = gmpy2.mod(pow(g, m) * pow(r, n), n_square)  # c =  32948
print(c)
# 输出结果 32948 即加密结果

g_lam = pow(g,lam)
print('g_lam为：', g_lam)
g_lam_c = gmpy2.mod(g_lam,pow(n, 2))
print('g_lam模n方为：', g_lam_c)

mu_test = gmpy2.mod((L(g_lam_c,n)), n)
print('mu的值为:', mu_test)

m  = gmpy2.mod(L(gmpy2.mod(pow(c, lam), n_square), n) * mu, n) # m = 8
print(m)

cc = np.lcm((p-1), (q-1))
print('lcm:',cc)
print(type(cc))
cca = int(cc)
print(type(cca))
print((cca))

aa = pow(2,10*18*11*19)

print(gmpy2.mod(aa, 11*19*11*19))
print(440018*439618)

for i in range(90,10000):
    if (i % 90 ==0) and (i % 209 ==1):
        print(i)
        print('\n')






