import numpy as np

# read instances from M* dataset

def read_instances(filepath):
    file = open(filepath)
    s = file.readline().split(" ")
    m, n = eval(s[0]), eval(s[1])
    f = np.zeros((m,))
    c = np.zeros((n, m))
    for j in range(m):
        s = file.readline().split(" ")
        f[j] = eval(s[1])
    for i in range(n):
        s = file.readline()
        s = file.readline().split(" ")
        for j in range(m):
            c[i][j] = eval(s[j])
    return f, c


def getData(datapath):
    # j is facilities, total number is m
    # i is customers, total number is n
    # f[j] is opening cost
    # c[i][j] is service cost
    # 为了防止出错, 建议遍历 facilities 时都用 j , 遍历 customers 时都用 i.
    # 用 numpy 代替 for 循环 提高效率，想到什么随时在群里面说，这种行为保持统一

    f, c = read_instances(datapath)
    m = len(f)
    n = len(c)
    return m, n, f, c