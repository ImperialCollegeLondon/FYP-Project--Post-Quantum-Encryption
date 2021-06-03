import reed_muller_coding
import math


def nCr(n,r):
    f = math.factorial
    return int(f(n) / f(r) / f(n-r))

for n in range(3,100):
    ncr_sum = 0
    for r in range(0,n-1):
        ncr_sum += nCr(n,r)


    print("r value: ", n-2, "n value: ",n, "input length: ", ncr_sum, "output length is ", 2**n)