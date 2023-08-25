####################################################################################################################

import random
import numpy as np 
from functools import reduce

large_width = 400
np.set_printoptions(linewidth=large_width)


def concat_two_codes(codes2,codes1,bit_len_2,bit_len_1):
    shifted_codes2 = [c2 << bit_len_1 for c2 in codes2]
    codes2_con_codes1 = [sh_c2 + c1 for sh_c2 in shifted_codes2 for c1 in codes1]    
    return codes2_con_codes1

D = [7, 11, 19, 35, 67, 131, 13, 21, 37, 69, 133, 25, 41, 73, 137, 49, 81, 145, 97, 161, 193, 14, 22, 38,
              70, 134, 26, 42, 74, 138, 50, 82, 146, 98, 162, 194, 28, 44, 76, 140, 52, 84, 148, 100, 164, 196, 56,
              88, 152, 104, 168, 200, 112, 176, 208, 224]
D = concat_two_codes(D,D,8,8)
H = [7, 11, 13, 14]
D = concat_two_codes(H,D,4,16)

A = np.array(
    [#1 2 3 4 5 6 7 8 9 10111213141516171819
     [1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1], # 1   d
     [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], # 2  d
     [0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0], # 3   d
     [0,0,1,0,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0], # 4  d
     [1,0,1,0,0,0,0,0,0,0,1,0,0,1,0,1,0,1,0], # 5 d
     [0,0,0,0,0,0,0,1,0,0,0,0,1,0,1,0,1,0,0], # 6   d
     [0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0], # 7     d
     [1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,0,0,0], # 8 d
     [0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,1,1,0,0], # 9   d
     [0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0], # 10  d
     [1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0], # 11    d
     [0,0,1,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0], # 12  d
     [1,0,0,0,0,0,0,0,1,1,1,0,0,0,1,0,0,0,0], # 13    d
     [0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0], # 14    d
     [1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,1,1], # 15 d
     ],dtype = int)


def compute_R_sta(A: np.ndarray, D: list, bit_le_D: int) -> list:
    once_bi = 2**bit_le_D - 1
    m, n = A.shape
    R_sta = []
    for i in range(m):
        for d in D:
            R = R_sta + [d] + [once_bi] * (m - 1 - len(R_sta))
            C = [reduce(lambda x,y: x & y, [R[i] for i in range(m) if A[i, j] == 1], 2**bit_le_D - 1) for j in range(n)]
            if all(0 < C[j] or sum(A[:,j]) == m for j in range(n)):
                R_sta.append(d)
                break
        else:
            raise ValueError("No solution found")
    return R_sta

print(compute_R_sta(A[range(4)], D, 20))
####################################################################################################################
