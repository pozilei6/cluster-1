

from z3 import Solver, BitVec, Or, reduce
import numpy as np

def CSP_ex(A, A_ex, D, R_pr, C_pr, bitlength=17):
    defa = 2 ** bitlength - 1
    m, n = A.shape
    m_ex, n_ex = A_ex.shape
    s = Solver()
    s.set("model.completion", True)
    s.set("timeout", 8888)

    R_ex = [BitVec(f"rs_{i + 1}", bitlength) for i in range(m_ex)]
    C_ex = [BitVec(f"cs_{j + 1}", bitlength) for j in range(n_ex)]

    s.add([Or([r == d for d in D]) for r in R_ex])

    for j in range(n_ex):
        s.add(C_ex[j] == reduce(lambda x,y: x & y, [R_ex[i] for i in range(m_ex) if A_ex[i, j] == 1], defa))

    s.add([R_ex[i] & C_ex[j] != C_ex[j] for i in range(m_ex) for j in range(n_ex) if A_ex[i, j] == 0])

    # Minimize the difference between R_pr and R_ex and between C_pr and C_ex
    s.minimize(sum([R_pr[i] != R_ex[i] for i in range(m)]) + sum([C_pr[j] != C_ex[j] for j in range(n)]))

    if s.check() == sat:
        model = s.model()
        R_ex_sol = [model.evaluate(R_ex[i]) for i in range(m)]
        C_ex_sol = [model.evaluate(C_ex[j]) for j in range(n)]
        viol_R = [i for i in range(m) if R_pr[i] != R_ex_sol[i]]
        viol_C = [j for j in range(n) if C_pr[j] != C_ex_sol[j]]
        print(f"Indices where R_pr and R_ex differ: {viol_R}")
        print(f"Indices where C_pr and C_ex differ: {viol_C}")
        return R_ex_sol, C_ex_sol
    else:
        return None


"""
CSP_ex() takes as input the binary 2D numpy arrays A and A_ext, the list of feasible values D, the lists of feasible solutions R_pr and C_pr, 
and an optional parameter bitlength (default value is 17). It returns a tuple of two lists representing an assignment to R_ext and C_ext that 
satisfies all the constraints given in the current CSP() function and minimizes the number of differences between R_pr and R_ext and between 
C_pr and C_ext. If no solution is found, the function returns None.
"""


"""
To test execute, we need (standard) CSP(), A, D and get solution R_in, C_in. Extend A somehow to A_ex and run CSP_ex(A, A_ex, D, R_in, C_in)
"""
