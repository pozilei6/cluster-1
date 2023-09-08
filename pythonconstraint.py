# pip install python-constraint                     https://pypi.org/project/python-constraint/

def CSP_pyConstraints(A, D, bitlength=17):
    defa = 2 ** bitlength - 1
    m, n = A.shape

    problem = Problem()

    R = [f"rs_{i + 1}" for i in range(m)]
    C = [f"cs_{j + 1}" for j in range(n)]

    problem.addVariables(R, range(defa + 1))
    problem.addVariables(C, range(defa + 1))

    problem.addConstraint(InSetConstraint(D), R)

    for j in range(n):
        def constraint_func(*args):
            return reduce(lambda x,y: x & y, args) == defa
        problem.addConstraint(FunctionConstraint(constraint_func), [R[i] for i in range(m) if A[i, j] == 1])

    for i in range(m):
        for j in range(n):
            if A[i, j] == 0:
                def constraint_func(r, c):
                    return r & c != c
                problem.addConstraint(FunctionConstraint(constraint_func), [R[i], C[j]])

    solution = problem.getSolution()

    if solution:
        R_sol = [solution[r] for r in R]
        C_sol = [solution[c] for c in C]
        return R_sol, C_sol
    else:
        return [], []





# to be compared with 
def CSP(A, D, bitlength=17):  
    defa = 2 ** bitlength - 1
    m, n = A.shape
    s = Solver()
    s.set("model.completion", True)
    s.set("timeout", 8888)

    R = [BitVec(f"rs_{i + 1}", bitlength) for i in range(m)]
    C = [BitVec(f"cs_{j + 1}", bitlength) for j in range(n)]

    s.add([Or([r == d for d in D]) for r in R])

    for j in range(n):
        s.add(C[j] == reduce(lambda x,y: x & y, [R[i] for i in range(m) if A[i, j] == 1], defa))

    s.add([R[i] & C[j] != C[j] for i in range(m) for j in range(n) if A[i, j] == 0])    

    if s.check() == sat:
        model = s.model()
        R_sol = [model.evaluate(R[i]).as_long() for i in range(m)]
        C_sol = [model.evaluate(C[j]).as_long() for j in range(n)]
        return R_sol, C_sol
    else:
        return [], []



# make some instances (A, D, bitlength) and store them in list Instances ... then run  compare_runtimes(Instances)



import time
import matplotlib.pyplot as plt

def compare_runtimes(Instances):
    CSP_times = []
    CSP_pyConstraints_times = []
    sizes = []

    for A, D, bitlength in Instances:
        m, n = A.shape
        sizes.append(m * n)

        start = time.time()
        CSP(A, D, bitlength)
        end = time.time()
        CSP_times.append(end - start)

        start = time.time()
        CSP_pyConstraints(A, D, bitlength)
        end = time.time()
        CSP_pyConstraints_times.append(end - start)

    plt.plot(sizes, CSP_times, label="CSP")
    plt.plot(sizes, CSP_pyConstraints_times, label="CSP_pyConstraints")
    plt.xlabel("m * n")
    plt.ylabel("Time (s)")
    plt.legend()
    plt.show()









