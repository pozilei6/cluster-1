


#-----------------------------------------------
from z3 import *

def dag_set_labeling(M, M_trr, K=None):
    n = M_trr.shape[0]
    
    # Create a Z3 solver instance
    s = Solver()
    
    # Set the bit length for the bit vectors
    bitlength = 30 if K is None else K
    
    # Create a list of Z3 bit vector variables for each node
    X = [BitVec(f'x_{i}', bitlength) for i in range(n)]
    
    # Add constraints for condition 1
    for i in range(n):
        for j in range(n):
            if M_trr[i,j] == 1:
                s.add(X[j] & X[i] == X[i], X[i] != X[j])
    
    # Add constraints for condition 2
    for i in range(n):
        for j in range(i+1, n):
            if M[i,j] == 0 and M[j,i] == 0:
                s.add(And(X[i] & X[j] != X[i], X[j] & X[i] != X[j]))
    
    # Check if a solution exists
    check = s.check()
    print(check)
    
    if check == sat:
        # A solution was found
        model = s.model()
        
        # Convert the Z3 solution to a list of sets
        assignment = [set() for _ in range(n)]
        max_element = 0
        for i in range(n):
            value = model[X[i]].as_long()
            max_element = max(max_element, value.bit_length())
            assignment[i] = set([j for j in range(bitlength) if (value >> j) & 1])
        
        if K is None:
            # Optimize the solution to find the minimum size assignment
            opt = Optimize()
            opt.add(s.assertions())
            t = BitVec('t', bitlength)
            opt.add(t == reduce(lambda a, b: a | b, [X[i] for i in range(n)]))
            opt.minimize(BV2Int(t))
            check = opt.check()
            print(check)
            
            if check == sat:
                # An optimized solution was found
                model = opt.model()
                
                # Convert the Z3 solution to a list of sets
                assignment = [set() for _ in range(n)]
                max_element = 0
                for i in range(n):
                    value = model[X[i]].as_long()
                    max_element = max(max_element, value.bit_length())
                    assignment[i] = set([j for j in range(bitlength) if (value >> j) & 1])
        
        elif max_element > K:
            # The solution is not valid because it exceeds the maximum size K
            return None
        
        return assignment
    
    else:
        # No solution was found
        return None
#-----------------------------------------------









#----------------------------------------------- finishing
def CSP_ass(assignment, A, D, bitlength):
    m, n = A.shape
    defa = 2 ** bitlength - 1
    all_C_inds = set.union(*assignment)

    R = [BitVec(f"r_{i}", bitlength) for i in range(m)]
    C = [BitVec(f"c_{j}", bitlength) for j in range(n)]

    s = Solver()

    for i in range(m):
        s.add(Or([R[i] == d for d in D]))

    for j in range(n):
        s.add(C[j] == reduce(lambda x,y: x & y, [R[i] for i in range(m) if A[i, j] == 1], defa))

    for i in range(m):
        for j in range(n):
            if A[i, j] == 1:
                s.add(R[i] & C[j] == C[j])
            if A[i, j] == 0:
                s.add(R[i] & C[j] != C[j])

    # Add the missing constraint
    for k in all_C_inds:
        # Find the unique bit position for value k
        b_k = BitVec(f"b_{k}", bitlength)
        s.add(And([b_k == (C[j] & (C[j] - 1)) ^ C[j] for j in range(n) if k in assignment[j]]))
        
        # Ensure that the unique bit position is different for any other value k1
        for k1 in all_C_inds:
            if k != k1:
                b_k1 = BitVec(f"b_{k1}", bitlength)
                s.add(b_k != b_k1)



"""
In this code, the missing constraint is added to the solver using a nested loop over all values k and k1 in all_C_inds. For each value k, a new bit vector 
variable b_k is created to represent the unique bit position where all C[j] with k in assignment[j] have a set bit. This variable is constrained to be equal 
to the result of the expression (C[j] & (C[j] - 1)) ^ C[j], which extracts the rightmost set bit of C[j]. The constraint is added for all j such that k is 
in assignment[j]. Then, for any other value k1 in all_C_inds, a new bit vector variable b_k1 is created to represent the unique bit position for value k1, 
and a constraint is added to ensure that b_k != b_k1.
"""
#-----------------------------------------------


















