


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
