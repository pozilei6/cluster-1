

from functools import reduce
import numpy as np
import scipy.special
import copy
binom = scipy.special.binom

def rem_A(A, Ir, Jc): #for displaying A-state only
  m, n = np.shape(A)
  del_I = [i for i in range(m) if i not in Ir]
  del_J = [j for j in range(n) if j not in Jc]
  A_copy = copy.deepcopy(A)
  A_copy_del = np.delete(A_copy, del_I, axis=0)
  A_copy_del = np.delete(A_copy_del, del_J, axis=1)
  return A_copy_del
  
def make_A_rand(m, n, p_True):
    A = np.zeros((2*m, 2*n), dtype = int)
    for i in range(2*m):
        for j in range(2*n):
            A[i,j] = p_True >= np.random.rand()
    A = np.unique(A, axis=0)   
    A = np.unique(A, axis=1) 
    for i in range(len(A)):
      if all(A[i]==0):
        A = np.delete(A, i, 0)
        return A
    return A

#copied--------------------------
def place_m_B(m_B, inc, lst_multipl):
    max1s_mu = [multpl for multpl in lst_multipl if multpl[1][0] <= inc] # or m_B < 5]
    if m_B <= 5:
        return max1s_mu[0]
    possible_tupls = []                                             
    for k in range(len(max1s_mu) - 1):
        if max1s_mu[k][0] <= m_B and m_B <= max1s_mu[k+1][0]:
            possible_tupls += [max1s_mu[k+1]]
            if 1 < inc and k < len(max1s_mu) - 2:
                possible_tupls += [max1s_mu[k+2]]            
    ind_max_l1 = np.argmax([potu[1][0] for potu in possible_tupls])
    return possible_tupls[ind_max_l1]  
    
def place_between_multipl(M_B, Inc):
    lst_multipl = [( int(binom(8 - (l_0 + l_1), 3 - l_1)), (l_1, l_0) ) for l_1 in range(3) for l_0 in range(4) if l_1 + l_0 < 5]
    lst_multipl.sort()
    Tup_ins = ()
    for m_B, inc in zip(M_B, Inc):
        Tup_ins += (place_m_B(m_B, inc, lst_multipl)[1],)
    return Tup_ins
    #----------------------------------------------











def tup_op(j1, j2):                                                             #
    return (j1, j2)
   
def inters_op(set1, set2):                                                      #
    return set1.intersection(set2)

def all_pairs1(lst, operation = tup_op):                                        #
    length = len(lst)
    if length < 2:
        return [] 
    return [operation(lst[k1], lst[k2]) for k1 in range(length) for k2 in range(k1 + 1, length)]

def all_pairs2Lists(lst1, lst2, operation = tup_op):                            #
    length1 = len(lst1)
    length2 = len(lst2) 
    return [operation(lst1[k1], lst2[k2]) for k1 in range(length1) for k2 in range(length2)]
    
def all_pairs2(lst1, lst2, operation = inters_op):                              #
    length1 = len(lst1)
    length2 = len(lst2)
    i_tups = all_pairs2Lists(range(length1), range(length2)) 
    return [operation(lst1[k1], lst2[k2]) for (k1, k2) in i_tups]

def next_Dec(Dec, Rc1, Rc2, Rc12):  #
    Rc1c2c12 = Rc12.union(Rc1, Rc2)
    unionDec = reduce(lambda s1,s2:s1.union(s2), Dec, set())
    unionRc  = Rc1c2c12 #reduce(lambda s1,s2:s1.union(s2), Rc1c2c12, set()) reduce(lambda s1,s2:s1.union(s2), Rc1c2c12)
    Inter = all_pairs2(Dec, (Rc1, Rc2, Rc12)) #all_pairs2(Dec, Rc1) + all_pairs2(Dec, Rc2) + all_pairs2(Dec, Rc12)
    diffs_Rcs_unDec = [Rc1.difference(unionDec)] + [Rc2.difference(unionDec)] + [Rc12.difference(unionDec)]
    diffs_Decs_unRc = [dec.difference(unionRc) for dec in Dec ]
    upDec = Inter + diffs_Rcs_unDec + diffs_Decs_unRc #make measure also
    return [upD for upD in upDec if upD]
    
def prior_R(Inc, l, Ir):            #
    sizes_lnc = [len(Inc[i_r]) for i_r in Ir]
    L1 = [l[i_r][0] for i_r in Ir]
    pr_i_R = sorted(range(len(Ir)), key=lambda i: (sizes_lnc[i], -L1[i]))
    pr_R = [Ir[i_s] for i_s in pr_i_R]
    return pr_R

def inz(j, Ir, Inc):
    inz_j = [r for r in Ir if j in Inc[r]]
    return inz_j

def upd_Jc(Jc, Ir, l, Inc):
    updJc = [j for j in Jc if all([l[r][0] > 0 for r in inz(j, Ir, Inc)])]
    return updJc

def get_valid_js(r, Ir, Inc, l):
    [j for j in Inc[r] ]
    
def get_Rc_triple(Inc, t, Ir):      #
    j1, j2 = t
    Rc1  = set(filter(lambda i:Inc[i].intersection(t) == {j1},     Ir))
    Rc2  = set(filter(lambda i:Inc[i].intersection(t) == {j2},     Ir))
    Rc12 = set(filter(lambda i:Inc[i].intersection(t) == {j1, j2}, Ir))
    return (Rc1, Rc2, Rc12)

def sort_by_overlap(T, Dec, Ir, Inc, min_pairs = 1, nbrPrior = 2):
    Rc_triples = [get_Rc_triple(Inc, t, Ir) for t in T]
    J_min_pairs = [j for j in range(len(T)) if len(Rc_triples[j][2]) > min_pairs] #subract also from l1's by corresp Rc's!!!
    T = [T[j] for j in J_min_pairs]
    Rc_triples = [Rc_triples[j] for j in J_min_pairs]
    Decs = [next_Dec(Dec, Rc1, Rc2, Rc12) for (Rc1, Rc2, Rc12) in Rc_triples]
    length = min(nbrPrior, len(Rc_triples))
    sort_by_siz_Decs = sorted(range(len(T)), key=lambda j:len(Decs[j]))[0:length]
    sort_T          = [         T[j_s] for j_s in sort_by_siz_Decs]
    sort_Decs       = [      Decs[j_s] for j_s in sort_by_siz_Decs]
    sort_Rc_triples = [Rc_triples[j_s] for j_s in sort_by_siz_Decs]
    return sort_T, sort_Decs, sort_Rc_triples
    
def upd_l(l, Rc1, Rc2, Rc12, m):
    return [(max(l[i][0] - 1, 0), l[i][1]) if i in Rc12.union(Rc1, Rc2) else l[i] for i in range(m)]

def upd_Inc_by_Rcs(Inc, Rc1, Rc2, Rc12, t, m): #get somehow t in here!    
    j1, j2 = t
    Inc = [Inc[i].difference({j1}) if i in Rc1 else Inc[i].difference({j2}) if i in Rc2 else Inc[i].difference(t) if i in Rc12 else Inc[i] for i                                                                                                                         in range(m)] 
    return Inc

def upd_Inc_by_Jc(Inc, Jc):
    updInc = [inc.intersection(Jc) for inc in Inc]
    return updInc
    
def best(States):
    Decs = [State[0] for State in States]
    print(f"Decs={Decs}")
    Irs = [State[3] for State in States]
    lens_Irs = [len(Ir) for Ir in Irs]
    lens_Decs = [len(Dec) for Dec in Decs]
    weights = [lens_Decs[k] / max(lens_Irs[k], 1) for k in range(len(Decs))]
    #weights = [(m/max(l_Ir,1))*l_D for (l_Ir, l_D) in zip(lens_Irs, lens_Decs)]
    #De_min = np.argmin(lens_Decs)
    w_min_ind = np.argmin(weights) #TODO flag exceeding depth parameter
    return States[w_min_ind]



def States2(State):

    Dec, Inc, l, Ir, Jc, t, d = State
    
    if len(Ir) < 2:
        return zip([Dec],[Inc],[l],[Ir],[Jc],[t]), d + 1
    r = prior_R(Inc, l, Ir)[-1]
    
    T = all_pairs1(list(Inc[r]))
    
    sort_T, sort_Decs, sort_Rc_triples = sort_by_overlap(T, Dec, Ir, Inc, min_pairs = 1)
    
    if sort_T == []:
        return zip([Dec],[Inc],[l],[Ir],[Jc],[t]), d + 1
    
    ls = [upd_l(l, Rc1, Rc2, Rc12, m) for (Rc1, Rc2, Rc12) in sort_Rc_triples]
    
    Incs = [upd_Inc_by_Rcs(Inc, Rc1, Rc2, Rc12, t, m) for (t, (Rc1, Rc2, Rc12)) in zip(sort_T, sort_Rc_triples)]
    Jcs = [upd_Jc(Jc, Ir, l, Inc) for (l, Inc) in zip(ls, Incs)]
    Incs = [upd_Inc_by_Jc(iInc, Jc) for iInc in Incs]

    Irs = [[ir for ir in Ir if len(iInc[ir])!=0] for iInc in Incs]
    
    newStates = zip(sort_Decs, Incs, ls, Irs, Jcs, sort_T)    #newStates = zip(sort_Decs, Incs, ls, Jcs, Irs, sort_T)
    
    return newStates, d + 1
    
    
    


Ir = [1,  4,5, 7,8,  11,12,13]
Jc = [3,4,5,8,9]
Inc = [set(),{1,3,4},set(),set(),{2,3,4},{8,9,10},set(),{2,3,4,6,7},{1},set(),set(),{2,8},{1,3,4,5,6},{2,4,6,7,9},set()]
l = [(0,2),(2,1),(0,2),(0,2),(1,3),(2,1),(0,2),(1,2),(1,1),(0,2),(0,2),(1,1),(2,1),(2,2),(0,2)]
Dec = []
t = (3,41)
d = 0
m = len(Inc)
State = [Dec, Inc, l, Ir, Jc, t, d]


A = make_A_rand(12, 11, p_True = .41)


A = np.array([       
     [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], 
     [0,0,0,0,1,1,1,1,0,0,0,1,1,1,0,0,0,1,1,1,0,0,0],
     [0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,0,1,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0],
     [1,0,1,0,1,0,1,0,1,0,1,0,1,1,0,0,0,1,0,0,0,0,0],
     [0,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
     [0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0],
     [0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,1,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,1,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,1,1,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,1,1,1,0,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,1,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
     [1,0,1,0,0,0,1,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,1,1,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0],
     [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0]
     ], dtype = int)

A = np.array([       
     [1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [1,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [1,1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
     [0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
     [0,0,1,1,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],
     [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
     [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],
     [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],
     [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],
     [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],
     [0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],
     [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],
     [0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
     [1,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
     ], dtype = int)
#initializing given A
m, n = np.shape(A)
len_incs = [sum(a) for a in A]   

np.random.seed(22)
M_B = np.random.randint(low = 1, high = 22, size = m)

                                
l = place_between_multipl(M_B, len_incs) 

Inc = [set([j for j in range(n) if r[j]]) for r in A]   #incidences of all r \in R, e.g., inc_R[0]=[0, 3, 6]

Ir = range(m)
Jc = range(n)


Jc = upd_Jc(Jc, Ir, l, Inc)
Inc = upd_Inc_by_Jc(Inc, Jc)
Ir = [ir for ir in Ir if Inc[ir]]

print("-----------------------------------")

Dec = []
t = (3,4)
d = 0

State = [Dec, Inc, l, Ir, Jc, t, d]
#----------------------------------------------------


# execution-----------------------------------------
def iterate(State, depth = 11):
    States = [State]
    for k in range(depth):
        save_States = []
        for State in States:
          if State[3] == []:
            save_States += [State]
            continue
          newStates_wout_d, d = States2(State)
          new_States = [newState_wout_d + (d,) for newState_wout_d in newStates_wout_d]
          save_States = save_States + new_States
        States = save_States
        print(States, k,"\n\n")
    return States
print("-----------------------------------------------------------------")    
States = iterate(State, depth = 3)
for State in States:
    for st in State:
        print(st,"\n")
    print("---------------")
Decs = [St[0] for St in States]
for De in Decs:
    indizes = []
    for D in De:
        print(A[list(D)],"\n")
        indizes += list(D)
    print(f"\nnp.sort(indizes)={np.sort(indizes)}")
    print(f"\nremaining_rows={[i for i in range(m) if i not in indizes]}")
    print("\n\n")

def print_nextStates(State):
    newStates_wout_d, d = States2(State)
    new_States = [newState_wout_d + (d,) for newState_wout_d in newStates_wout_d]
    print(f"d={d}")
    print("---------------")
    for State in new_States:
        Dec, Inc, l, Ir, Jc, t, d = State
        print(f"Dec={Dec}, len(Dec)={len(Dec)}\n")
        print(f"Ir={Ir}\n")
        print(f"Jc={Jc}\n")
        with np.printoptions(threshold=np.inf):
            print(rem_A(A, Ir, Jc),"\n\n")
    print("\n\n")
    print("bestDec=",best(new_States)[0])
    
