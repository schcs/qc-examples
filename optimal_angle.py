import os, sys

SAGE_ROOT = "/usr/local/lib/sage"
SAGE_LOCAL = os.path.join(SAGE_ROOT, "local")
PYVER = f"{sys.version_info.major}.{sys.version_info.minor}"

# Candidate site-packages for Sage; add those that exist
candidates = [
    os.path.join(SAGE_LOCAL, f"lib/python{PYVER}/site-packages"),
    os.path.join(SAGE_LOCAL, f"var/lib/sage/venv-python{PYVER}/lib/python{PYVER}/site-packages"),
    os.path.join(SAGE_ROOT, "src"),
]
for p in candidates:
    if os.path.isdir(p):
        sys.path.insert(0, p)

os.environ["SAGE_ROOT"] = SAGE_ROOT
os.environ["SAGE_LOCAL"] = SAGE_LOCAL

from sage.all import *

MAXIMA_SHARE  = '@SAGE_MAXIMA_SHARE@'
MAXIMA_FAS = '/usr/local/lib/sage/local/lib/ecl/maxima.fas'
from sage.libs.ecl import EclObject, ecl_eval
ecl_eval("(require 'maxima \"{}\")".format(MAXIMA_FAS))

b = sage.calculus.var.var('b', domain=RR)
c = sage.calculus.var.var('c', domain=RR)

f4 = -0.5*cos(b)**8*cos(3*c) - 0.5*cos(b)**6*cos(4*c)*sin(b)**2 - 1.0*cos(b)**6*cos(2*c)*sin(b)**2 - 1.0*cos(b)**4*cos(4*c)*sin(b)**4 + 1.0*cos(b)**4*cos(3*c)*sin(b)**4 - 2.0*cos(b)**4*cos(2*c)*sin(b)**4 - 0.5*cos(b)**2*cos(4*c)*sin(b)**6 - 1.0*cos(b)**2*cos(2*c)*sin(b)**6 - 0.5*cos(3*c)*sin(b)**8 - 0.25*cos(b)**7*sin(b)*sin(4*c) - 0.25*cos(b)**5*sin(b)**3*sin(4*c) + 0.25*cos(b)**3*sin(b)**5*sin(4*c) + 0.25*cos(b)*sin(b)**7*sin(4*c) + 1.0*cos(b)**7*sin(b)*sin(3*c) + 1.0*cos(b)**5*sin(b)**3*sin(3*c) - 1.0*cos(b)**3*sin(b)**5*sin(3*c) - 1.0*cos(b)*sin(b)**7*sin(3*c) - 0.5*cos(b)**7*sin(b)*sin(2*c) - 0.5*cos(b)**5*sin(b)**3*sin(2*c) + 0.5*cos(b)**3*sin(b)**5*sin(2*c) + 0.5*cos(b)*sin(b)**7*sin(2*c) - 0.5*cos(b)**6*sin(b)**2 - 1.0*cos(b)**4*sin(b)**4 - 0.5*cos(b)**2*sin(b)**6

f5 = -0.125*cos(b)**10*cos(4*c) - 0.25*cos(b)**10*cos(3*c) - 0.125*cos(b)**10*cos(2*c) - 0.375*cos(b)**8*cos(4*c)*sin(b)**2 - 0.75*cos(b)**8*cos(3*c)*sin(b)**2 - 0.625*cos(b)**8*cos(2*c)*sin(b)**2 - 0.5*cos(b)**8*cos(c)*sin(b)**2 - 0.5*cos(b)**6*cos(4*c)*sin(b)**4 - 1.*cos(b)**6*cos(3*c)*sin(b)**4 - 1.25*cos(b)**6*cos(2*c)*sin(b)**4 - 1.5*cos(b)**6*cos(c)*sin(b)**4 - 0.5*cos(b)**4*cos(4*c)*sin(b)**6 - 1.*cos(b)**4*cos(3*c)*sin(b)**6 - 1.25*cos(b)**4*cos(2*c)*sin(b)**6 - 1.5*cos(b)**4*cos(c)*sin(b)**6 - 0.375*cos(b)**2*cos(4*c)*sin(b)**8 - 0.75*cos(b)**2*cos(3*c)*sin(b)**8 - 0.625*cos(b)**2*cos(2*c)*sin(b)**8 - 0.5*cos(b)**2*cos(c)*sin(b)**8 - 0.125*cos(4*c)*sin(b)**10 - 0.25*cos(3*c)*sin(b)**10 - 0.125*cos(2*c)*sin(b)**10 + 0.125*cos(b)**9*sin(b)*sin(4*c) + 0.25*cos(b)**7*sin(b)**3*sin(4*c) - 0.25*cos(b)**3*sin(b)**7*sin(4*c) - 0.125*cos(b)*sin(b)**9*sin(4*c) + 0.25*cos(b)**9*sin(b)*sin(3*c) + 0.5*cos(b)**7*sin(b)**3*sin(3*c) - 0.5*cos(b)**3*sin(b)**7*sin(3*c) - 0.25*cos(b)*sin(b)**9*sin(3*c) - 0.25*cos(b)**9*sin(b)*sin(c) - 0.5*cos(b)**7*sin(b)**3*sin(c) + 0.5*cos(b)**3*sin(b)**7*sin(c) + 0.25*cos(b)*sin(b)**9*sin(c) - 0.25*cos(b)**8*sin(b)**2 - 0.75*cos(b)**6*sin(b)**4 - 0.75*cos(b)**4*sin(b)**6 - 0.25*cos(b)**2*sin(b)**8


f6 = 1-0.03125*cos(b)**12*cos(5*c) - 0.125*cos(b)**12*cos(4*c) - 0.1875*cos(b)**12*cos(3*c) - 0.125*cos(b)**12*cos(2*c) - 0.03125*cos(b)**12*cos(c) - 0.0625*cos(b)**10*cos(5*c)*sin(b)**2 - 0.375*cos(b)**10*cos(4*c)*sin(b)**2 - 0.875*cos(b)**10*cos(3*c)*sin(b)**2 - 1.0*cos(b)**10*cos(2*c)*sin(b)**2 - 0.5625*cos(b)**10*cos(c)*sin(b)**2 + 0.03125*cos(b)**8*cos(5*c)*sin(b)**4 - 0.375*cos(b)**8*cos(4*c)*sin(b)**4 - 1.8125*cos(b)**8*cos(3*c)*sin(b)**4 - 2.875*cos(b)**8*cos(2*c)*sin(b)**4 - 1.96875*cos(b)**8*cos(c)*sin(b)**4 + 0.125*cos(b)**6*cos(5*c)*sin(b)**6 - 0.25*cos(b)**6*cos(4*c)*sin(b)**6 - 2.25*cos(b)**6*cos(3*c)*sin(b)**6 - 4.0*cos(b)**6*cos(2*c)*sin(b)**6 - 2.875*cos(b)**6*cos(c)*sin(b)**6 + 0.03125*cos(b)**4*cos(5*c)*sin(b)**8 - 0.375*cos(b)**4*cos(4*c)*sin(b)**8 - 1.8125*cos(b)**4*cos(3*c)*sin(b)**8 - 2.875*cos(b)**4*cos(2*c)*sin(b)**8 - 1.96875*cos(b)**4*cos(c)*sin(b)**8 - 0.0625*cos(b)**2*cos(5*c)*sin(b)**10 - 0.375*cos(b)**2*cos(4*c)*sin(b)**10 - 0.875*cos(b)**2*cos(3*c)*sin(b)**10 - 1.0*cos(b)**2*cos(2*c)*sin(b)**10 - 0.5625*cos(b)**2*cos(c)*sin(b)**10 - 0.03125*cos(5*c)*sin(b)**12 - 0.125*cos(4*c)*sin(b)**12 - 0.1875*cos(3*c)*sin(b)**12 - 0.125*cos(2*c)*sin(b)**12 - 0.03125*cos(c)*sin(b)**12 + 0.0625*cos(b)**11*sin(b)*sin(5*c) + 0.1875*cos(b)**9*sin(b)**3*sin(5*c) + 0.125*cos(b)**7*sin(b)**5*sin(5*c) - 0.125*cos(b)**5*sin(b)**7*sin(5*c) - 0.1875*cos(b)**3*sin(b)**9*sin(5*c) - 0.0625*cos(b)*sin(b)**11*sin(5*c) + 0.1875*cos(b)**11*sin(b)*sin(4*c) + 0.5625*cos(b)**9*sin(b)**3*sin(4*c) + 0.375*cos(b)**7*sin(b)**5*sin(4*c) - 0.375*cos(b)**5*sin(b)**7*sin(4*c) - 0.5625*cos(b)**3*sin(b)**9*sin(4*c) - 0.1875*cos(b)*sin(b)**11*sin(4*c) + 0.125*cos(b)**11*sin(b)*sin(3*c) + 0.375*cos(b)**9*sin(b)**3*sin(3*c) + 0.25*cos(b)**7*sin(b)**5*sin(3*c) - 0.25*cos(b)**5*sin(b)**7*sin(3*c) - 0.375*cos(b)**3*sin(b)**9*sin(3*c) - 0.125*cos(b)*sin(b)**11*sin(3*c) - 0.125*cos(b)**11*sin(b)*sin(2*c) - 0.375*cos(b)**9*sin(b)**3*sin(2*c) - 0.25*cos(b)**7*sin(b)**5*sin(2*c) + 0.25*cos(b)**5*sin(b)**7*sin(2*c) + 0.375*cos(b)**3*sin(b)**9*sin(2*c) + 0.125*cos(b)*sin(b)**11*sin(2*c) - 0.1875*cos(b)**11*sin(b)*sin(c) - 0.5625*cos(b)**9*sin(b)**3*sin(c) - 0.375*cos(b)**7*sin(b)**5*sin(c) + 0.375*cos(b)**5*sin(b)**7*sin(c) + 0.5625*cos(b)**3*sin(b)**9*sin(c) + 0.1875*cos(b)*sin(b)**11*sin(c) - 0.125*cos(b)**10*sin(b)**2 - 0.5*cos(b)**8*sin(b)**4 - 0.75*cos(b)**6*sin(b)**6 - 0.5*cos(b)**4*sin(b)**8 - 0.125*cos(b)**2*sin(b)**10





def matrix_C(nr_vertices, edge_list):
    """Return the cost matrix C (diagonal) counting cut edges for each bitstring.
    Base ring is SR to be compatible with symbolic unitaries.
    """
    sequences = [list(map(int, format(i, f'0{nr_vertices}b'))) for i in range(2**nr_vertices)]
    C = zero_matrix(SR, 2**nr_vertices, 2**nr_vertices)
    for i, s in enumerate(sequences):
        C[i, i] = -len([e for e in edge_list if s[e[0]] != s[e[1]]])
    return C


def matrix_edge(nr_vertices, edge):
    """Return the diagonal matrix for a single edge: 1 if the edge is cut by the bitstring, else 0.
    Base ring is SR to match symbolic operations.
    """
    sequences = [list(map(int, format(i, f'0{nr_vertices}b'))) for i in range(2**nr_vertices)]
    C_edge = zero_matrix(SR, 2**nr_vertices, 2**nr_vertices)
    a, b = edge
    for i, s in enumerate(sequences):
        C_edge[i, i] = -1 if s[a] != s[b] else 0
    return C_edge


def UB_tensor(nr_bits, b):
    """
    Construct UB = exp(-i * b * B) for B = sum_i X_i
    via tensor product: UB = ⊗_{k=1}^n exp(-i b X).
    This is symbolic in b and efficient.
    Base ring is SR (Symbolic Ring) to interoperate with other symbolic matrices.
    """
    X = matrix(SR, [[0, 1],
                    [1, 0]])
    I2 = identity_matrix(SR, 2)
    # exp(-i b X) = cos(b) * I2 - i * sin(b) * X
    U1 = cos(b) * I2 - I * sin(b) * X
    U = U1
    for _ in range(nr_bits - 1):
        U = U.tensor_product(U1)
    return U

def expC(C, c):
    Cdiag = C.diagonal()
    UCdiag = [exp(-I*c*Cdiag[i]) for i in range(C.nrows())]
    return diagonal_matrix(UCdiag)

def function_graph(nr_vertices, edge_list, edge):
    b = var('b', domain=RR)
    c = var('c', domain=RR)

    C = matrix_C(nr_vertices, edge_list)
    C_edge = matrix_edge(nr_vertices, edge)
    UB = UB_tensor(nr_vertices, b)
    UB_dag = UB.conjugate().transpose()
    
    UC = expC(C, c)
    
    BCB = UC.conjugate().transpose()*UB_dag*C_edge*UB
    s = (2.0**nr_vertices)**(-1/2)*vector([1.0]*2**nr_vertices)
    assert abs(s.norm() - 1) < 10.0**-5
    f = s*BCB*s
    return f.real().simplify()
    

def maximize_f(f, b_min, b_max, c_min, c_max, b_steps=360, c_steps=180, refine_factor=10):
    """
    Numerically maximize f(b, c) over [b_min, b_max] x [c_min, c_max].
    Returns (f_max, b_argmax, c_argmax).
    """
    f_fast = fast_callable(f, vars=[b, c], domain=RDF)
    def eval_f(bb, cc):
        return f_fast(float(bb), float(cc))

    db = (b_max - b_min)/b_steps
    dc = (c_max - c_min)/c_steps
    max_val = -infinity
    b_star = None
    c_star = None
    bb_vals = srange(b_min, b_max, db, include_endpoint=True)
    cc_vals = srange(c_min, c_max, dc, include_endpoint=True)
    for bb in bb_vals:
        for cc in cc_vals:
            val = eval_f(bb, cc)
            if val > max_val:
                max_val = val
                b_star, c_star = bb, cc
    # local refinement around the coarse maximizer
    db_fine = db/refine_factor
    dc_fine = dc/refine_factor
    b_lo = max(b_min, b_star - 2*db)
    b_hi = min(b_max, b_star + 2*db)
    c_lo = max(c_min, c_star - 2*dc)
    c_hi = min(c_max, c_star + 2*dc)
    bb_vals_fine = srange(b_lo, b_hi, db_fine, include_endpoint=True)
    cc_vals_fine = srange(c_lo, c_hi, dc_fine, include_endpoint=True)
    for bb in bb_vals_fine:
        for cc in cc_vals_fine:
            val = eval_f(bb, cc)
            if val > max_val:
                max_val = val
                b_star, c_star = bb, cc
    return {'max_val':RR(max_val), 'b':RR(b_star), 'c':RR(c_star)}

# Example usage:
# f_max, b_argmax, c_argmax = maximize_f(f4, 0, 2*pi, 0, pi)
# print("max f(b,c) ≈", f_max)
# print("at b ≈", b_argmax, "c ≈", c_argmax)


# given a cubic graph, we compute the number of 
# - isolated triangles
# - crossed squares
# - the third type of neighbourhood

def number_of_neighbors(graph):

    edge_list = []
    for edge in graph.edges():
        neigh_j = list(graph.neighbors(edge[0]))
        neigh_k = list(graph.neighbors(edge[1]))
        neigh_jk = set(neigh_j+neigh_k)
        edge_list.append(len(neigh_jk))

    counts = {4:0, 5:0, 6:0}
    for val in edge_list:
        if not val in [4,5,6]:
            raise('value not in range')
            
        counts[val] += 1
    return counts

def optimal_angles_graph(graph):

    nr_edges_of_type = number_of_neighbors(graph)
    func = nr_edges_of_type[4]*f4+nr_edges_of_type[5]*f5+nr_edges_of_type[6]*f6 
    return maximize_f(-func, 0, 2*pi, 0, pi)

if __name__ == '__main__':
    g = graphs.RandomRegular(3,20)
    print(optimal_angles_graph(g))