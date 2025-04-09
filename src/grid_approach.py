import numpy as np
import random
import math
from numba import njit, prange,types
from numba.typed import List
import pynndescent

LOOKUP_TABLE = np.array([
    1, 1, 2, 6, 24, 120, 720, 5040, 40320,
    362880, 3628800, 39916800, 479001600,
    6227020800, 87178291200, 1307674368000,
    20922789888000, 355687428096000, 6402373705728000,
    121645100408832000, 2432902008176640000], dtype='int64')

@njit
def fast_factorial(n):
    if n > 20:
        raise ValueError
    return LOOKUP_TABLE[n]

# assign vetex v to the box, b is the sidelength of the box
def Assign(v, b):
    box = np.floor(v / b)
    transfer_vector = [np.array([1, -1]), np.array([1, 1]), np.array([-1, -1]), np.array([-1, 1])]
    chose_vector = [np.array([-1/3,-1/3]),np.array([-2/3,-2/3])]
    if (box[0] == v[0] / b) and (box[1] == v[1] / b):
        chosen_item=random.choice(chose_vector)
        return box * b+0.5*b*chosen_item
    center_box = box * b + 0.5 * b*np.array([1, 1])
    boxes = []
    dist = []
    for i in range(1, 3):
        for j in transfer_vector:
            new_box = center_box + (i / 3)* b*j
            boxes.append(new_box)
            dist.append(np.linalg.norm(v - new_box)**2)
    boxes.append(center_box)
    dist.append(np.linalg.norm(v - center_box)**2)
    indice = np.argmin(np.array(dist))
    return boxes[indice].copy()
#create index set for the multinomial coefficient

# pre-calculate the indexes and coeffs
@njit
def generate_partitions(w):
    """Generate all 6-tuples (a1,a2,a3,a4,a5,a6) where sum(a1..a6) = w."""
    partitions = []
    for a1 in range(w + 1):
        for a2 in range(w - a1 + 1):
            for a3 in range(w - a1 - a2 + 1):
                for a4 in range(w - a1 - a2 - a3 + 1):
                    for a5 in range(w - a1 - a2 - a3 - a4 + 1):
                        a6 = w - a1 - a2 - a3 - a4 - a5
                        partitions.append((a1, a2, a3, a4, a5, a6))
    return np.array(partitions, dtype=np.int64)
# Precompute partitions and derived values
def precompute_partitions(k):
    partition_list = List()
    for w in range(k + 1):
        p = generate_partitions(w)
        comp1 = 2 * p[:, 1] + p[:, 2]
        comp2 = 2 * p[:, 5] + p[:, 4]
        exp_factors = 4.0 ** (p[:, 2] + p[:, 4])
        coeffs = np.array([multi_coeff(w, *a_row) for a_row in p], dtype=np.float64)
        comp3 = 2 * p[:, 0] + p[:, 2]
        comp4 = 2 * p[:, 3] + p[:, 4]
        partition_list.append((p, comp1, comp2, exp_factors, coeffs, comp3, comp4))
    return partition_list

@njit
def generate_partitions_2(w):
    partitions = []
    for a1 in range(w + 1):
        for a2 in range(w - a1 + 1):
            for a3 in range(w - a1 - a2 + 1):
                a4 = w - a1 - a2 - a3
                partitions.append((a1, a2, a3, a4))
    return np.array(partitions, dtype=np.int64)
# Precompute partitions and derived values
TUPLE_TYPE = types.Tuple(
    (types.int64[:, :],  # p
     types.int64[:],     # a_1 
     types.int64[:],     # a_2
     types.int64[:],     # a_3
     types.float64[:])   # coeffs
)

@njit
def precompute_partitions_2(k):
    # Initialize list with predefined type
    partition_list = List.empty_list(TUPLE_TYPE)
    
    for w in range(k + 1):
        p = generate_partitions(w)
        
        # Handle empty partitions
        if p.size == 0:
            empty_2d = np.empty((0, 3), dtype=np.int64)
            empty_1d = np.empty(0, dtype=np.int64)
            empty_coeffs = np.empty(0, dtype=np.float64)
            entry = (empty_2d, empty_1d, empty_1d, empty_1d, empty_coeffs)
            partition_list.append(entry)
            continue
            
        # Process non-empty partitions
        a_1 = p[:, 0].copy()
        a_2 = p[:, 1].copy()
        a_3 = p[:, 2].copy()
        
        coeffs = np.zeros(len(p), dtype=np.float64)
        for i in range(len(p)):
            coeffs[i] = multi_coeff(w, p[i, 0], p[i, 1], p[i, 2])
        
        entry = (p, a_1, a_2, a_3, coeffs)
        partition_list.append(entry)
    
    return partition_list
def precompute_utils(k):
    inv_factorial = np.ones(k + 1, dtype=np.float64)
    binom = np.zeros((k + 1, k + 1), dtype=np.int64)

    # Compute inverse factorials
    for s in range(1, k + 1):
        inv_factorial[s] = inv_factorial[s - 1] / s

    # Compute binomial coefficients
    for s in range(k + 1):
        binom[s, 0] = 1
        for f in range(1, s + 1):
            binom[s, f] = binom[s - 1, f - 1] + binom[s - 1, f]

    return inv_factorial, binom


@njit
def cal_index(w):
    index_1=[]
    index_2=[]
    for a1 in range(w + 1):
        for a2 in range(w - a1 + 1):
            for a3 in range(w - a1 - a2 + 1):
                index_2.append((a1,a2,a3))
                for a4 in range(w - a1 - a2 - a3 + 1):
                    for a5 in range(w - a1 - a2 - a3 - a4 + 1):
                        index_1.append((2*a2+a3,2*(w-a1-a2-a3-a4-a5)+a5))
    return index_1,index_2
# @njit
# def index_tri(w):
#     index_1=[]
#     # index_2=[]
#     for s in range(w+1):
#         for a in range(s+1):
#             for b in range(s+1):
#                 if a+b>s:
#                     continue
#                 else:
#                     index_1.append((s,a,b))
#     return index_1
#calculate log-likelihood between boxes
@njit(parallel=True,fastmath=True)
def L_BB_cal(m,k,L_BB,G_1,D_1):
    for i in prange(m - 1):
        for j in range(i + 1, m):
          total_0=0.0
          total_1=0.0
          for s in range(k + 1):
            total_0+= inv_factorial[s]* G_1[0, s, i, j] * D_1[0, s, i, j]
            total_1+= inv_factorial[s] * G_1[1, s, i, j] * D_1[1, s, i, j]
          L_BB[i, j]=total_0+total_1
          L_BB[j, i]=total_0+total_1
    return L_BB
#calculate log-likelihood inside boxes
@njit(parallel=True,fastmath=True)
def L_B_cal(Points_in_box,k,L_B,G_2,D_2):
    for i in Points_in_box:
        total_0=0.0
        total_1=0.0
        for s in prange(k + 1):
          total_0+= inv_factorial[s] * G_2[0, s, i] * D_2[0, s, i]
          total_1+= inv_factorial[s] * G_2[1, s, i] * D_2[1, s, i]

        L_B[i]=total_0+total_1
    return L_B

#calculate diff and D_center
@njit(parallel=True,fastmath=True)
def cal_distance(m,r,B_short,D_center,diff,V,B):
    for i in prange(r):
      diff_value=V[i,:] - B[i]
      diff[1,i]=np.dot(diff_value,diff_value)
      if i<m:
        for j in range(i,m):
            center_value=B_short[i] - B_short[j]
            D_center[1, i, j] = np.dot(center_value,center_value)
            D_center[1, j, i] = np.dot(center_value,center_value)
    return diff,D_center
#power the matrix
@njit(parallel=True)
def power(matrix):
    if matrix.ndim==3:
        x=matrix.shape[0]
        y=matrix.shape[1]
        z=matrix.shape[2]
        for i in prange(x):
            for j in prange(y):
                for s in prange(z):
                    matrix[i,j,s] = matrix[1,j,s] ** i
    else:
        x=matrix.shape[0]
        y=matrix.shape[1]
        for i in prange(x):
            for j in prange(y):
                matrix[i,j]=matrix[1,j]**i
    return matrix

@njit(parallel=True)
def generate_position_matrix(A, m, r):
    position_matrix = np.zeros((m, r), dtype=np.int32)  # Correct initialization

    for j in prange(r):  # Only loop over `r`
        i = A[j].item()  # Directly get the row index
        position_matrix[i, j] = j  # Assign j to the correct row

    return position_matrix

@njit(parallel=True, fastmath=True)
def cal_M_1(position_matrix, V, k, e, m):
    M_1 = np.zeros((2, k + 1, m, m))  # Correct initialization
    r = position_matrix.shape[0]

    # Parallelize only the outermost loop
    for i in prange(m):
        for j in range(i + 1, m):  # Only compute for i < j to avoid redundant calculations
            i_position = np.where(position_matrix[i, :])[0]  # Faster than np.nonzero
            j_position = np.where(position_matrix[j, :])[0]

            for y in range(2):  # Small range; no need for prange
                for w in range(k + 1):
                    total = 0.0  # Thread-local accumulator

                    for i_idx in i_position:
                        for j_idx in j_position:
                            if e[i_idx, j_idx] == y:
                                diff = V[i_idx, :] - V[j_idx, :]
                                diff_sq = np.dot(diff, diff)  # Squared Euclidean distance
                                total += diff_sq**w  # Accumulate the w-th power

                    M_1[y, w, i, j] = total  # Single write to memory
                    M_1[y, w, j, i] = total  # Ensure symmetry

    return M_1

@njit(parallel=True, fastmath=True)
def cal_M_2(position_matrix, V, k, e, r,A):
    M_2 = np.zeros((2, k + 1, r))  # Correct initialization
    r = position_matrix.shape[0]

    # Parallelize only the outermost loop
    for i in prange(r):
      i_idx=A[i].item()
      i_position = np.where(position_matrix[i_idx, :])[0]

      for y in range(2):  # Small range; no need for prange
          for w in range(k + 1):
              total = 0.0  # Thread-local accumulator
              for j_index in i_position:
                if i==j_index:
                  continue
                else:
                  if e[i, j_index] == y:
                      diff = V[i, :] - V[j_index, :]
                      diff_sq = np.dot(diff, diff)  # Squared Euclidean distance
                      total += diff_sq**w  # Accumulate the w-th power

              M_2[y, w, i] = total  # Single write to memor

    return M_2

@njit(parallel=True, fastmath=True)
def cal_M_3(position_matrix, V, k, e, r, m, A, index_3):
  M_3 = np.zeros((2, 2 * k + 1, 2 * k + 1, r, m))
  for i in prange(r):
      i_idx = A[i].item()  # Ensure `i_idx` is an integer

      for j in range(m):
          if i_idx ==j:
              continue
          else:
            j_position = np.where(position_matrix[j, :])[0]

            for y in range(2):
                for w, w_1 in index_3:
                    total = 0.0

                    for j_index in j_position:

                      if e[i_idx, j_index] == y:  # Ensure `e` is properly indexed
                          total += V[j, 0]**w * V[j, 1]**w_1

                    M_3[y, w, w_1, i, j] = total

  return M_3

@njit(parallel=True, fastmath=True)
def cal_M_4(position_matrix, V, k, e, r,A,index_4):
    M_4 = np.zeros((2, 2*k + 1,2*k + 1,2*k+1, r))  # Correct initialization
    r = position_matrix.shape[0]

    # Parallelize only the outermost loop
    for i in prange(r):
      i_idx=A[i].item()
      j_position = np.where(position_matrix[i_idx, :])[0]

      for y in range(2): 
          for a_1,a_2,a_3 in index_4:
              total = 0.0  

              for j_index in j_position:
                  if e[i_idx, j_index] == y:
                      total += (V[j_index,0]**2+V[j_index,1]**2)**a_1*(-2*V[j_index,0])**a_2*(-2*V[j_index,1])**a_3 

              M_4[y, a_1,a_2,a_3,i] = total

    return M_4

@njit(parallel=True,fastmath=True)
def D_1_cal(V,C_short,k,e,position_matrix,m):
    D_1=np.zeros((2,k+1,m,m))
    for i in prange(m-1):
      select_box_i=np.where(position_matrix[i, :])[0]
      for j in range(i+1,m):
          select_box_j=np.where(position_matrix[j, :])[0]
          diff_c = C_short[i, :] - C_short[j, :]
          diff_c_sq = np.dot(diff_c, diff_c)
          for y in range(2):
            for w in range(k + 1):
              total = 0.0
              for i_idx in select_box_i:
                for j_idx in select_box_j:
                    if e[i_idx,j_idx]==y:
                      diff = V[i_idx, :] - V[j_idx, :]
                      diff_sq = np.dot(diff, diff)-diff_c_sq
                      total += diff_sq**w
              D_1[y,w,i,j]=total
    return D_1
@njit(parallel=True,fastmath=True)
def D_2_cal(V,C_short,A,k,e,position_matrix,r):
    D_2=np.zeros((2,k+1,r))
    for i in prange(r):
      i_idx=A[i].item()
      select_box_i=np.where(position_matrix[i_idx, :])[0]
      diff_c = C_short[i_idx, :] - V[i, :]
      diff_c_sq = np.dot(diff_c, diff_c)
      for y in range(2):
        for w in range(k + 1):
          total = 0.0
          for j in select_box_i:
            if e[i,j]==y:
              diff = V[i, :] - V[j, :]
              diff_sq = np.dot(diff, diff)-diff_c_sq
              total += diff_sq**w
          D_2[y,w,i]=total
    return D_2


@njit
def multi_coeff(w, *args):
    """
    Flexible multinomial coefficient calculator
    Computes w! / (a1! * a2! * ... * an!) where sum(args) == w
    """
    if w > 20:
        raise ValueError("w must be ≤ 20 for factorial precision")
    
    # Calculate denominator product
    denominator = 1.0
    sum_args = 0
    for a in args:
        denominator *= fast_factorial(a)
        sum_args += a
    
    # Verify partition sum equals w
    if sum_args != w:
        return 0.0
    
    return fast_factorial(w) / denominator

@njit(parallel=True,fastmath=True)
def sum_log_likelihood(L_BB,L_B):
    L=0
    for i in prange(L_BB.shape[0]-1):
        for j in range(i,L_BB.shape[0]):
            L+=L_BB[i,j]
    for f in prange(L_B.shape[0]):
        L+=0.5*L_B[f]
    return L
@njit(parallel=True)
def cal_center_box(V,A,A_short,counts,r):
  # C=np.empty_like(V)
  # for i in prange(V.shape[0]):
  #   position=(A_short==A[i])
  #   position=np.nonzero(position)[0]
  #   C[i]=1/counts[position]*np.sum(V[(A==A[i])],axis=0)
  """
  Calculates the center of each box.

  Args:
      V: The latent positions (n x d array).
      A: The box assignments for each point in V (n x 1 array).
      A_short: Unique box assignments (m x 1 array).
      counts: Number of points in each box (m x 1 array).

  Returns:
      C: The center of each box corresponding to the points in V (n x d array).
  """
  C = np.zeros((r,2))
  for i in prange(V.shape[0]):
    # Find the index of the box for point i in A_short
    box_index = np.where(A_short == A[i])[0][0]

    # Get the indices of all points in the same box as point i
    points_in_box_indices = np.where(A == A[i])[0]

    # Calculate the center of the box
    C[i] = (1 / counts[box_index]) * np.sum(V[points_in_box_indices], axis=0)
  return C

@njit(parallel=True)
def assign_number_box(counts,A,A_short,r):
  N=np.zeros(r,dtype=np.int64)
  for i in prange(r):
    position=np.where(A_short==A[i])[0]
    N[i]=counts[position[0]]
  return N

def preprocessing(V, e, k, b, theta):
    """V is the latent positions, e is the edge matrix, k is the order of the Taylor expansion, b is the sidelength of box, theta is the parameter of link function
    B is the results of assigning vertices. B_short is non-empty box list,
    diff is the distance between vertex to the center of its box, D_center is the distance between different boxes
    G_1 is the derivative of d(centre[i],centre[j]), G_2 is the derivative of d(V[i],centre[V[i]]) (only when there are more than 1 points in the box)
    M_1,M_2 is the moments of between boxes and inside box, D_1,D_2 is the moments of distance between boxes and inside box,
    L_BB is the log-likelihood between boxes, L_B is the log-likelihood inside large box(contains more than one point)
    """
    r = V.shape[0]
    B = []
    diff = np.zeros((k + 1, r))
    index,index_2=cal_index(k)
    index=np.array(index)
    index_2=np.array(index_2)
    index_1=np.unique(index,axis=0)


    #Assign points to the boxes
    for i in range(r):
        B.append(Assign(V[i, :], b))

    B = np.stack(B)

    #select non-empty boxes,save the locations of the centre
    B_short= np.unique(B, axis=0)
    #get new index to the box
    A = []
    for i in range(B.shape[0]):
        positions = (B_short[:, 0] == B[i, 0]) & (B_short[:, 1] == B[i, 1])
        positions = np.nonzero(positions)[0]
        A.append(positions if len(positions) > 0 else np.array([-1], dtype=np.longlong))

    A = np.stack(A)
    A_short,counts = np.unique(A,axis=0,return_counts=True)

    # select boxes contain more than one point
    large_box=A_short[counts>1]
    selected_positions=[]
    for i in range(large_box.shape[0]):
        select_position=[]
        select_position=np.where(A==large_box[i])[0]
        selected_positions.append(select_position)
    if selected_positions:
        Points_in_box = np.concatenate(selected_positions)
    else:
        Points_in_box = np.array([])
    C=cal_center_box(V,A,A_short,counts,r)
    C_short=np.unique(C,axis=0)
    N=assign_number_box(counts,A,A_short,r)
    m = A_short.shape[0]
    
    # initialize the moments
    G_1 = np.zeros((2, k + 1, m, m))
    G_2 = np.zeros((2, k + 1, r))
    L_BB = np.zeros((m, m))
    L_B = np.zeros(r)
    D_center = np.zeros((k + 1, m, m))
    diff,D_center=cal_distance(m,r,B_short,D_center,diff,V,B)
    # calculating the derivatives
    for i in Points_in_box:
        for s in range(k + 1):
            G_2[0, s, i] = first_order_derivative(x=diff[1,i],theta=theta,e=0,derivative_order=s)
            G_2[1, s, i] = first_order_derivative(x=diff[1,i],theta=theta,e=1,derivative_order=s)
    for i in range(m - 1):
        for j in range(i + 1, m):
            for s in range(k + 1):
                G_1[0, s, i, j] = first_order_derivative(x=D_center[1][i, j],theta=theta,e=0,derivative_order=s)
                G_1[1, s, i, j] = first_order_derivative(x=D_center[1][i, j],theta=theta,e=1,derivative_order=s)
    D_center=power(D_center)
    diff=power(diff)
    position_matrix=generate_position_matrix(A,m,r)
    print("start calculating moments")
    # collecting the moments
    M_1=cal_M_1(position_matrix, V, k, e, m)
    print("#### finish,M_1")
    if Points_in_box.size>0:
        assert Points_in_box.size != 0, "no points in the same box"
        M_2=cal_M_2(position_matrix, V, k, e, r,A)
        print("#### finish,M_2")
        M_3=cal_M_3(position_matrix, V, k, e, r,m,A,index_1)
        print("#### finish,M_3")
        M_4=cal_M_4(position_matrix, V, k, e, r,A,index_2)
        print("#### finish,M_4")
        D_2=D_2_cal(V,C_short,A,k,e,position_matrix,r)
        print("#### finish,D_2")
        L_B=L_B_cal(Points_in_box,k,L_B,G_2,D_2)
        print("#### finish,L_B")
    D_1=D_1_cal(V,C_short,k,e,position_matrix,m)
    print("#### finish,D_1")
    L_BB=L_BB_cal(m,k,L_BB,G_1,D_1)
    print("#### finish,L_BB")
    neighbor_matrix=k_near_neighbor(V,V.shape[0])
    print("#### finish neighbor_matrix")
    # L = np.sum(L_BB) + 0.5 * np.sum(L_B)
    L = sum_log_likelihood(L_BB,L_B)

    return L_BB, L_B, L, M_1,M_2,M_3,M_4, D_1, D_2, G_1, G_2, D_center, diff, A, C_short,N,Points_in_box,counts,position_matrix,neighbor_matrix




#calculate derivative, if derivative_order=0, return the value of link function
@njit
def first_order_derivative(x, theta, e, derivative_order):
    if derivative_order==0:
        if e == 1:
            p = -np.log( 1 + np.exp(3*x-theta))
        else:
            p = -np.log(1 + np.exp(theta-3*x))
    else:
        if e==1:
            p=-3/(np.exp(theta-3*x) + 1)
        else:
            p=3/(np.exp(3*x-theta) + 1)
    return p

### spectral_embedding function
def spectral_embed(A,k):
  degrees = np.array(A.sum(axis=1)).flatten()
  D = np.diag(degrees)
  I = np.identity(A.shape[0])
  L = I - np.matmul(np.linalg.inv(D),A)
#   D_inv_sqrt = np.linalg.inv(np.sqrt(D))
#   L_sym=np.matmul(np.matmul(D_inv_sqrt, L), D_inv_sqrt)
  eigvals, eigvecs = np.linalg.eigh(L)
  sorted_indices = np.argsort(eigvals)


  Embed=eigvecs[:, sorted_indices[1:k+1]]
#   Embed=Norm(Embed)
  return Embed



##### calculate nearest neighbors
def k_near_neighbor(V,k):
  index=pynndescent.NNDescent(V,n_neighbors=k,metric='euclidean')
  return index.neighbor_graph[0]

# @njit(parallel=True)
# def cal_nearest_graph(V,k_near_matrix,k_near_graph):
#   for i in prange(V.shape[0]):
#     for j in prange(V.shape[0]):
#       if j in k_near_graph[i,1:]:
#         k_near_matrix[i,j]+=1
#   return k_near_matrix
# @njit(parallel=True)
# def cal_distance_matrix(V,k_near_graph,k_dist_matrix):
#   z=k_dist_matrix.shape[1]
#   for i in prange(V.shape[0]):
#     for j in range(1,z):
#       index_j=k_near_graph[i,j]
#       k_dist_matrix[i,j]+=np.linalg.norm(V[i,:]-V[index_j,:])**2
#   return k_dist_matrix

# # rank_matrix=np.zeros(neighbor_matrix.shape)
# @njit(parallel=True)
# def average_rank(k_near_matrix,rank_matrix):
#   t=rank_matrix.shape[0]-1
#   for i in prange(k_near_matrix.shape[0]):
#     for j in prange(k_near_matrix.shape[1]):
#       if j==i:
#         rank_matrix[i][j]=(t**2-t)/2
#       else:
#         # rank_matrix[i][j]=rank_matrix[i][j]+np.where(k_near_matrix[i]==j)[0]
#         # Get the index where k_near_matrix[i] is equal to j
#         index_array = np.where(k_near_matrix[i]==j)[0]
#         # If index_array is not empty, get the first element; otherwise, use 0.
#         rank_increment = index_array[0] if index_array.size > 0 else 0
#         rank_matrix[i][j]=rank_matrix[i][j]+rank_increment
#   return rank_matrix

@njit(parallel=True)
def cal_test_statistic(rank_matrix,T):
  test_statistic=np.zeros(rank_matrix.shape[0])
  n=T/2
  t=rank_matrix.shape[0]-1
  average_rank_matrix=(rank_matrix/t-(t-1)/2)**2
  constant_matrix=average_rank_matrix.sum(axis=1)
  constant_factor = 12 * n / (t**2 + t)
  for i in prange(rank_matrix.shape[0]):
    test_statistic[i] = constant_factor * constant_matrix[i]
  return test_statistic


##### update the values after accepting the updated value
@njit(parallel=True,fastmath=True)
def update_U_m_3(proposed_v,U_m_3,i,V,k,A):
  i_idx=A[i].item()
  for a_1 in prange(2*k+1):
    for a_2 in range(2*k+1):
      U_m_3[a_1,a_2,i_idx]+=proposed_v[0]**a_1*proposed_v[1]**a_2-V[i,0]**a_1*V[i,1]**a_2
  return U_m_3
  
@njit(parallel=True,fastmath=True)
def update_U_m_4(proposed_v,U_m_4,i,V,k,A):
  i_idx=A[i].item()
  for a_1 in prange(k+1):
    for a_2 in range(k+1):
      for a_3 in range(k+1):
        if a_1+a_2+a_3>k:
          continue
        else:
          U_m_4[a_1,a_2,a_3,i_idx]+=(proposed_v[0]**2+proposed_v[1]**2)**a_1*(-2*proposed_v[0])**a_2*(-2*proposed_v[1])**a_3-(V[i,0]**2+V[i,1]**2)**a_1*(-2*V[i,0])**a_2*(-2*V[i,1])**a_3
  return U_m_4


@njit(parallel=True,fastmath=True)
def delta_1_cal(i,proposed_v, V,k):
  delta_1=np.zeros((2*k+1,2*k+1))
  for j in prange(2*k+1):
    for s in range(2*k+1):
      delta_1[j,s]=proposed_v[0]**j*proposed_v[1]**s-V[i,0]**j*V[i,1]**s
  return delta_1
# calculate delta_2
@njit(nogil=True, parallel=True, fastmath=True)
def delta_2_optimized_fast(k, e, i, A, M_3, m, delta_1_arr,precomp_partitions):
    delta_2 = np.zeros((2, k + 1, m), dtype=np.float64)
    for idxes in prange(m):
        if i == idxes:
            continue
        M_slice = M_3[:, :, :, i, idxes]
        for w in range(k + 1):
            p, comp1, comp2, exp_factors, coeffs, comp3, comp4 = precomp_partitions[w]
            n_partitions = p.shape[0]
            local_delta0 = 0.0
            local_delta1 = 0.0

            for idx in range(n_partitions):
                coeff = coeffs[idx]
                exp_factor = exp_factors[idx]
                c1 = comp1[idx]
                c2 = comp2[idx]
                delta_1_term = delta_1_arr[comp3[idx.item()].item(), comp4[idx.item()].item()]

                term = coeff * exp_factor * delta_1_term
                local_delta0 += term * M_slice[0, c1, c2]
                local_delta1 += term * M_slice[1, c1, c2]

            delta_2[0, w, idxes] += local_delta0
            delta_2[1, w, idxes] += local_delta1

    return delta_2

@njit(parallel=True,fastmath=True)
def cal_func_h(proposed_v,k):
  func_h=np.zeros((k+1,k+1,k+1,k+1))
  for w in prange(k+1):
    for a_1 in range(w+1):
      for a_2 in range(w+1):
        for a_3 in range(w+1):
          if a_1+a_2+a_3>w:
            continue
          else:
            func_h[w,a_1,a_2,a_3]=proposed_v[0]**a_2*proposed_v[1]**a_3*(proposed_v[0]**2+proposed_v[1]**2)**a_1
  return func_h



### update the likelihood
@njit(parallel=True,fastmath=True)
def update_likelihood_optimized(i,i_box,e,k,L,theta, A,N, M_1,M_2,M_4, L_BB, L_B, r, m, D_2, Points_in_box, C_short, V, delta_2, proposed_v, U_m_3, U_m_4, func_h, binom, inv_factorial,position_matrix,precomp_partitions_2):
    i_box_center = C_short[i_box]
    C_short_new=i_box_center+(proposed_v-V[i,:])/N[i].item()
    # Initialize new arrays
    center_distance = np.zeros((k+1, m))
    G_1_new = np.zeros((2, k+1, m))
    G_2_new = np.zeros((2, k+1, r))
    diff_new = np.zeros((k+1, r))
    M_1_new = np.zeros((2, k+1, m))
    M_2_new = np.zeros((2, k+1, r))
    M_3_new = np.zeros((2, 2*k+1, 2*k+1, m))
    M_4_new = np.zeros((2, k+1, k+1, k+1))
    D_1_new = np.zeros((2, k+1, m))
    D_2_new = np.zeros((2, k+1, r))
    L_B_new = np.zeros(r)
    L_BB_new = np.zeros(m)

    # Precompute M_4_new outside parallel loop
    for a1 in range(k+1):
        for a2 in range(k+1):
            for a3 in range(k+1):
                for y in range(2):
                    M_4_new[y, a1, a2, a3] = M_4[y, a1, a2, a3, i] + U_m_4[a1, a2, a3, i_box]

    # Parallel loop over idxes
    for idxes in prange(m):
        if idxes == i_box:
            continue
        # Compute center distance powers
        diff_center = C_short_new - C_short[idxes]
        diff_sq = np.dot(diff_center, diff_center)
        current_power = 1.0
        center_distance[0, idxes] = current_power
        for w in range(1, k+1):
            current_power *= diff_sq
            center_distance[w, idxes] = current_power
        # Compute G_1 derivatives
        for w in range(k+1):
            G_1_new[0, w, idxes] = first_order_derivative(diff_sq, theta, 0, w)
            G_1_new[1, w, idxes] = first_order_derivative(diff_sq, theta, 1, w)
        # Update M_1 and D_1
        for w in range(k+1):
            M_1_new[0, w, idxes] = M_1[0, w, i_box, idxes] + delta_2[0, w, idxes]
            M_1_new[1, w, idxes] = M_1[1, w, i_box, idxes] + delta_2[1, w, idxes]
        for a1 in range(2*k+1):
            for a2 in range(2*k+1):
                for y in range(2):
                    M_3_new[y, a1, a2, idxes] = U_m_3[a1, a2, idxes]
        for s in range(k+1):
            for y in range(2):
                total = 0.0
                for f in range(s + 1):
                    term = 0.0
                    term = binom[s, f] * (-center_distance[s-f, idxes]) * M_1_new[y, f, idxes]
                    total += term
                D_1_new[y, s, idxes] = total

    # Process vertices in the same box
    j_positions = position_matrix[i_box]
    delta_3 = np.zeros((2, k+1, r))
    for j in prange(len(j_positions)):
        j_idx = j_positions[j]
        if j_idx == i:
            diff_vertice = proposed_v - C_short_new 
            diff_sq = np.dot(diff_vertice, diff_vertice)
            current_power = 1.0
            diff_new[0, i] = current_power
            for w in range(1, k+1):
                current_power *= diff_sq
                diff_new[w, i] = current_power
        else:
            diff_vertice = V[j_idx] - C_short_new 
            diff_sq = np.dot(diff_vertice, diff_vertice)
            current_power = 1.0
            diff_new[0, j_idx] = current_power
            for w in range(1, k+1):
                current_power *= diff_sq
                diff_new[w, j_idx] = current_power
        # Update M_2 and delta_3
        if j_idx != i:
            for w in range(k+1):
                for y in range(2):
                    if e[i, j_idx] == y:
                        diff_1 = V[i] - V[j_idx]
                        diff_sq_1 = np.dot(diff_1, diff_1)
                        diff_2 = proposed_v - V[j_idx]
                        diff_sq_2 = np.dot(diff_2, diff_2)
                        delta_3[y, w, j_idx] += (diff_sq_1**w - diff_sq_2**w)
                        M_2_new[y, w, j_idx] = M_2[y, w, j_idx] + delta_3[y, w, j_idx]

    # Update D_2 and L_B if i is in Points_in_box
    if i in Points_in_box:
        for s in prange(k+1):
            total_0 = 0.0
            total_1 = 0.0
            for f in range(s + 1):
                coeff = binom[s, f] * (-diff_new[s - f, i])
                p, a1, a2, a3, coeffs = precomp_partitions_2[f]
                n_part = p.shape[0]
                local_0 = 0.0
                local_1 = 0.0
                for idx in range(n_part):
                    c = coeffs[idx]
                    term = c * func_h[f, a1[idx], a2[idx], a3[idx]]
                    local_0 += term * M_4_new[0, a1[idx], a2[idx], a3[idx]]
                    local_1 += term * M_4_new[1, a1[idx], a2[idx], a3[idx]]
                total_0 += coeff * local_0
                total_1 += coeff * local_1
            D_2_new[0, s, i] = total_0
            D_2_new[1, s, i] = total_1
            G_2_new[0, s, i] = first_order_derivative(diff_new[1, i], theta, 0, s)
            G_2_new[1, s, i] = first_order_derivative(diff_new[1, i], theta, 1, s)

        # Update D_2 for other j in box
        for j in prange(len(j_positions)):
            j_idx = j_positions[j]
            if j_idx == i:
                continue
            for s in range(k+1):
                total_0 = 0.0
                total_1 = 0.0
                for f in range(s + 1):
                    term = binom[s, f] * (-diff_new[s - f, j_idx])
                    total_0 += term * M_2[0, f, j_idx]
                    total_1 += term * M_2[1, f, j_idx]
                D_2_new[0, s, j_idx] = D_2[0, s, j_idx] + total_0
                D_2_new[1, s, j_idx] = D_2[1, s, j_idx] + total_1
                G_2_new[0, s, j_idx] = first_order_derivative(diff_new[1, j_idx], theta, 0, s)
                G_2_new[1, s, j_idx] = first_order_derivative(diff_new[1, j_idx], theta, 1, s)

        # Compute L_B_new
        for j in prange(len(j_positions)):
            j_idx = j_positions[j]
            total_0 = 0.0
            total_1 = 0.0
            for s in range(k+1):
                total_0 += inv_factorial[s] * G_2_new[0, s, j_idx] * D_2_new[0, s, j_idx]
                total_1 += inv_factorial[s] * G_2_new[1, s, j_idx] * D_2_new[1, s, j_idx]
            L_B_new[j_idx] = total_0 + total_1

    # Update L_BB_new
    for f in prange(m):
        if f == i_box:
          continue
        else:
          total_0 = 0.0
          total_1 = 0.0
          for s in range(k+1):
              total_0 += inv_factorial[s] * G_1_new[0, s, f] * D_1_new[0, s, f]
              total_1 += inv_factorial[s] * G_1_new[1, s, f] * D_1_new[1, s, f]
          L_BB_new[f] = total_0 + total_1

    # Calculate total L
    total_L = 0.0
    for f in prange(m):
        if f != i_box:
            total_L += L_BB_new[f] - L_BB[f, i_box]
    for j in prange(len(j_positions)):
        j_idx = j_positions[j]
        total_L += 0.5*(L_B_new[j_idx] - L_B[j_idx])

    L_new= L+total_L

    return (M_1_new, M_2_new, M_3_new, M_4_new, D_1_new, L_BB_new,
            D_2_new, diff_new, center_distance, C_short_new, G_1_new,
            G_2_new, L_B_new,L_new)

@njit
def Recenter(V,C_short,center):
  # after T=1000 times, we recenter the vertices and the centers by v_i=v_i-V.mean()
  # so we need to update d(v_i,C_i) and diff(1,i) G_2[y,1,i]
  # center=V.mean(axis=0)
  V_new=V-center
  C_short_new=C_short-center
  return V_new,C_short_new

@njit(parallel=True,fastmath=True)
def update_moments(i,i_box,k,m,M_1_new, M_2_new, M_3_new, M_4_new, D_1_new, L_BB_new,D_2_new, diff_new, center_distance, C_short_new, G_1_new,G_2_new, L_B_new,M_1, M_2, M_3, M_4, D_1, L_BB,D_2, diff, D_center, C_short, G_1,G_2, L_B,L_new,Points_in_box,position_matrix):
  C_short[i_box,:]=C_short_new
  for a1 in range(k+1):
        for a2 in range(k+1):
            for a3 in range(k+1):
                for y in range(2):
                    M_4[y, a1, a2, a3,i] = M_4_new[y, a1, a2, a3]
  # Parallel loop over idxes
  for idxes in prange(m):
      if idxes == i_box:
          continue
      for w in range(k+1):
        D_center[w,i_box,idxes]=center_distance[w,idxes]
        D_center[w,idxes,i_box]=center_distance[w,idxes]
      # Compute G_1 derivatives
        G_1[0,w,idxes,i_box]=G_1_new[0,w,idxes]
        G_1[0,w,i_box,idxes]=G_1_new[0,w,idxes]
        G_1[1,w,idxes,i_box]=G_1_new[1,w,idxes]
        G_1[1,w,i_box,idxes]=G_1_new[1,w,idxes]
      # Update M_1 and D_1
        M_1[0, w, i_box, idxes]=M_1_new[0, w, idxes]
        M_1[0, w, idxes, i_box]=M_1_new[0, w, idxes]
        M_1[1, w, i_box, idxes]=M_1_new[1, w, idxes]
        M_1[1, w, idxes, i_box]=M_1_new[1, w, idxes]
        D_1[0,w,i_box,idxes]=D_1_new[0, w, idxes]
        D_1[0,w,idxes,i_box]=D_1_new[0, w, idxes]
        D_1[1,w,i_box,idxes]=D_1_new[1, w, idxes]
        D_1[1,w,idxes,i_box]=D_1_new[1, w, idxes]

      for a1 in range(2*k+1):
          for a2 in range(2*k+1):
              M_3[1, a1, a2, i,idxes] = M_3_new[1, a1, a2, idxes]
              M_3[0, a1, a2, i,idxes] = M_3_new[0, a1, a2, idxes]


  # Process vertices in the same box
  j_positions = position_matrix[i_box]
  for j in prange(len(j_positions)):
      j_idx = j_positions[j]
      for w in range(1, k+1):
          diff[w, j_idx] = diff_new[w, j_idx]
      # Update M_2 and delta_3
      if j_idx != i:
        for w in range(k+1):
            M_2[0, w, j_idx] = M_2_new[0, w, j_idx]
            M_2[1, w, j_idx] = M_2_new[1, w, j_idx]

  # Update D_2 and L_B if i is in Points_in_box
  if i in Points_in_box:
      # Update D_2 for other j in box
      for j in prange(len(j_positions)):
          j_idx = j_positions[j]
          if j_idx == i:
              continue
          for s in range(k+1):
              D_2[0, s, j_idx] = D_2_new[0, s, j_idx]
              D_2[1, s, j_idx] = D_2_new[1, s, j_idx]
              G_2[0, s, j_idx] = G_2_new[0, s, j_idx]
              G_2[1, s, j_idx] = G_2_new[1, s, j_idx]

      # Compute L_B_new
      for j in prange(len(j_positions)):
          j_idx = j_positions[j]
          L_B[j_idx] = L_B_new[j_idx]

  # Update L_BB_new
  for f in prange(m):
      if f == i_box:
        continue
      L_BB[i_box,f] = L_BB_new[f]
      L_BB[f,i_box] = L_BB_new[f]

  L=0.0
  L=L_new
  return M_1, M_2, M_3, M_4, D_1, L_BB,D_2, diff, D_center, C_short, G_1,G_2, L_B,L

@njit
def multivariate_normal_sample(mean, cov, size=1):
    """Optimized multivariate normal sampler with precomputed Cholesky."""
    L = np.linalg.cholesky(cov)
    z = np.random.standard_normal((size, mean.shape[0]))
    return mean + z @ L.T

@njit
def in_box(x, box_center, box_halfwidth):
    """Vectorized box check with precomputed boundaries."""
    return ((box_center[0] - box_halfwidth <= x[0] <= box_center[0] + box_halfwidth) &
            (box_center[1] - box_halfwidth <= x[1] <= box_center[1] + box_halfwidth))

@njit
def propose_mix(vertex, box_center, b, alpha=0.75, batch_size=100):
    """
    Optimized proposal generator with:
    - Batch sampling to reduce iterations
    - Precomputed box boundaries
    - Vectorized operations
    """
    sigma = b * alpha
    box_halfwidth = b / 2
    cov = (sigma ** 2) * np.eye(2)  # Diagonal covariance for efficiency
    
    while True:
        # Generate batch of samples directly using optimized normal sampler
        samples = multivariate_normal_sample(vertex, cov, batch_size)
        
        # Vectorized box check for entire batch
        for i in range(batch_size):
            if in_box(samples[i], box_center, box_halfwidth):
                return samples[i]
            

def cal_posterior_prob(S,k,b,theta,e,T,L_BB, L_B, L, M_1,M_2,M_3,M_4, D_1, D_2, G_1, G_2, D_center, diff, A, C_short,N,Points_in_box,inv_factorial,binom_coeff,position_matrix,precomp_for_delta2,precomp_partitions_2,test_statistic,rank_matrix ):
  """
  where i and j are node in V, q is order of taylor expansion, k is the required k-th closest, theta is parameter, b is side length of box, e is the adjacency matrix
  """
  V=S.copy()
  r = V.shape[0]
  m = C_short.shape[0]
  k_near_matrix=np.zeros((r,r))
  k_dist_matrix=np.zeros((r,r))

  for t in range(T):
    print("current calculate ",t,"iteration")
    # at start of each t, we set the update function to be zero matrix
    U_m_3=np.zeros((2*k+1,2*k+1,m))
    U_m_4=np.zeros((k+1,k+1,k+1,m))
    # np.save(os.path.join('/content/drive/My Drive/experiments_data',f'V_{t}.npy'), V)
    # np.save(os.path.join('/content/drive/My Drive/experiments_data', f'k_near_matrix_{t}.npy'), k_near_matrix)
    # np.save(os.path.join('/content/drive/My Drive/experiments_data', f'k_dist_matrix_{t}.npy'), k_dist_matrix)
    for i in range(r):
      i = int(i)  # Ensure `i` is an integer
      V_new = V.copy()
      i_box= A[i].item()
      proposed_v = propose_mix(V_new[i, :], C_short[i], b)
      func_h=cal_func_h(proposed_v,k)
      delta_1=delta_1_cal(i,proposed_v, V,k)
      delta_2=delta_2_optimized_fast(k, e, i, A, M_3, m, delta_1,precomp_for_delta2)
      print("### start update likelihood")
      M_1_new, M_2_new, M_3_new, M_4_new, D_1_new, L_BB_new,D_2_new, diff_new, center_distance, C_short_new, G_1_new,G_2_new, L_B_new,L_new=update_likelihood_optimized(i,i_box,e,k,L,theta, A,N, M_1,M_2,M_4, L_BB, L_B, r, m, D_2, Points_in_box, C_short, V, delta_2, proposed_v, U_m_3, U_m_4, func_h, binom, inv_factorial,position_matrix,precomp_partitions_2)
      alpha_1 = np.minimum(0, L_new - L)

      if np.log(np.random.rand()) < alpha_1:
          V_new[i, :] = proposed_v
          print("### start update moments")
          M_1, M_2, M_3, M_4, D_1, L_BB,D_2, diff, D_center, C_short, G_1,G_2, L_B,L= update_moments(i,i_box,k,m,M_1_new, M_2_new, M_3_new, M_4_new, D_1_new, L_BB_new,D_2_new, diff_new, center_distance, C_short_new, G_1_new,G_2_new, L_B_new,M_1, M_2, M_3, M_4, D_1, L_BB,D_2, diff, D_center, C_short, G_1,G_2, L_B,L_new,Points_in_box,position_matrix)
    if (t>0)&((t/T)%0.25==0):
      print("has run ",25*(4*t/T),"percent")
      center=V.mean(axis=0)
      V,C_short=Recenter(V,C_short,center)
    elif t>=(T/2):
      print("#### start computing statistics")
      k_near_graph=k_near_neighbor(V,r)
      k_near_matrix=cal_nearest_graph(V,k_near_matrix,k_near_graph)
      k_dist_matrix=cal_distance_matrix(V,k_near_graph,k_dist_matrix)
      # calculate rank
      rank_matrix=average_rank(k_near_graph,rank_matrix)
  test_statistic=cal_test_statistic(rank_matrix,T/2)
  return k_near_matrix,k_dist_matrix,test_statistic,V



from scipy.sparse import csr_matrix
from sknetwork.data import load_netset
dataset = load_netset('openflights')
adjacency = dataset.adjacency
names = dataset.names

e=adjacency.toarray()
embedding=spectral_embed(e,2)
n=embedding.shape[0]

V=embedding.copy()
k = 1
theta = 0.5
b=0.02
#calculate Taylor expansion coeff
inv_factorial, binom_coeff = precompute_utils(k)
precomp_for_delta2 = precompute_partitions(k)
precomp_partitions_2 = precompute_partitions_2(k)

print("##### start proprecessing")
L_BB, L_B, L, M_1,M_2,M_3,M_4, D_1, D_2, G_1, G_2, D_center, diff, A,C_short,N,Points_in_box,counts,position_matrix,neighbor_matrix=preprocessing(V, e, k, b, theta)
test_statistic=np.zeros(neighbor_matrix.shape[0])  
rank_matrix=np.zeros(neighbor_matrix.shape)


@njit(parallel=True)
def cal_nearest_graph(V, k_near_matrix, k_near_graph):
    for i in prange(k_near_graph.shape[0]):
        neighbors = k_near_graph[i, 1:]  # Skip the first element if self-included
        for neighbor in neighbors:
            if neighbor >= 0 and neighbor < k_near_matrix.shape[1]:
                k_near_matrix[i, neighbor] += 1
    return k_near_matrix

@njit(parallel=True)
def cal_distance_matrix(V, k_near_graph, k_dist_matrix):
    z = k_dist_matrix.shape[1]
    num_features = V.shape[1]
    for i in prange(V.shape[0]):
        for j in range(1, z):
            index_j = k_near_graph[i, j]
            if index_j < 0 or index_j >= V.shape[0]:
                continue
            dist_sq = 0.0
            for d in range(num_features):
                diff = V[i, d] - V[index_j, d]
                dist_sq += diff * diff
            k_dist_matrix[i, j] += dist_sq
    return k_dist_matrix

@njit(parallel=True,fastmath=True)
def average_rank(k_near_matrix, rank_matrix):
    t = rank_matrix.shape[0] - 1
    n = k_near_matrix.shape[0]
    m = k_near_matrix.shape[1]
    for i in prange(n):
        index_map = np.full(n, -1, dtype=np.int64)
        for k in range(m):
            j_val = k_near_matrix[i, k]
            if j_val >= 0 and j_val < n and index_map[j_val] == -1:
                index_map[j_val] = k
        for j in prange(n):
            if j == i:
                rank_matrix[i, j] = (t * t - t) // 2
            else:
                idx = index_map[j]
                if idx != -1:
                    rank_matrix[i, j] += idx
    return rank_matrix

k_near_matrix,k_dist_matrix,test_statistic,V=cal_posterior_prob(V,k,b,theta,e,20,L_BB, L_B, L, M_1,M_2,M_3,M_4, D_1, D_2, G_1, G_2, D_center, diff, A, C_short,N,Points_in_box,inv_factorial,binom_coeff,position_matrix,precomp_for_delta2,precomp_partitions_2,test_statistic,rank_matrix)
i=np.random.randint(V.shape[0])
r=V.shape[0]
i_box=A[i].item()
m=C_short.shape[0]
# Pre-allocate these outside main loop
U_m_3 = np.zeros((2*k+1, 2*k+1, m)) 
U_m_4 = np.zeros((k+1, k+1, k+1, m))
proposed_v = propose_mix(V[i, :], C_short[i_box], b)
assert np.all(proposed_v >= C_short[i_box] - b/2)
assert np.all(proposed_v <= C_short[i_box] + b/2)
i_box= A[i].item()
func_h=cal_func_h(proposed_v,k)
delta_1=delta_1_cal(i,proposed_v, V,k)
delta_2=delta_2_optimized_fast(k, e, i, A, M_3, m, delta_1,precomp_for_delta2)
print("### start update likelihood")
M_1_new, M_2_new, M_3_new, M_4_new, D_1_new, L_BB_new,D_2_new, diff_new, center_distance, C_short_new, G_1_new,G_2_new, L_B_new,L_new=update_likelihood_optimized(i,i_box,e,k,L,theta, A,N, M_1,M_2,M_4, L_BB, L_B, r, m, D_2, Points_in_box, C_short, V, delta_2, proposed_v, U_m_3, U_m_4, func_h, binom_coeff, inv_factorial,position_matrix,precomp_partitions_2)
alpha_1 = np.minimum(0, L_new - L)
# Should show smooth changes
print(f"L: {L} -> L_new: {L_new}") 