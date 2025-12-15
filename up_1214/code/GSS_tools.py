# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
import random
from numpy import linalg as LA
from numpy.linalg import inv
from scipy.linalg import schur, eigvals
from scipy.linalg import sqrtm
from scipy.optimize import curve_fit
from tqdm import tqdm

dev = qml.device("default.mixed", wires=1)

# Single qubit Clifford group:
single_qubit_cliffords = [
 'HH',
 'H', 'S',
 'HS', 'SH', 'SS',
 'HSH', 'HSS', 'SHS', 'SSH', 'SSS',
 'HSHS', 'HSSH', 'HSSS', 'SHSS', 'SSHS',
 'HSHSS', 'HSSHS', 'SHSSH', 'SHSSS', 'SSHSS',
 'HSHSSH', 'HSHSSS', 'HSSHSS']


# +
# Functions for generating the noisy random circuit:
# -

def rand_pi(N,max_bound):
    dict={}
    p1=0.0
    p2=0.0
    p3=0.0
    p0=0.0
    i=1
    while i<=N:
        c3=random.uniform(0,max_bound)
        c1=random.uniform(0,max_bound)
        c2=random.uniform(0,max_bound)
        
        c0=(1.0-c1-c2-c3)
        
        p1=p1+c1
        p2=p2+c2
        p3=p3+c3
        p0=p0+c0
        i=i+1
        
    dict['pI']=p0*(1.0/N)
    dict['pX']=p1*(1.0/N)
    dict['pY']=p2*(1.0/N)
    dict['pZ']=p3*(1.0/N)
    return dict


def rand_Cl1_unimat(list_clifford_strings):
    choice=random.randint(0, len(list_clifford_strings)-1)
    clifford=list_clifford_strings[choice]
    gen_gates=list(clifford)
    g=np.eye(2,dtype=complex)
    for gate in gen_gates:
        if gate == 'H':
            g=np.matmul(g,qml.Hadamard(wires=0).matrix())
        else:
            sign = 1 
            g=np.matmul(g,qml.PhaseShift(sign * np.pi/2, wires=0).matrix())
    return g


def sequence_matrices_noise(list_clifford_strings,m):
    dict_mat={}
    seq_ideal=np.eye(2,dtype=complex)
    seq_ideal_inv=np.eye(2,dtype=complex)
    for i in range(m):
        C=rand_Cl1_unimat(list_clifford_strings)
        seq_ideal=np.matmul(C,seq_ideal)
        dict_mat[str(i)]=C
    return dict_mat


def noisy_Clifford_sequence(dict,m,paulis):
    K0 = np.sqrt(paulis['pI'])*np.eye(2)
    K1 = np.sqrt(paulis['pX'])*np.array([[0.0,1.0],[1.0,0.0]])
    K2 = np.sqrt(paulis['pY'])*np.array([[0.0,0.0-1j],[0.0+1j,0.0]])
    K3 = np.sqrt(paulis['pZ'])*np.array([[1.0,0.0],[0.0,-1.0]])
    for i in range(m):
        qml.QubitUnitary(dict[str(i)], wires=0,id="Clifford"+"_"+str(i)) # Corresponds to C|0><0|C^{\dagger}
    # Adding the noise:
        qml.QubitChannel([K0, K1, K2, K3], wires=0)
 
    return qml.state()


# +
# Normalized Pauli basis:

I=np.eye(2,dtype=complex)
X=np.array([[0.0,1.0],[1.0,0.0]],dtype=complex)
Y=np.array([[0.0,-1j*1.0],[1j*1.0,0.0]],dtype=complex)
Z=np.array([[1.0,0.0],[0.0,-1.0]],dtype=complex)
norm_fac=1.0/np.sqrt(2.0)

norm_pauli_basis={'0':norm_fac*I, # sigma_0 
                  '1':norm_fac*X, # sigma_x
                  '2':norm_fac*Y, # sigma_y
                  '3':norm_fac*Z} # sigma_z 


# -

def GSS_experiment(params,m_list,K_tot):
    results_seq=np.empty( (len(m_list),K_tot), dtype=dict)
    results_bit=np.zeros( (len(m_list),K_tot) )
    # Specify the qubits in the system:
    qubits=0
    # Define the input state:
    rho = np.zeros((2,2), dtype=np.complex128) # state |0><0|
    rho[0, 0] = 1
    # Define projection states:
    rho_0=rho # state |0><0|
    rho_1=np.zeros((2,2), dtype=np.complex128)
    rho_1[1,1] = 1 # state |1><1|
    e0=rho_0
    e1=rho_1
    bit_list=[]
    
    
    for i,m in enumerate(tqdm(m_list)):
        for K in np.arange(K_tot):
        # Computing sequence of Clifford matrices to use:
            seq=sequence_matrices_noise(single_qubit_cliffords,m)
            results_seq[i][K]=seq
        # Initialize QNodes:
            noisy_qnode = qml.QNode(noisy_Clifford_sequence, dev)
        # Execute the QNodes:
            noisy_state= noisy_qnode(seq,m,params)
            
            prob_1 = np.trace(np.matmul(e1.conj().T,noisy_state)).real
            prob_0 = 1-prob_1
        
        #we do 1 shot per sequence as in the paper
            choice=np.random.rand() # pick a random number
            if choice>prob_1:
                outcome=0
                results_bit[i][K]=outcome
            else: 
                outcome=1
                results_bit[i][K]=outcome

    return results_seq,results_bit  


# +
# Function that constructs the Pauli transfer matrix for any given gate:

def PTM(gate):
    C=gate
    Cd=gate.conj().T
    R=np.zeros((len(norm_pauli_basis),len(norm_pauli_basis)),dtype=complex)
    for i in np.arange(len(norm_pauli_basis)):
        sigma_i=norm_pauli_basis[str(i)]
        for j in np.arange(len(norm_pauli_basis)):
            sigma_j=norm_pauli_basis[str(j)]
            R[i,j]=np.trace(np.matmul(sigma_i.conj().T,np.matmul(C,np.matmul(sigma_j,Cd))))
    return R


# +
# Function constructing the sequence correlation function (eq.(2)) in the paper:

def f(seq,A,E,rho):
    op=np.eye(4,dtype=complex)
    E=E.flatten() # --> |E>>
    rho=rho.flatten()
    for k in np.arange(len(seq)-1):
            #print(k)
            gate=seq[str(k)]
            C=PTM(gate) # PTM representation
            C[0,:]=0.0
            C[:,0]=0.0
            op=np.matmul(C,op)
            op=np.matmul(A,op)
            
    gate=seq[str(len(seq)-1)]
    C=PTM(gate) # PTM representation
    C[0,:]=0.0
    C[:,0]=0.0
    op=np.matmul(C,op)
    res=np.matmul(E.conj().T,np.matmul(op,rho))
    return res


# -

def k_estimator_mean(dic_seq,dic_bits,K_tot):
    rho = (1.0/np.sqrt(2.0))*np.eye((2), dtype=np.complex128) # state |0>>
    
    
    A=np.eye(4)
    A[0,0]=0.0
    
    r=0.0
    
    for i in tqdm(np.arange(K_tot)):
        outcome = dic_bits[i]
        
        #we pick the projector based on the measurement outcome
        if outcome == 0:
            E=(1.0/np.sqrt(2.0))*np.array([1.0,0.0,0.0,1.0])
        else:
            E=(1.0/np.sqrt(2.0))*np.array([1.0,0.0,0.0,-1.0])

        fA=f(dic_seq[i],A,E,rho)
        
        r=r+fA
 
    return r/K_tot


# +
# Median-0f-means: Divide the total number of collected sampes (S=K_tot) into N sub-sets (aka batches).
#                  The function outputs the mean evaluated for each sub-set. 
#                  The final result can be obtained by taking the median of the vector.

def k_estimator_mom(dic_seq,dic_bits,K_tot,N):
    rho = (1.0/np.sqrt(2.0))*np.eye((2), dtype=np.complex128) # state |0>>
    K=int(K_tot/N) # total number of batches
    n=np.arange(0,K_tot,N) # starting point for each evaluation of the sample mean over the batches
    A=np.eye(4)
    A[0,0]=0.0
    list_res=[]
    for i in tqdm(np.arange(K)): # for each batch, compute the mean
        r=0.0
        l=n[i]
        for j in np.arange(N):
            outcome = dic_bits[l+j]
            #we pick the projector based on the measurement outcome
            if outcome == 0:
                E=(1.0/np.sqrt(2.0))*np.array([1.0,0.0,0.0,1.0])
            else:
                E=(1.0/np.sqrt(2.0))*np.array([1.0,0.0,0.0,-1.0])

            fA=f(dic_seq[l+j],A,E,rho)
        
            r=r+fA
        list_res.append(r/N)
    return np.array(list_res).real


# -

# Fitting model for the sequence average (see eq.(5) in the original paper):
def model_fit(m,c0,p):
    return c0*p**(m-1.0)


