import numpy as np
from scipy.linalg import eigh, qr

# Define dd as a placeholder function (user should replace this with actual logic if needed)
def dd(x):
    return 1.0 if x == 0 else 0.0



def gso(a, kx, ry, rz):
	return np.array([[(0.15915494309189535*(513.4253797050053*dd(ry)*dd(rz) + 3.970973114137498*a**2*dd(ry)*dd(rz) - 199.03308034282972*np.cos(a*kx)*dd(ry)*dd(rz) - 31.677098575367705*(3.141592653589793*dd(-1. + ry)*dd(rz) + 3.141592653589793*dd(1. + ry)*dd(rz)) - 46.02166990090616*dd(ry)*(1.2533141373155001*dd(-1. + rz) + 1.2533141373155001*dd(1. + rz))))/a**2, 0., (0.15915494309189535*((0. - 4.917450345633026*1j)*((0. + 3.141592653589793*1j)*dd(-1. + ry)*dd(rz) - (0. + 3.141592653589793*1j)*dd(1. + ry)*dd(rz)) - 30.897251760466602*dd(ry)*dd(rz)*np.sin(a*kx)))/a, (0.15915494309189535*((0. - 4.917450345633026*1j)*((0. + 3.141592653589793*1j)*dd(-1. + ry)*dd(rz) - (0. + 3.141592653589793*1j)*dd(1. + ry)*dd(rz)) + 30.897251760466602*dd(ry)*dd(rz)*np.sin(a*kx)))/a, ((0. + 4.630696198911051*1j)*dd(ry)*dd(-1. + rz))/a - ((0. + 4.630696198911051*1j)*dd(ry)*dd(1. + rz))/a, 0., 0., 0., 0., 0.], [0., (0.15915494309189535*(958.8447974942432*dd(ry)*dd(rz) + 1.8723892215395164*a**2*dd(ry)*dd(rz) - 455.48000967665007*np.cos(a*kx)*dd(ry)*dd(rz) - 72.4918950195832*(3.141592653589793*dd(-1. + ry)*dd(rz) + 3.141592653589793*dd(1. + ry)*dd(rz)) - 19.103262588064492*dd(ry)*(1.2533141373155001*dd(-1. + rz) + 1.2533141373155001*dd(1. + rz))))/a**2, 0., 0., 0., 0., 0., 0., 0., 0.], [(0.15915494309189535*((0. + 4.917450345633026*1j)*((0. + 3.141592653589793*1j)*dd(-1. + ry)*dd(rz) - (0. + 3.141592653589793*1j)*dd(1. + ry)*dd(rz)) - 30.897251760466602*dd(ry)*dd(rz)*np.sin(a*kx)))/a, 0., (0.15915494309189535*(-554.338134148627*dd(ry)*dd(rz) + 231.46065209987634*np.cos(a*kx)*dd(ry)*dd(rz) + 36.83810691296881*(3.141592653589793*dd(-1. + ry)*dd(rz) + 3.141592653589793*dd(1. + ry)*dd(rz)) + 36.47003860687391*dd(ry)*(1.2533141373155001*dd(-1. + rz) + 1.2533141373155001*dd(1. + rz))))/a**2, (-3.303066225709695*(-3.141592653589793*dd(-1. + ry)*dd(rz) + 6.283185307179586*np.cos(a*kx)*dd(ry)*dd(rz) - 3.141592653589793*dd(1. + ry)*dd(rz) - 3.141592653589793*dd(-1. + ry)*dd(rz)*np.sin(a*kx) + 3.141592653589793*dd(1. + ry)*dd(rz)*np.sin(a*kx)))/a**2, (2.148169594243227*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz))*((0. - 1.*1j)*((0. + 1.2533141373155001*1j)*dd(-1. + ry) - (0. + 1.2533141373155001*1j)*dd(1. + ry)) + 2.5066282746310002*dd(ry)*np.sin(a*kx)))/a**2, 0., 0., 0., 0., 0.], [(0.15915494309189535*((0. + 4.917450345633026*1j)*((0. + 3.141592653589793*1j)*dd(-1. + ry)*dd(rz) - (0. + 3.141592653589793*1j)*dd(1. + ry)*dd(rz)) + 30.897251760466602*dd(ry)*dd(rz)*np.sin(a*kx)))/a, 0., (-3.303066225709695*(-3.141592653589793*dd(-1. + ry)*dd(rz) + 6.283185307179586*np.cos(a*kx)*dd(ry)*dd(rz) - 3.141592653589793*dd(1. + ry)*dd(rz) + 3.141592653589793*dd(-1. + ry)*dd(rz)*np.sin(a*kx) - 3.141592653589793*dd(1. + ry)*dd(rz)*np.sin(a*kx)))/a**2, (0.15915494309189535*(-554.338134148627*dd(ry)*dd(rz) - 1.1736990153811466*a**2*dd(ry)*dd(rz) + 231.46065209987634*np.cos(a*kx)*dd(ry)*dd(rz) + 36.83810691296881*(3.141592653589793*dd(-1. + ry)*dd(rz) + 3.141592653589793*dd(1. + ry)*dd(rz)) + 36.47003860687391*dd(ry)*(1.2533141373155001*dd(-1. + rz) + 1.2533141373155001*dd(1. + rz))))/a**2, (-2.148169594243227*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz))*((0. + 1.*1j)*((0. + 1.2533141373155001*1j)*dd(-1. + ry) - (0. + 1.2533141373155001*1j)*dd(1. + ry)) + 2.5066282746310002*dd(ry)*np.sin(a*kx)))/a**2, 0., 0., 0., 0., 0.12841059146347703*dd(ry)*dd(rz)], [((0. + 4.630696198911051*1j)*dd(ry)*dd(-1. + rz))/a - ((0. + 4.630696198911051*1j)*dd(ry)*dd(1. + rz))/a, 0., (2.148169594243227*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz))*((0. + 1.*1j)*((0. + 1.2533141373155001*1j)*dd(-1. + ry) - (0. + 1.2533141373155001*1j)*dd(1. + ry)) + 2.5066282746310002*dd(ry)*np.sin(a*kx)))/a**2, (-2.148169594243227*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz))*((0. - 1.*1j)*((0. + 1.2533141373155001*1j)*dd(-1. + ry) - (0. + 1.2533141373155001*1j)*dd(1. + ry)) + 2.5066282746310002*dd(ry)*np.sin(a*kx)))/a**2, (0.15915494309189535*(-408.4667344978722*dd(ry)*dd(rz) - 2.2757697182604457*a**2*dd(ry)*dd(rz) + 99.75835830102662*np.cos(a*kx)*dd(ry)*dd(rz) + 15.877035838340799*(3.141592653589793*dd(-1. + ry)*dd(rz) + 3.141592653589793*dd(1. + ry)*dd(rz)) + 83.35899662927821*dd(ry)*(1.2533141373155001*dd(-1. + rz) + 1.2533141373155001*dd(1. + rz))))/a**2, 0., 0., 0., 0.12841059146347703*dd(ry)*dd(rz), 0.], [0., 0., 0., 0., 0., (0.15915494309189535*(513.4253797050053*dd(ry)*dd(rz) + 3.970973114137498*a**2*dd(ry)*dd(rz) - 199.03308034282972*np.cos(a*kx)*dd(ry)*dd(rz) - 31.677098575367705*(3.141592653589793*dd(-1. + ry)*dd(rz) + 3.141592653589793*dd(1. + ry)*dd(rz)) - 46.02166990090616*dd(ry)*(1.2533141373155001*dd(-1. + rz) + 1.2533141373155001*dd(1. + rz))))/a**2, 0., (0.15915494309189535*((0. - 4.917450345633026*1j)*((0. + 3.141592653589793*1j)*dd(-1. + ry)*dd(rz) - (0. + 3.141592653589793*1j)*dd(1. + ry)*dd(rz)) + 30.897251760466602*dd(ry)*dd(rz)*np.sin(a*kx)))/a, (0.15915494309189535*((0. - 4.917450345633026*1j)*((0. + 3.141592653589793*1j)*dd(-1. + ry)*dd(rz) - (0. + 3.141592653589793*1j)*dd(1. + ry)*dd(rz)) - 30.897251760466602*dd(ry)*dd(rz)*np.sin(a*kx)))/a, ((0. + 4.630696198911051*1j)*dd(ry)*dd(-1. + rz))/a - ((0. + 4.630696198911051*1j)*dd(ry)*dd(1. + rz))/a], [0., 0., 0., 0., 0., 0., (0.15915494309189535*(958.8447974942432*dd(ry)*dd(rz) + 1.8723892215395164*a**2*dd(ry)*dd(rz) - 455.48000967665007*np.cos(a*kx)*dd(ry)*dd(rz) - 72.4918950195832*(3.141592653589793*dd(-1. + ry)*dd(rz) + 3.141592653589793*dd(1. + ry)*dd(rz)) - 19.103262588064492*dd(ry)*(1.2533141373155001*dd(-1. + rz) + 1.2533141373155001*dd(1. + rz))))/a**2, 0., 0., 0.], [0., 0., 0., 0., 0., (0.15915494309189535*((0. + 4.917450345633026*1j)*((0. + 3.141592653589793*1j)*dd(-1. + ry)*dd(rz) - (0. + 3.141592653589793*1j)*dd(1. + ry)*dd(rz)) + 30.897251760466602*dd(ry)*dd(rz)*np.sin(a*kx)))/a, 0., (0.15915494309189535*(-554.338134148627*dd(ry)*dd(rz) + 231.46065209987634*np.cos(a*kx)*dd(ry)*dd(rz) + 36.83810691296881*(3.141592653589793*dd(-1. + ry)*dd(rz) + 3.141592653589793*dd(1. + ry)*dd(rz)) + 36.47003860687391*dd(ry)*(1.2533141373155001*dd(-1. + rz) + 1.2533141373155001*dd(1. + rz))))/a**2, (-3.303066225709695*(-3.141592653589793*dd(-1. + ry)*dd(rz) + 6.283185307179586*np.cos(a*kx)*dd(ry)*dd(rz) - 3.141592653589793*dd(1. + ry)*dd(rz) + 3.141592653589793*dd(-1. + ry)*dd(rz)*np.sin(a*kx) - 3.141592653589793*dd(1. + ry)*dd(rz)*np.sin(a*kx)))/a**2, (-2.148169594243227*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz))*((0. + 1.*1j)*((0. + 1.2533141373155001*1j)*dd(-1. + ry) - (0. + 1.2533141373155001*1j)*dd(1. + ry)) + 2.5066282746310002*dd(ry)*np.sin(a*kx)))/a**2], [0., 0., 0., 0., 0.12841059146347703*dd(ry)*dd(rz), (0.15915494309189535*((0. + 4.917450345633026*1j)*((0. + 3.141592653589793*1j)*dd(-1. + ry)*dd(rz) - (0. + 3.141592653589793*1j)*dd(1. + ry)*dd(rz)) - 30.897251760466602*dd(ry)*dd(rz)*np.sin(a*kx)))/a, 0., (-3.303066225709695*(-3.141592653589793*dd(-1. + ry)*dd(rz) + 6.283185307179586*np.cos(a*kx)*dd(ry)*dd(rz) - 3.141592653589793*dd(1. + ry)*dd(rz) - 3.141592653589793*dd(-1. + ry)*dd(rz)*np.sin(a*kx) + 3.141592653589793*dd(1. + ry)*dd(rz)*np.sin(a*kx)))/a**2, (0.15915494309189535*(-554.338134148627*dd(ry)*dd(rz) - 1.1736990153811466*a**2*dd(ry)*dd(rz) + 231.46065209987634*np.cos(a*kx)*dd(ry)*dd(rz) + 36.83810691296881*(3.141592653589793*dd(-1. + ry)*dd(rz) + 3.141592653589793*dd(1. + ry)*dd(rz)) + 36.47003860687391*dd(ry)*(1.2533141373155001*dd(-1. + rz) + 1.2533141373155001*dd(1. + rz))))/a**2, (2.148169594243227*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz))*((0. - 1.*1j)*((0. + 1.2533141373155001*1j)*dd(-1. + ry) - (0. + 1.2533141373155001*1j)*dd(1. + ry)) + 2.5066282746310002*dd(ry)*np.sin(a*kx)))/a**2], [0., 0., 0., 0.12841059146347703*dd(ry)*dd(rz), 0., ((0. + 4.630696198911051*1j)*dd(ry)*dd(-1. + rz))/a - ((0. + 4.630696198911051*1j)*dd(ry)*dd(1. + rz))/a, 0., (-2.148169594243227*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz))*((0. - 1.*1j)*((0. + 1.2533141373155001*1j)*dd(-1. + ry) - (0. + 1.2533141373155001*1j)*dd(1. + ry)) + 2.5066282746310002*dd(ry)*np.sin(a*kx)))/a**2, (2.148169594243227*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz))*((0. + 1.*1j)*((0. + 1.2533141373155001*1j)*dd(-1. + ry) - (0. + 1.2533141373155001*1j)*dd(1. + ry)) + 2.5066282746310002*dd(ry)*np.sin(a*kx)))/a**2, (0.15915494309189535*(-408.4667344978722*dd(ry)*dd(rz) - 2.2757697182604457*a**2*dd(ry)*dd(rz) + 99.75835830102662*np.cos(a*kx)*dd(ry)*dd(rz) + 15.877035838340799*(3.141592653589793*dd(-1. + ry)*dd(rz) + 3.141592653589793*dd(1. + ry)*dd(rz)) + 83.35899662927821*dd(ry)*(1.2533141373155001*dd(-1. + rz) + 1.2533141373155001*dd(1. + rz))))/a**2]])


def dgso(a, kx, ry, rz):
	return np.array([[(31.677098575367697*dd(ry)*dd(rz)*np.sin(a*kx))/a, 0, -4.917450345633025*np.cos(a*kx)*dd(ry)*dd(rz), 4.917450345633025*np.cos(a*kx)*dd(ry)*dd(rz), 0, 0, 0, 0, 0, 0], [0, (72.49189501958318*dd(ry)*dd(rz)*np.sin(a*kx))/a, 0, 0, 0, 0, 0, 0, 0, 0], [-4.917450345633025*np.cos(a*kx)*dd(ry)*dd(rz), 0, (-36.838106912968804*dd(ry)*dd(rz)*np.sin(a*kx))/a, (-3.303066225709695*(-3.141592653589793*a*np.cos(a*kx)*dd(-1. + ry)*dd(rz) + 3.141592653589793*a*np.cos(a*kx)*dd(1. + ry)*dd(rz) - 6.283185307179586*a*dd(ry)*dd(rz)*np.sin(a*kx)))/a**2, (5.384662643632676*np.cos(a*kx)*dd(ry)*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz)))/a, 0, 0, 0, 0, 0], [4.917450345633025*np.cos(a*kx)*dd(ry)*dd(rz), 0, (-3.303066225709695*(3.141592653589793*a*np.cos(a*kx)*dd(-1. + ry)*dd(rz) - 3.141592653589793*a*np.cos(a*kx)*dd(1. + ry)*dd(rz) - 6.283185307179586*a*dd(ry)*dd(rz)*np.sin(a*kx)))/a**2, (-36.838106912968804*dd(ry)*dd(rz)*np.sin(a*kx))/a, (-5.384662643632676*np.cos(a*kx)*dd(ry)*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz)))/a, 0, 0, 0, 0, 0], [0, 0, (5.384662643632676*np.cos(a*kx)*dd(ry)*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz)))/a, (-5.384662643632676*np.cos(a*kx)*dd(ry)*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz)))/a, (-15.877035838340797*dd(ry)*dd(rz)*np.sin(a*kx))/a, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, (31.677098575367697*dd(ry)*dd(rz)*np.sin(a*kx))/a, 0, 4.917450345633025*np.cos(a*kx)*dd(ry)*dd(rz), -4.917450345633025*np.cos(a*kx)*dd(ry)*dd(rz), 0], [0, 0, 0, 0, 0, 0, (72.49189501958318*dd(ry)*dd(rz)*np.sin(a*kx))/a, 0, 0, 0], [0, 0, 0, 0, 0, 4.917450345633025*np.cos(a*kx)*dd(ry)*dd(rz), 0, (-36.838106912968804*dd(ry)*dd(rz)*np.sin(a*kx))/a, (-3.303066225709695*(3.141592653589793*a*np.cos(a*kx)*dd(-1. + ry)*dd(rz) - 3.141592653589793*a*np.cos(a*kx)*dd(1. + ry)*dd(rz) - 6.283185307179586*a*dd(ry)*dd(rz)*np.sin(a*kx)))/a**2, (-5.384662643632676*np.cos(a*kx)*dd(ry)*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz)))/a], [0, 0, 0, 0, 0, -4.917450345633025*np.cos(a*kx)*dd(ry)*dd(rz), 0, (-3.303066225709695*(-3.141592653589793*a*np.cos(a*kx)*dd(-1. + ry)*dd(rz) + 3.141592653589793*a*np.cos(a*kx)*dd(1. + ry)*dd(rz) - 6.283185307179586*a*dd(ry)*dd(rz)*np.sin(a*kx)))/a**2, (-36.838106912968804*dd(ry)*dd(rz)*np.sin(a*kx))/a, (5.384662643632676*np.cos(a*kx)*dd(ry)*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz)))/a], [0, 0, 0, 0, 0, 0, 0, (-5.384662643632676*np.cos(a*kx)*dd(ry)*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz)))/a, (5.384662643632676*np.cos(a*kx)*dd(ry)*((0. + 1.2533141373155001*1j)*dd(-1. + rz) - (0. + 1.2533141373155001*1j)*dd(1. + rz)))/a, (-15.877035838340797*dd(ry)*dd(rz)*np.sin(a*kx))/a]])

# print(dgso(10, 0, 0, 0))
# ---------------------------------------------
#               Create the S matrix
# ---------------------------------------------


def spin_matrix_small():
    dim = 5
    zero5 = np.zeros((dim, dim), dtype=complex)

    # --- sx ---
    sud = zero5.copy()
    sud[0,0] = 1
    sud[1,1] = 1
    sud[4,4] = 1
    sud[2,3] = 1
    sud[3,2] = 1

    sx = 0.5 * np.block([
        [zero5, sud],
        [sud.conj(), zero5]
    ])

    # --- sy ---
    sud = -1j * sud

    sy = 0.5 * np.block([
        [zero5, sud],
        [sud.conj(), zero5]
    ])

    # --- sz ---
    sz = 0.5 * np.block([
        [ np.eye(dim),  zero5],
        [ zero5,       -np.eye(dim)]
    ])

    return sx, sy, sz



# def spin_matrix(NN):
#     dim = 5
#     zero5 = np.zeros((dim, dim), dtype=complex)
    
#     # --- sx matrix ---
#     szup = zero5.copy()
#     szdn = -szup
    
#     sud = zero5.copy()
#     sud[0, 0] = 1
#     sud[1, 1] = 1
#     sud[4, 4] = 1
#     sud[2, 3] = 1
#     sud[3, 2] = 1
    
#     sdu = np.conjugate(sud)
#     sx = 0.5 * np.block([[szup, sud],
#                          [sdu, szdn]])
    
#     # --- sy matrix ---
#     szup = zero5.copy()
#     szdn = -szup
    
#     sud = zero5.copy()
#     sud[0, 0] = -1j
#     sud[1, 1] = -1j
#     sud[4, 4] = -1j
#     sud[2, 3] = -1j
#     sud[3, 2] = -1j
    
#     sdu = np.conjugate(sud)
#     sy = 0.5 * np.block([[szup, sud],
#                          [sdu, szdn]])
    
#     # --- sz matrix ---
#     szup = np.eye(dim, dtype=complex)
#     szdn = -szup
    
#     sud = zero5.copy()
#     sdu = np.conjugate(sud)
#     sz = 0.5 * np.block([[szup, sud],
#                          [sdu, szdn]])
    
#     # --- Expand to full matrices ---
#     sxfull = np.kron(np.eye(NN , dtype=complex), sx)
#     syfull = np.kron(np.eye(NN , dtype=complex), sy)
#     szfull = np.kron(np.eye(NN , dtype=complex), sz)
#     return sxfull, syfull, szfull



#-----------------------------------------------
# basis rotation
#-----------------------------------------------
# =================================================================
# 3. BASIS TRANSFORMATION (Spin Alignment)
# =================================================================

def apply_spin_operator(psi, s):
    """
    Apply I⊗s without constructing kron matrix.

    psi shape = (NN*10,)
    s shape   = (10,10)
    """

    NN = psi.size // 10

    psi_rs = psi.reshape(NN, 10)

    # apply spin matrix on internal index
    out = psi_rs @ s.T

    return out.reshape(-1)


def psi_new_basis(p1, p2, N):

    sx, sy, sz = spin_matrix_small()

    # apply operators efficiently
    sz_p1 = apply_spin_operator(p1, sz)
    sz_p2 = apply_spin_operator(p2, sz)

    # expectation matrix
    sz11 = np.vdot(p1, sz_p1)
    sz22 = np.vdot(p2, sz_p2)
    sz12 = np.vdot(p1, sz_p2)
    sz21 = np.vdot(p2, sz_p1)

    sz_exp_matrix = np.array([
        [sz11, sz12],
        [sz21, sz22]
    ])

    # rotate Kramer pair
    psi_stack = np.column_stack((p1, p2))

    _, v_rot = eigh(sz_exp_matrix)

    psi_rotated = psi_stack @ v_rot

    psi_ortho, _ = qr(psi_rotated, mode="economic")

    p1_new = psi_ortho[:,0]
    p2_new = psi_ortho[:,1]

    # phase fixing using Sx
    sx_p2 = apply_spin_operator(p2_new, sx)

    sx12 = np.vdot(p1_new, sx_p2)

    rel_phase = np.exp(-1j * np.angle(sx12))

    p2_new *= rel_phase

    return p1_new, p2_new


# def psi_new_basis(p1, p2, N):
#     """
#     Transforms p1 and p2 into a basis where the Sz matrix is diagonal.
#     This effectively aligns the states with Up/Down spin projections.
#     """
#     # Retrieve Spin matrices (Sx and Sz)
#     sx_full = spin_matrix(N**2)[0]
#     sz_full = spin_matrix(N**2)[2]

#     # Calculate expectation values of Sz in the original subspace
#     sz11 = np.vdot(p1, sz_full @ p1)
#     sz22 = np.vdot(p2, sz_full @ p2)
#     sz12 = np.vdot(p1, sz_full @ p2)
#     sz21 = np.vdot(p2, sz_full @ p1)

#     sz_exp_matrix = np.array([[sz11, sz12], [sz21, sz22]])

#     # Stack original vectors: Shape (D, 2)
#     psi_stack = np.column_stack((p1, p2)) 

#     # Find the rotation matrix V that diagonalizes the 2x2 Sz matrix
#     _, v_rot = eigh(sz_exp_matrix)

#     # Rotate into the new basis and re-orthogonalize
#     psi_rotated = psi_stack @ v_rot
#     psi_ortho, _ = qr(psi_rotated, mode="economic")

#     p1_new = psi_ortho[:, 0]
#     p2_new = psi_ortho[:, 1]

#     # Adjust relative phase using Sx to ensure consistency
#     sx12 = np.vdot(p1_new, sx_full @ p2_new)
#     rel_phase = np.exp(-1j * np.angle(sx12))
#     p2_new *= rel_phase 
    
#     return p1_new, p2_new
