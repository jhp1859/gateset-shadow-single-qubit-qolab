import numpy as np
from tqdm import tqdm

def b_to_depth_seq_dicts_labels(b: np.ndarray, depths):
    """
    b: (K, Mmax) int array. Row=k sequence index. Col=i gate position.
    depths: iterable of m values.
    returns: out[depth_index][seq_index] = {"0": label0, ..., "m-1": label_{m-1}}
    """
    K, Mmax = b.shape
    depths = list(depths)

    out = []
    for m in depths:
        if m > Mmax:
            raise ValueError(f"m={m} > Mmax={Mmax}")

        depth_row = []
        for k in range(K):
            labels = b[k, :m]
            d = {str(i): int(labels[i]) for i in range(m)}
            depth_row.append(d)

        out.append(depth_row)

    return out


def b_to_depth_seq_dicts_unitaries(b: np.ndarray, depths, U_lookup):
    K, Mmax = b.shape
    depths = list(depths)

    out = []
    for m in tqdm(depths):
        if m > Mmax:
            raise ValueError(f"m={m} > Mmax={Mmax}")

        depth_row = []
        for k in tqdm(range(K)):
            labels = b[k, :m]
            # assert all labels exist in lookup
            missing = [int(lbl) for lbl in labels if int(lbl) not in U_lookup]
            if missing:
                raise KeyError(f"Missing Clifford labels in U_lookup: {sorted(set(missing))}")

            d = {str(i): U_lookup[int(labels[i])] for i in range(m)}
            depth_row.append(d)

        out.append(depth_row)

    return out



# --- Pauli matrices ---
I2 = np.array([[1, 0], [0, 1]], dtype=complex)
X  = np.array([[0, 1], [1, 0]], dtype=complex)
Y  = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z  = np.array([[1, 0], [0, -1]], dtype=complex)


# --- SU(2) rotations from Pauli generators ---
def U_rot(P: np.ndarray, theta: float) -> np.ndarray:
    """
    Single-qubit rotation: exp(-i * theta/2 * P), with P^2 = I.
    """
    return np.cos(theta/2) * I2 - 1j * np.sin(theta/2) * P


# elementary x/y rotations
Ux90  = U_rot(X,  np.pi/2)
U_x90 = U_rot(X, -np.pi/2)
Ux180 = U_rot(X,  np.pi)

Uy90  = U_rot(Y,  np.pi/2)
U_y90 = U_rot(Y, -np.pi/2)
Uy180 = U_rot(Y,  np.pi)

# --- Clifford table: label -> unitary ---
U_clifford = {}

U_clifford[0]  = I2
U_clifford[1]  = Ux180
U_clifford[2]  = Uy180
U_clifford[3]  = Ux180 @ Uy180

U_clifford[4]  = Uy90  @ Ux90
U_clifford[5]  = U_y90 @ Ux90
U_clifford[6]  = Uy90  @ U_x90
U_clifford[7]  = U_y90 @ U_x90

U_clifford[8]  = Ux90  @ Uy90
U_clifford[9]  = U_x90 @ Uy90
U_clifford[10] = Ux90  @ U_y90
U_clifford[11] = U_x90 @ U_y90

U_clifford[12] = Ux90
U_clifford[13] = U_x90
U_clifford[14] = Uy90
U_clifford[15] = U_y90

U_clifford[16] = Ux90  @ Uy90  @ U_x90
U_clifford[17] = Ux90  @ U_y90 @ U_x90
U_clifford[18] = Uy90  @ Ux180
U_clifford[19] = U_y90 @ Ux180
U_clifford[20] = Ux90  @ Uy180
U_clifford[21] = U_x90 @ Uy180
U_clifford[22] = Ux90  @ Uy90  @ Ux90
U_clifford[23] = U_x90 @ Uy90  @ U_x90
