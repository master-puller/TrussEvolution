import numpy as np
import matplotlib.pyplot as plt

E = 1e4
A = 0.111
nodes = []
bars = []

nodes.append([0, 0, 0])  # Repeat for the rest
nodes.append([0, 100, 0])

bars.append([0, 1])  # Repeat

nodes = np.array(nodes).astype(float)
bars = np.array(bars)  # Convert to numpy for faster processing

# Applied forces
P = np.zeros_like(nodes)
P[1, 1] = -10  # idx 1 is y direction

# Support displacement
Ur = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

DOFCON = np.ones_like(nodes).astype(int)
# Constraints of DOF
DOFCON[0, :] = 0  # Fix node 0


def TrussAnalysis():
    NN = len(nodes)
    NE = len(bars)

    DOF = 3
    NDOF = DOF * NN

    # Start structurl analasys
    d = nodes[bars[:, 1], :] - nodes[bars[:, 0], :]
    L = np.sqrt((d**2).sum(axis=1))
    angle = d.t / L

    a = np.concatenate((-angle.T, angle.T), axis=1)
    K = np.zeros([NDOF, NDOF])  # actual matrix we will be solving
    for k in range(NE):  # bar iteration
        aux = 3 * bars[k, :]
        index = np.r_[aux[0]:aux[0] + DOF, aux[1]:aux[1] + DOF]
        ES = np.dot(a[k][np.newaxis].T * E * A, a[k][np.newaxis]) / L[k]

        K[np.ix_(index, index)] = K[np.ix_(index, index)] + ES

    freeDOF = DOFCON.flatten().nonzero()[0]
    supportDOF = (DOFCON.flatten() == 0).nonzero()[0]

    Kff = K[np.ix_(freeDOF, freeDOF)]
    Kfr = K[np.ix_(freeDOF, supportDOF)]
    Krf = Kfr.T
    Krr = K[np.ix_(supportDOF, supportDOF)]
    Pf = P.flatten()[freeDOF]
    Uf = np.linalg.solve(Kff, Pf)
    U = DOFCON.astype(float).flatten()
    U[freeDOF] = Uf
    U[supportDOF] = Ur
    U = U.reshape(NN, DOF)

    u = np.concatenate((U[bars[:, 0]], U[bars[:, 1]]), axis=1)

    N = E * A / L[:] * (a[:] * u[:]).sum(
        axis=1)  # Structural forces of all the bars

    R = (Krf[:] * Uf).sum(axis=1) + (Krr[:] * Ur).sum(axis=1)
    R = R.reshape(4, DOF)

    return np.array(N), np.array(R), U


def Plot(nodes, c, lt, lw, lg):
    plt.gca(projection="3d")
    for i in range(len(bars)):
        xi, xf = nodes[bars[i, 0], 0], nodes[bars[:, 1], 0]
        yi, yf = nodes[bars[i, 0], 1], nodes[bars[:, 1], 1]
        zi, zf = nodes[bars[i, 0], 2], nodes[bars[:, 1], 2]
        line = plt.plot([xi, xf], [yi, yf], [zi, zf],
                        color=c,
                        linestyle=lt,
                        linewidth=lw)
    line.set_label(lg)
    plt.legend(prop={'size': 14})
