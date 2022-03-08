from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

from Beam import Beam, Beam_Properties, Material
from Vector import Vector3
from Arrow import Arrow3D


class Truss:

    def __init__(self):
        self.nodes = []
        self.beams = []
        self.anchors = []
        self.loads = []

    @property
    def num_anchors(self):
        return np.count_nonzero(self.anchors, 0) > 0

    def add_points(self, points: Iterable[Vector3]) -> None:
        self.nodes.extend(points)
        x = np.zeros_like(points)
        self.anchors.extend(1 - x)
        self.loads.extend(x)

    def add_beams(self, beams: Iterable[Beam]) -> None:
        self.beams.extend(beams)
        for beam in beams:
            beam.truss = self

    def add_anchor(self, point: Vector3, restriction: Vector3) -> None:
        self.nodes.append(point)
        self.anchors.append(restriction)
        self.loads.append(Vector3.zero)

    def add_loads(self, loads: Dict[int, Vector3]):
        for (k, v) in loads.items():
            self.loads[k] = v

    def set_anchor(self, idx: int, restriction: Vector3) -> None:
        self.anchors[idx] = restriction

    def solve(self):
        DOF = 3
        NDOF = DOF * len(self.nodes)
        beams = np.array(self.beams)
        nodes = np.array(self.nodes)
        anchors = np.array(self.anchors).flatten()

        isanchor = (np.count_nonzero(self.anchors, axis=1) != 3)

        Ur = np.array([
            self.anchors[i] for i in range(len(self.anchors)) if isanchor[i]
        ]).flatten()
        # 1 is movable

        displacement = nodes[beams[:, 1], :] - nodes[beams[:, 0], :]
        L = np.sqrt((displacement**2).sum(axis=1))
        theta = displacement.T / L

        a = np.concatenate((-theta.T, theta.T), axis=1)
        K = np.zeros([NDOF, NDOF])  # actual matrix we will be solving
        for k in range(len(self.beams)):  # bar iteration
            aux = 3 * beams[k, :2]
            index = np.r_[aux[0]:aux[0] + DOF, aux[1]:aux[1] + DOF]
            ES = np.dot(a[k][np.newaxis].T * self.beams[k].spring_const,
                        a[k][np.newaxis]) / L[k]
            # Elastic constant

            K[np.ix_(index, index)] = K[np.ix_(index, index)] + ES

        freeDOF = anchors.nonzero()[0]
        supportDOF = (anchors == 0).nonzero()[0]

        Kff = K[np.ix_(freeDOF, freeDOF)]
        Kfr = K[np.ix_(freeDOF, supportDOF)]
        Krf = Kfr.T
        Krr = K[np.ix_(supportDOF, supportDOF)]

        P = np.array(self.loads)

        Pf = P.flatten()[freeDOF]
        Displacement_Free = np.linalg.solve(Kff, Pf)
        Displacement = anchors.astype(float)
        Displacement[freeDOF] = Displacement_Free
        Displacement[supportDOF] = Ur
        Displacement = Displacement.reshape(len(self.nodes), DOF)

        u = np.concatenate(
            (Displacement[beams[:, 0]], Displacement[beams[:, 1]]), axis=1)

        # E * A is different for many bars, use indexing to get attr
        Normal = np.array([beam.spring_const
                           for beam in self.beams]) / L[:] * (a[:] * u[:]).sum(
                               axis=1)  # Structural forces of all the bars

        Resistance = ((Krf[:] * Displacement_Free).sum(axis=1) +
                      (Krr[:] * Ur).sum(axis=1))
        Resistance = Resistance.reshape(np.sum(isanchor), DOF)

        return Normal, Resistance, Displacement

    def plot(self, color, linewidth, linestyle="-"):
        plt.gca(projection="3d")
        for beam in self.beams:
            (xi, yi, zi) = self.nodes[beam[0]]
            (xj, yj, zj) = self.nodes[beam[1]]
            line = plt.plot([xi, xj], [yi, yj], [zi, zj],
                            color=color,
                            linestyle=linestyle,
                            linewidth=linewidth)

        for idx in range(len(self.nodes)):
            if sum(self.anchors[idx]) != 3:
                Truss.draw_anchor(self.nodes[idx])

        # line.set_label(lg)
        plt.legend(prop={'size': 14})

    @staticmethod
    def draw_anchor(pos, size=10, color='black'):
        x, y, z = pos
        plt.plot([x, x - size], [y, y - size], [z, z - size], color=color)
        plt.plot([x, x + size], [y, y - size], [z, z - size], color=color)
        plt.plot([x, x + size], [y, y + size], [z, z - size], color=color)
        plt.plot([x, x - size], [y, y + size], [z, z - size], color=color)
        plt.plot([x + size, x - size], [y - size] * 2, [z - size], color=color)
        plt.plot([x + size, x - size], [y + size] * 2, [z - size], color=color)
        plt.plot([x + size] * 2, [y + size, y - size], [z - size], color=color)
        plt.plot([x - size] * 2, [y + size, y - size], [z - size], color=color)

    @staticmethod
    def draw_force(pos, force, scale):
        new = pos + force
        Arrow3D(*zip(pos, new),
                mutation_scale=20,
                lw=3,
                arrowstyle="-|>",
                color="r")


if __name__ == "__main__":
    # Do tests
    truss = Truss()
    truss.add_points(
        ((-37.5, 0, 200), (37.5, 0, 200), (-37.5, 37.5, 100),
         (37.5, 37.5, 100), (37.5, -37.5, 100), (-37.5, -37.5, 100)))
    truss.add_anchor((-100, 100, 0), (0, 0, 0))  # Four anchors
    truss.add_anchor((100, 100, 0), (0, 0, 0))
    truss.add_anchor((100, -100, 0), (0, 0, 0))
    truss.add_anchor((-100, -100, 0), (0, 0, 0))

    balsa = Material(4.4E5, 2E3, 1E3)
    beam_profile = Beam_Properties(balsa, 1, 1 / 12)

    bars = Beam.from_array(
        (0, 1),  # 1
        (0, 3),
        (1, 2),
        (0, 4),
        (1, 5),  # 5
        (1, 3),
        (1, 4),
        (0, 2),
        (0, 5),
        (2, 5),
        (5, 6),
        (3, 4),
        (2, 3),
        (4, 5),
        (2, 9),
        (3, 8),
        (4, 7),
        (3, 6),
        (2, 7),
        (4, 9),
        (5, 8),
        (5, 9),
        (2, 6),
        (3, 7),
        (4, 8)  # 25
    )

    beam_profile.set_beam_properties(bars)
    truss.add_beams(bars)
    truss.add_loads({0: (0, 100, -10000), 1: (0, -100, -10000)})

    truss.plot('gray', 1, linestyle='--')
    stresses, resistance, deformations = truss.solve()
    scale = 1
    truss.nodes += deformations * scale
    truss.plot('red', 1)
    plt.show()
