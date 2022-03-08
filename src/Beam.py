import numpy as np

sqrt2 = np.sqrt(2)


class Material:

    def __init__(self,
                 modulus: np.float,
                 tensile_yield: np.float,
                 compressive_strength: np.float,
                 cost: np.float = 0,
                 density: np.float = 1):
        self.modulus = modulus
        self.tensile_yield = tensile_yield
        self.compressive_strength = compressive_strength
        self.cost = cost
        self.density = density


class Beam(np.ndarray):
    '''Class representing a link from two points'''

    def __new__(cls, *args):
        if len(args) == 1:
            if isinstance(args[0], Beam):
                return args[0].copy()
            if isinstance(args[0], np.matrix):
                return Beam(args[0].flatten().tolist()[0])
        arr = np.array(np.pad(args, (0, 3 - len(args)))[:2],
                       dtype=np.int,
                       copy=True)
        return np.ndarray.__new__(cls, shape=(2, ), buffer=arr, dtype=int)

    def __repr__(self):
        return 'Bar' + repr(tuple(self))

    def __getattr__(self, idx):
        if isinstance(idx, int):
            if idx == 2:
                return self.tensile_yield
            return self[idx]
        else:
            return self.__dict__[idx]

    def __setattr__(self, idx, val):
        if isinstance(idx, int):
            if idx == 2:
                return
            self[idx] = val
        else:
            self.__dict__[idx] = val

    def set_properties(self, properties):
        self.prop = properties

    def __eq__(self, other):
        return self[0] not in other[:] or self[1] not in other[:]

    def __hash__(self):
        return hash(tuple(self))

    @staticmethod
    def _buckle(length, k, least_radius, modulus_elasticity, tensile_yield):
        # Buckling load:
        # P_c = (pi^2*E*I)/(K * L^2)
        # https://en.wikipedia.org/wiki/File:ColumnEffectiveLength.png

        # critical slenderness ratio is
        # sqrt(2pi^2 E/O_y) # s is critical stress
        # Above this, the euler formula should be used
        # below, johnson formula
        # O_y is yield strength

        # O_r = O_y - 1/E * (O_y/2pi)^2 * (l/k)^2
        # Buckling strength lambda
        slenderness_ratio = length / least_radius
        if slenderness_ratio <= sqrt2 * np.pi / np.sqrt(
                tensile_yield
        ):  # Calculate critical ratio: which formula to use
            # Johnson formula
            return tensile_yield - 1 / modulus_elasticity * (
                modulus_elasticity / (2 * np.pi))**2 * (length / k)**2
        # Euler formula
        return (np.pi**2 * modulus_elasticity) / (length / k)**2

    @staticmethod
    def from_array(*array):
        return [Beam(r) for r in array]

    @property
    def length(self):
        return self.truss.points[self[0]] - self.truss.points[self[1]]

    @property
    def buckle_force(self):
        return Beam._buckle(self.length, 1, self.prop.least_radius,
                            self.prop.mat.modulus,
                            self.prop.area * self.prop.mat.tensile_yield)

    @property
    def spring_const(self):
        return self.prop.spring_const

    @property
    def volume(self):
        return self.length * self.prop.area

    def does_break(self, load):
        return load > self.tensile_yield or -load > self.buckle_force


class Beam_Properties:
    '''Structural class for a Bar'''

    def __init__(self,
                 material: Material,
                 area: np.float,
                 inertia: np.float,
                 mass=0,
                 breakable=True):
        self.mat = material
        self.area = area
        self.inertia = inertia
        self.mass = mass
        self.breakable = breakable

    @property
    def least_radius(self):
        return np.sqrt(self.inertia / self.area)

    @property
    def spring_const(self):
        return self.area * self.mat.modulus

    def set_beam_properties(self, beams):
        for beam in beams:
            beam.set_properties(self)
