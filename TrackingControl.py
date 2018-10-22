import numpy as np
from types import MethodType, FunctionType
from wrapper import *

"""
This code creates the molecular spectral profiles to be 
discriminated using principles of Tracking Control.
--------------------------------------------------------
Authors: Alicia Magann, Denys Bondar, Ayan Chattopadhyay
--------------------------------------------------------
"""


class ADict(dict):
    """
    Dictionary where you can access keys as attributes
    """
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            dict.__getattribute__(self, item)


class TrackingControl:
    """
    Class variable containing:
    1. Molecular and Control Parameters
    2. Spectra Calculation Functions
    3. Tracking Control Functions
    """

    def __init__(self, params, **kwargs):
        """
        __init__ function call to initialize variables from the
        parameters for the class instance provided in __main__ and
        add new variables for use in other functions in this class.

        DO NOT READ TOO MUCH INTO THIS NOW: JUST UNDERSTAND THAT THIS
        CREATES CLASS VARIABLES AUTOMATICALLY FROM THE DICTIONARY OF
        PARAMETERS PROVIDED IN THE PARAMETERS IN "__main__" (OUTSIDE
        THIS CLASS WHERE THE CLASS INSTANCE IS CALLED)
        """

        for name, value in kwargs.items():
            if isinstance(value, FunctionType):
                setattr(self, name, MethodType(value, self, self.__class__))
            else:
                setattr(self, name, value)

        self.time_AE = np.linspace(-params.timeAMP_AE, params.timeAMP_AE, params.timeDIM_AE)

        self.frequency_A = 1./np.linspace(1./params.frequencyMAX_A, 1./params.frequencyMIN_A, params.frequencyDIM_AE)
        self.frequency_E = 1./np.linspace(1./params.frequencyMAX_E, 1./params.frequencyMIN_E, params.frequencyDIM_AE)

        self.field_A = np.empty(params.timeDIM_AE, dtype=np.complex)
        self.field_E = np.empty(params.timeDIM_AE, dtype=np.complex)

        self.gamma_decay = np.ascontiguousarray(self.gamma_decay)
        self.gamma_dephasingA = np.ascontiguousarray(self.gamma_dephasingA)
        self.gamma_dephasingB = np.ascontiguousarray(self.gamma_dephasingB)
        self.mu = np.ascontiguousarray(self.mu)
        self.rho_0 = np.ascontiguousarray(params.rho_0_A)
        self.rhoA = np.ascontiguousarray(params.rho_0_A.copy())
        self.rhoB = np.ascontiguousarray(params.rho_0_A.copy())
        self.energies_A = np.ascontiguousarray(self.energies_A)
        self.energies_B = np.ascontiguousarray(self.energies_B)

        N = len(self.energies_A)

        self.abs_spectraA = np.ascontiguousarray(np.zeros(len(self.frequency_A)))
        self.abs_spectraB = np.ascontiguousarray(np.zeros(len(self.frequency_A)))
        self.ems_spectraA = np.ascontiguousarray(np.zeros(len(self.frequency_E)))
        self.ems_spectraB = np.ascontiguousarray(np.zeros(len(self.frequency_E)))

    def create_molecules(self, molA, molB):
        molA.nDIM = len(self.energies_A)
        molA.energies = self.energies_A.ctypes.data_as(POINTER(c_double))
        molA.gamma_decay = self.gamma_decay.ctypes.data_as(POINTER(c_double))
        molA.gamma_dephasing = self.gamma_dephasingA.ctypes.data_as(POINTER(c_double))
        molA.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        molA.rho = self.rhoA.ctypes.data_as(POINTER(c_complex))
        molA.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        molA.abs_spectra = self.abs_spectraA.ctypes.data_as(POINTER(c_double))
        molA.ems_spectra = self.ems_spectraA.ctypes.data_as(POINTER(c_double))

        molB.nDIM = len(self.energies_B)
        molB.energies = self.energies_B.ctypes.data_as(POINTER(c_double))
        molB.gamma_decay = self.gamma_decay.ctypes.data_as(POINTER(c_double))
        molB.gamma_dephasing = self.gamma_dephasingB.ctypes.data_as(POINTER(c_double))
        molB.mu = self.mu.ctypes.data_as(POINTER(c_complex))
        molB.rho = self.rhoB.ctypes.data_as(POINTER(c_complex))
        molB.rho_0 = self.rho_0.ctypes.data_as(POINTER(c_complex))
        molB.abs_spectra = self.abs_spectraB.ctypes.data_as(POINTER(c_double))
        molB.ems_spectra = self.ems_spectraB.ctypes.data_as(POINTER(c_double))

    def create_parameters_spectra(self, spectra_params, params):
        spectra_params.rho_0_A = params.rho_0_A.ctypes.data_as(POINTER(c_complex))
        spectra_params.rho_0_E = params.rho_0_E.ctypes.data_as(POINTER(c_complex))
        spectra_params.time_AE = self.time_AE.ctypes.data_as(POINTER(c_double))
        spectra_params.frequency_A = self.frequency_A.ctypes.data_as(POINTER(c_double))
        spectra_params.frequency_E = self.frequency_E.ctypes.data_as(POINTER(c_double))

        spectra_params.field_amp_AE = params.field_amp_AE

        spectra_params.nDIM = len(self.energies_A)
        spectra_params.nEXC = params.nEXC

        spectra_params.timeDIM_AE = len(self.time_AE)

        spectra_params.freqDIM_A = len(self.frequency_A)
        spectra_params.freqDIM_E = len(self.frequency_E)

        spectra_params.field_A = self.field_A.ctypes.data_as(POINTER(c_complex))
        spectra_params.field_E = self.field_E.ctypes.data_as(POINTER(c_complex))

    def calculate_spectra(self, params):
        molA = Molecule()
        molB = Molecule()
        self.create_molecules(molA, molB)
        params_spectra = Parameters_Spectra()
        self.create_parameters_spectra(params_spectra, params)
        CalculateSpectra(molA, molB, params_spectra)
        return


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import time

    np.set_printoptions(precision=4)
    energy_factor = 1. / 27.211385
    time_factor = .02418884 / 1000

    energies_A = np.array((0.000, 0.08233, 0.09832, 0.16304, 0.20209, 1.7679256, 1.85871, 1.87855, 1.96783, 2.02991)) * energy_factor
    energies_B = np.array((0.000, 0.08313, 0.09931, 0.17907, 0.21924, 1.7712000, 1.86871, 1.88855, 1.97783, 2.12991)) * energy_factor

    N = len(energies_A)
    N_vib = N - 5
    N_exc = N - N_vib
    rho_0_ems = np.zeros((N, N), dtype=np.complex)
    rho_0_ems[N_vib, N_vib] = 1. + 0j
    rho_0_abs = np.zeros((N, N), dtype=np.complex)
    rho_0_abs[0, 0] = 1. + 0j

    mu = 4.97738 * np.ones_like(rho_0_abs)
    np.fill_diagonal(mu, 0j)
    population_decay = 2.418884e-8
    electronic_dephasingA = 2.7 * 2.418884e-4
    electronic_dephasingB = 4.0 * 2.418884e-4
    vibrational_dephasing = 0.2 * 2.418884e-5

    gamma_decay = np.ones((N, N)) * population_decay
    np.fill_diagonal(gamma_decay, 0.0)
    gamma_decay = np.tril(gamma_decay)

    gamma_dephasingA = np.ones_like(gamma_decay) * vibrational_dephasing
    gamma_dephasingB = np.ones_like(gamma_decay) * vibrational_dephasing
    np.fill_diagonal(gamma_dephasingA, 0.0)

    """
    MOLECULE A gamma_dephasing parameters: The numbers are scalings of
    the dephasing parameter to account for specific line-widths of each 
    corresponding transition. The widths of the lines near absorption
    peak are narrow and tall and near the hump it is short and wide.
    """

    for i in range(N_vib):
        for j in range(N_vib, N):
            gamma_dephasingA[i, j] = electronic_dephasingA
            gamma_dephasingA[j, i] = electronic_dephasingA

    gamma_dephasingA[5, 4] = electronic_dephasingA * 0.65
    gamma_dephasingA[4, 5] = electronic_dephasingA * 0.65
    gamma_dephasingA[0, 9] = electronic_dephasingA * 0.65
    gamma_dephasingA[9, 0] = electronic_dephasingA * 0.65

    gamma_dephasingA[5, 3] = electronic_dephasingA * 0.70
    gamma_dephasingA[3, 5] = electronic_dephasingA * 0.70
    gamma_dephasingA[0, 8] = electronic_dephasingA * 0.70
    gamma_dephasingA[8, 0] = electronic_dephasingA * 0.70

    gamma_dephasingA[5, 2] = electronic_dephasingA * 0.20
    gamma_dephasingA[2, 5] = electronic_dephasingA * 0.20
    gamma_dephasingA[0, 7] = electronic_dephasingA * 0.20
    gamma_dephasingA[7, 0] = electronic_dephasingA * 0.20

    gamma_dephasingA[5, 1] = electronic_dephasingA * 0.18
    gamma_dephasingA[1, 5] = electronic_dephasingA * 0.18
    gamma_dephasingA[0, 6] = electronic_dephasingA * 0.18
    gamma_dephasingA[6, 0] = electronic_dephasingA * 0.18

    gamma_dephasingA[5, 0] = electronic_dephasingA * 0.60
    gamma_dephasingA[0, 5] = electronic_dephasingA * 0.60
    mu[5, 0] *= 0.10
    mu[0, 5] *= 0.10

    """
    MOLECULE B gamma_dephasing parameters. 
    """

    for i in range(N_vib):
        for j in range(N_vib, N):
            gamma_dephasingB[i, j] = electronic_dephasingB
            gamma_dephasingB[j, i] = electronic_dephasingB

    gamma_dephasingB = np.ones_like(gamma_decay) * vibrational_dephasing
    np.fill_diagonal(gamma_dephasingB, 0.0)
    gamma_dephasingB[5, 4] = electronic_dephasingB * 0.65
    gamma_dephasingB[4, 5] = electronic_dephasingB * 0.65
    gamma_dephasingB[0, 9] = electronic_dephasingB * 0.65
    gamma_dephasingB[9, 0] = electronic_dephasingB * 0.65

    gamma_dephasingB[5, 3] = electronic_dephasingB * 0.70
    gamma_dephasingB[3, 5] = electronic_dephasingB * 0.70
    gamma_dephasingB[0, 8] = electronic_dephasingB * 0.70
    gamma_dephasingB[8, 0] = electronic_dephasingB * 0.70

    gamma_dephasingB[5, 2] = electronic_dephasingB * 0.20
    gamma_dephasingB[2, 5] = electronic_dephasingB * 0.20
    gamma_dephasingB[0, 7] = electronic_dephasingB * 0.20
    gamma_dephasingB[7, 0] = electronic_dephasingB * 0.20

    gamma_dephasingB[5, 1] = electronic_dephasingB * 0.18
    gamma_dephasingB[1, 5] = electronic_dephasingB * 0.18
    gamma_dephasingB[0, 6] = electronic_dephasingB * 0.18
    gamma_dephasingB[6, 0] = electronic_dephasingB * 0.18

    gamma_dephasingB[5, 0] = electronic_dephasingB * 0.60
    gamma_dephasingB[0, 5] = electronic_dephasingB * 0.60

    np.set_printoptions(precision=2)

    params = ADict(
        energy_factor=energy_factor,
        time_factor=time_factor,
        rho_0_A=rho_0_abs,
        rho_0_E=rho_0_ems,

        timeDIM_AE=1000,
        timeAMP_AE=2000,

        frequencyDIM_AE=250,

        frequencyMIN_A=1.5 * energy_factor,
        frequencyMAX_A=2.7 * energy_factor,

        frequencyMIN_E=1.2 * energy_factor,
        frequencyMAX_E=2.3 * energy_factor,

        field_amp_AE=0.000003,

        nEXC=N_exc
    )

    FourLevels = dict(
        energies_A=energies_A,
        energies_B=energies_B,
        gamma_decay=gamma_decay,
        gamma_dephasingA=gamma_dephasingA,
        gamma_dephasingB=gamma_dephasingB,
        mu=mu,
    )

    """
    PLOTTING THE MOLECULAR ABSORPTION SPECTRA
    """

    def render_ticks(axes):
        axes.get_xaxis().set_tick_params(which='both', direction='in', width=1, labelrotation=0, labelsize='large')
        axes.get_yaxis().set_tick_params(which='both', direction='in', width=1, labelcolor='r', labelsize='large')
        axes.get_xaxis().set_ticks_position('both')
        axes.get_yaxis().set_ticks_position('both')
        axes.grid()

    start = time.time()
    molecules = TrackingControl(params, **FourLevels)
    molecules.calculate_spectra(params)

    end_spectra = time.time()
    print("Time to calculate spectra: ", end_spectra - start)
    print()

    fig, axes = plt.subplots(nrows=2, ncols=1)

    axes[0].set_title(
        'Field for Electronic Spectra \n $\\tau_E$= {} fs'
            .format(int(1e3*time_factor/electronic_dephasingA)))
    axes[0].plot(molecules.time_AE * time_factor, molecules.field_A.real, 'r')
    axes[0].plot(molecules.time_AE * time_factor, molecules.field_E.real, 'b')
    render_ticks(axes[0])
    axes[0].set_xlabel("Time (in ps)")

    axes[1].set_title("Absorption & Emission spectra")
    axes[1].plot(1239.84 / (molecules.frequency_A / energy_factor), molecules.abs_spectraA, 'r')
    axes[1].plot(1239.84 / (molecules.frequency_E / energy_factor), molecules.ems_spectraA, 'b')
    axes[1].set_xlim(450., 850.)

    axes[0].set_xlabel("Time (in ps)")
    axes[0].yaxis.tick_left()
    axes[0].yaxis.set_label_position("left")
    axes[1].yaxis.tick_right()
    axes[1].yaxis.set_label_position("right")

    axes[0].ticklabel_format(scilimits=(-2, 2))
    render_ticks(axes[1])

    fig.subplots_adjust(left=0.32, bottom=None, right=0.68, top=0.8, wspace=0.025, hspace=0.55)

    plt.show()