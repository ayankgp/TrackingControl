import os
import ctypes
from ctypes import c_int, c_double, c_char_p, POINTER, Structure


__doc__ = """
Python wrapper for Spectra.c
Compile with:
gcc -O3 -shared -o Spectra.so Spectra.c -lm -fopenmp -fPIC
"""


class c_complex(Structure):
    """
    Complex double ctypes
    """
    _fields_ = [
        ('real', c_double),
        ('imag', c_double)
    ]


class Parameters_Spectra(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('rho_0_A', POINTER(c_complex)),
        ('rho_0_E', POINTER(c_complex)),

        ('time_AE', POINTER(c_double)),

        ('frequency_A', POINTER(c_double)),
        ('frequency_E', POINTER(c_double)),

        ('field_amp_AE', c_double),

        ('nDIM', c_int),
        ('nEXC', c_int),

        ('timeDIM_AE', c_int),

        ('freqDIM_A', c_int),
        ('freqDIM_E', c_int),
        ('field_A', POINTER(c_complex)),
        ('field_E', POINTER(c_complex)),
    ]


class Molecule(Structure):
    """
    Parameters structure ctypes
    """
    _fields_ = [
        ('nDIM', c_int),
        ('energies', POINTER(c_double)),
        ('gamma_decay', POINTER(c_double)),
        ('gamma_dephasing', POINTER(c_double)),
        ('mu', POINTER(c_complex)),
        ('rho', POINTER(c_complex)),
        ('rho_0', POINTER(c_complex)),
        ('abs_spectra', POINTER(c_double)),
        ('ems_spectra', POINTER(c_double)),
    ]


try:
    # Load the shared library assuming that it is in the same directory
    lib1 = ctypes.cdll.LoadLibrary(os.getcwd() + "/Spectra.so")
except OSError:
    raise NotImplementedError(
        """
        The library is absent. You must compile the C shared library using the commands:
        gcc -O3 -shared -o Spectra.so Spectra.c -lnlopt -lm -fopenmp -fPIC
        """
    )

#####################################################
#                                                   #
#          Declaring CalculateSpectra function      #
#                                                   #
#####################################################

lib1.CalculateSpectra.argtypes = (
    POINTER(Molecule),                  # molecule molA
    POINTER(Molecule),                  # molecule molB
    POINTER(Parameters_Spectra),        # parameter field_params
)
lib1.CalculateSpectra.restype = POINTER(c_complex)


def CalculateSpectra(molA, molB, spec_params):
    return lib1.CalculateSpectra(
        molA,
        molB,
        spec_params
    )
