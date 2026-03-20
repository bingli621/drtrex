"""Constants and functions to convert between ToF, Energy in meV and Wavelengths in Å"""

TOF_CONSTANT = 252.78  # µs·Å-1·m-1 for conversion between lambda and ToF
ENERGY_CONSTANT = 81.8042  # meV·Å**2
H_OVER_MN = 3956  # m/s


def Lechner(Ld, lambda_f, lambda_i, tau_M, tau_P):
    """
    Calculate energy resolution using Lechner Formula.
    Inputs:
        - Ld: Any flight path uncertainties, in mm
        - lambda_f: Scattered wavelength in Å
        - lambda_i: Incident wavelength in Å
        - tau_M: M-chopper opening time (seconds)
        - tau_P: P-chopper opening time (seconds)
    Outputs:
        - Returns delta E / FWHM in meV
    """
    # Convert Ld from mm to m
    dL = Ld * 1e-3
    lambda_i = lambda_i * 1e-10
    lambda_f = lambda_f * 1e-10

    A = (0.2041 * tau_M) * (
        L_PM + L_MS + L_SD * (lambda_f**3 / lambda_i**3)
    )  # [A] = s * m
    B = (0.2887 * tau_P) * (L_MS + L_SD * (lambda_f**3 / lambda_i**3))  # [B] = s * m
    C = mn * L_PM * lambda_f * dL / h  # [C] = s * m

    # Final energy resolution calculation
    return (
        ((h**3.0) / (mn**2))
        * np.sqrt((A * A) + (B * B) + (C * C))
        * 1.0
        / (L_SD * L_PM * (lambda_f**3.0))
        * 6.21
        * 1e21
    )  # Convert Joules -> meV
