import os
import numpy as np
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)

def compute_gci(phi1, phi2, phi3, node_count_coarse, node_count_medium, dim=1, Fs=1.25):
    """
    Compute GCI using Richardson extrapolation with dimensionality awareness.

    Parameters:
        phi1, phi2, phi3 (float): Quantity values from coarse to fine resolutions.
        node_count_coarse (int): Node count for coarse grid (phi1).
        node_count_medium (int): Node count for medium grid (phi2).
        dim (int): Problem dimensionality (1, 2, or 3).
        Fs (float): Safety factor (default: 1.25).

    Returns:
        dict: Contains convergence metrics or NaN values on error.
    """
    try:
        # Validate inputs
        if not all(isinstance(v, (int, float)) for v in [phi1, phi2, phi3]:
            raise ValueError("Non-numeric phi values")
        if node_count_coarse >= node_count_medium:
            raise ValueError("Invalid node counts: medium grid must be finer than coarse")
        if dim not in [1, 2, 3]:
            raise ValueError("Dimensionality must be 1, 2, or 3")

        # Calculate effective refinement ratio
        r = (node_count_medium / node_count_coarse) ** (1/dim)
        
        eps21 = phi2 - phi3
        eps10 = phi1 - phi2

        # Numerical stability checks
        if (np.isclose(eps21, 0, atol=1e-12) or 
            np.isclose(eps10, 0, atol=1e-12) or 
            (eps10 / eps21) <= 0):
            raise ValueError("Non-monotonic convergence detected")

        p = np.log(eps10 / eps21) / np.log(r)
        if not np.isfinite(p):
            raise ValueError("Invalid convergence order")

        phi_ext = phi3 + eps21 / (r**p - 1)
        
        # Handle potential division by zero in error calculation
        if np.isclose(phi3, 0, atol=1e-12):
            error = np.inf
        else:
            error = abs((phi3 - phi2) / phi3)
        
        gci = Fs * error / (r**p - 1) * 100

        return {
            'p': p,
            'phi_ext': phi_ext,
            'GCI (%)': gci,
            'error (%)': error * 100
        }

    except Exception as e:
        print(f"{Fore.YELLOW}GCI Warning: {str(e)}{Style.RESET_ALL}")
        return {
            'p': np.nan,
            'phi_ext': np.nan,
            'GCI (%)': np.nan,
            'error (%)': np.nan
        }

def generate_gci_report(results, node_counts, angles, output_path, dim=1):
    """
    Enhanced report generator with dimensionality awareness and error resilience.
    """
    with open(output_path, 'w') as f:
        # Write report header
        header = (
            f"GCI Report (Dimension: {dim}D)\n"
            "Method: 3-point Richardson Extrapolation (ASME V&V20-2009)\n"
            f"Safety Factor: 1.25 | Quantities: {', '.join(['Izz', 'Iyy', 'Izy', 'Ixx', 'Area'])}\n"
            + "=" * 115 + "\n\n"
        )
        f.write(header)
        print(Fore.CYAN + header)

        for angle in angles:
            angle_data = results.get(angle, {})
            if not angle_data:
                msg = f"No data for angle {angle}°"
                f.write(msg + "\n")
                print(Fore.RED + msg)
                continue

            # Sort resolutions by actual node counts
            try:
                sorted_res = sorted(angle_data.keys(), key=lambda x: node_counts[x])
                if len(sorted_res) < 3:
                    raise ValueError(f"Require 3 grids, found {len(sorted_res)}")
                
                r_coarse, r_medium, r_fine = sorted_res[-3:]
                nc_coarse = node_counts[r_coarse]
                nc_medium = node_counts[r_medium]
                
                # Table headers
                f.write(f"Rotation Angle: {angle}°\n")
                f.write("-"*115 + "\n")
                f.write(f"{'Quantity':<8} | {'p':^8} | {'phi_ext':^12} | {'GCI (%)':^10} | "
                        f"{'Error (%)':^10} | {'Classification'}\n")
                f.write("-"*115 + "\n")

                print(Fore.MAGENTA + f"Rotation Angle: {angle}°")
                print(Fore.MAGENTA + "-"*115)

                # Process each section property
                for idx, name in enumerate(['Izz', 'Iyy', 'Izy', 'Ixx', 'Area']):
                    try:
                        phi1 = angle_data[r_coarse][idx]
                        phi2 = angle_data[r_medium][idx]
                        phi3 = angle_data[r_fine][idx]
                        
                        gci_data = compute_gci(phi1, phi2, phi3, nc_coarse, nc_medium, dim)
                        classification = classify_gci(gci_data['GCI (%)'])

                        # File logging
                        f.write(
                            f"{name:<8} | {gci_data['p']:^8.2f} | "
                            f"{gci_data['phi_ext']:^12.4e} | "
                            f"{gci_data['GCI (%)']:^10.2f} | "
                            f"{gci_data['error (%)']:^10.2f} | "
                            f"{classification}\n"
                        )

                        # Terminal output
                        gci_str = colorize_gci(gci_data['GCI (%)'])
                        err_str = (f"{gci_data['error (%)']:.2f}%" 
                                   if not np.isnan(gci_data['error (%)']) 
                                   else "N/A")
                        print(
                            f"{name:<8} | {gci_data['p']:^8.2f} | "
                            f"{gci_data['phi_ext']:^12.4e} | "
                            f"{gci_str:^10} | {err_str:^10} | {classification}"
                        )

                        # Convergence order validation
                        if not (0.5 < gci_data['p'] < 5.0):
                            warn_msg = (f"Questionable convergence order p={gci_data['p']:.2f} "
                                       f"for {name} at {angle}°")
                            print(Fore.YELLOW + "⚠️  " + warn_msg)

                    except KeyError:
                        msg = f"Missing data for {name} at {angle}°"
                        f.write(msg + "\n")
                        print(Fore.RED + msg)

            except Exception as e:
                msg = f"Angle {angle}°: {str(e)}"
                f.write(msg + "\n")
                print(Fore.RED + msg)

            f.write("\n")
            print()

def classify_gci(gci_value):
    if np.isnan(gci_value):
        return "Invalid GCI (zero or sign-changing differences)"
    elif gci_value < 0.5:
        return "Excellent convergence (asymptotic range)"
    elif gci_value < 1.5:
        return "Very good convergence"
    elif gci_value < 5.0:
        return "Moderate convergence — may need refinement"
    else:
        return "Poor convergence — refine or verify discretisation"


def colorize_gci(value):
    if np.isnan(value):
        return f"{Fore.YELLOW}N/A{Style.RESET_ALL}"
    elif value < 0.5:
        return f"{Fore.GREEN}{value:.2f}%{Style.RESET_ALL}"
    elif value < 5.0:
        return f"{Fore.YELLOW}{value:.2f}%{Style.RESET_ALL}"
    else:
        return f"{Fore.RED}{value:.2f}%{Style.RESET_ALL}"