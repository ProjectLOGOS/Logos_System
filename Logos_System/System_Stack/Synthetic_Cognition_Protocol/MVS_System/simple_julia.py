# HEADER_TYPE: LEGACY_REWRITE_CANDIDATE
# EXECUTION: FORBIDDEN
# IMPORT: FORBIDDEN
# AUTHORITY: NONE
# DESTINATION: Logos_System_Rebuild
# ARCHIVE_AFTER_REWRITE: REQUIRED

"""
Simple Julia Set Generator using LOGOS c values
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import argparse

def generate_julia_set(c_value, width=800, height=600, max_iter=256):
    """Generate a Julia set for the given c value"""
    x = np.linspace(-2, 2, width)
    y = np.linspace(-1.5, 1.5, height)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    iterations = np.zeros(Z.shape, dtype=int)
    mask = np.ones(Z.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask]**2 + c_value
        escaped = np.abs(Z) > 2.0
        iterations[mask & escaped] = i
        mask &= ~escaped

    return iterations

def display_julia_set(c_value):
    """Display a Julia set"""
    print(f"Generating Julia set for c = {c_value}")

    iterations = generate_julia_set(c_value)

    plt.figure(figsize=(10, 8))
    plt.imshow(iterations, extent=[-2, 2, -1.5, 1.5],
               cmap='viridis', norm=LogNorm(vmin=1, vmax=256), origin='lower')
    plt.colorbar(label='Iterations')
    plt.title(f'Julia Set: c = {c_value}')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Simple Julia Set Display")
    parser.add_argument('--c-value', type=str, default="-0.7+0.27015j",
                       help='c value as complex number')

    args = parser.parse_args()

    try:
        c_val = complex(args.c_value)
        display_julia_set(c_val)
    except ValueError as e:
        print(f"Invalid c value: {e}")

if __name__ == '__main__':
    main()
