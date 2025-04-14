import matplotlib.pyplot as plt

def plot_geometry_stages(original, translated, enhanced_dict, connectivity=None):
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    axs = axs.ravel()

    # Plot original geometry
    axs[0].plot(original[:, 0], original[:, 1], 'o-', label='Original')
    axs[0].set_title('Original Input Geometry')
    axs[0].axis('equal')
    axs[0].legend()

    # Plot translated geometry
    axs[1].plot(translated[:, 0], translated[:, 1], 's-', label='Translated')
    axs[1].set_title('Translated Geometry (Centroid at Origin)')
    axs[1].axis('equal')
    axs[1].legend()

    # Plot enhanced geometries
    for res, enhanced in enhanced_dict.items():
        axs[2].plot(enhanced[:, 0], enhanced[:, 1], label=f'Res {res}')
    axs[2].set_title('Enhanced Geometry (Various Resolutions)')
    axs[2].axis('equal')
    axs[2].legend()

    plt.tight_layout()
    plt.show()