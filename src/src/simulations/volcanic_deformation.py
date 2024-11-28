from src.importation import numpy as np, List, Tuple

def spatial_noise_2d(
    shape: Tuple, 
    complex_gaussian_noise: np.ndarray, 
    pixel_size: float, 
    l: float
) -> np.ndarray:

    size_x, size_y = [x * pixel_size for x in shape]

    xs, ys = np.mgrid[
        -l * size_x / 2: l * size_x / 2: shape[0] * 1j,  # complex number make mgrid shift from step to num parameter (like np.linspace and np.arange)
        -l * size_y / 2: l * size_y / 2: shape[1] * 1j
    ]
    rs = np.sqrt(xs ** 2 + ys ** 2)
    cs = np.exp(-rs * l) * complex_gaussian_noise

    im = np.real(np.fft.ifft2(np.fft.ifftshift(cs)))
    im /= np.std(im)

    return im

POISSON_RATIO_LAMBDA = 2.3e10
POISSON_RATIO_MU = 2.3e10
POISSON_RATIO = POISSON_RATIO_LAMBDA / (2*(POISSON_RATIO_LAMBDA + POISSON_RATIO_MU))

class Volcanic_Deformation:

    def __init__(self, x: float, y: float, noise: bool = False) -> None:
        self.position = np.array([x, y])
        self.noise = noise

    def __call__(self, observator, position: np.ndarray) -> float:
        deformation_value = self.deformation(observator, position)
        if self.noise:
            # noise_value = self.noise_map(*[position[i] for i in range(2)]) 
            # noise_value -= .5
            # noise_value *= 2 * np.quantile(np.abs(deformation_value), 0.9)
            # deformation_value += noise_value
            deformation_value += spatial_noise_2d(
                deformation_value.shape, 
                np.random.gamma(0.1, 1, deformation_value.shape) * np.exp(1j * np.random.uniform(0, 2 * np.pi, deformation_value.shape)), 
                0.1, 
                4
            ) * self.noise
        return deformation_value
    
    # abstract method
    def deformation(self, observator, position) -> float:
        pass

    def grid(
        self, 
        observator
    ) -> np.ndarray:
        return self(
            observator, 
            np.meshgrid(
                *[
                    np.arange(lims[0], lims[1], observator.resolutions[i]) 
                    for i, lims in enumerate(
                        [
                            [-(-1) ** j * res * 64 / 2 for j in range(2)] 
                            for res in observator.resolutions
                        ]
                    )
                ]
            )
        )

    def show(
        self,
        ax,
        observator,
        x_lims: List[float], 
        y_lims: List[float],
        log = False
    ) -> None:
        
        deformation_grid = self.grid(
            observator
        )

        v_max = 0.5

        x_ticks, y_ticks = [np.linspace(lims[0], lims[1], 5) for lims in [x_lims, y_lims]]

        if log:
            pos_df_grid = np.ma.masked_array(deformation_grid, deformation_grid < 0)
            neg_df_grid = -np.ma.masked_array(deformation_grid, deformation_grid > 0)

            ax.imshow(np.log10(pos_df_grid), vmin=0, vmax=np.log10(v_max), cmap = "Reds")
            ax.imshow(np.log10(neg_df_grid), vmin=0, vmax=np.log10(v_max), cmap = "Blues")

        else:
            ax.imshow(deformation_grid, vmin=-v_max, vmax=v_max, cmap="coolwarm")
        
        ax.set_xticks(np.linspace(0, deformation_grid.shape[0], 5), x_ticks)
        ax.set_yticks(np.linspace(0, deformation_grid.shape[1], 5), y_ticks)
        #ax.set_title(f"Depth: {self.depth:.2f} km ")
        ax.set_title(f"\
            Depth: {self.depth:.1e} $km$    $\Delta$V: {self.delta_volume:.1e} $km^3$\n\
            Azimuth: {observator.azimuth / np.pi * 180:.1f}°    Incidence: {observator.incidence / np.pi * 180:.1f}°"
        )

class Mogi(Volcanic_Deformation):
    """
    depth (float): depth of the event (in kilometers)
    delta_volume (float): volume change (in cubic kilometers)
    position (tuple of floats): x and y coordinates of the event (in kilometers)
    """
    def __init__(self, depth, delta_volume, x, y, noise) -> None:
        super().__init__(x, y, noise)
        self.depth = depth
        self.delta_volume = delta_volume
        self.noise = noise
    
    def deformation(self, observator, position: np.ndarray) -> np.ndarray:
        """
        LOOKS LIKE AN ERROR, DONT USE THIS FORMULA
        displacement_coefficient = 1e6 * self.delta_volume * (1 - POISSON_RATIO) / np.pi

        position_diff = np.array([position[i] - self.position[i] for i in range(2)])

        top_distance = np.linalg.norm(position_diff, axis=0)
        angle_distance = np.arctan2(*position_diff.T[::-1].T)        

        total_distance = (top_distance ** 2 + self.depth ** 2) ** 1.5
        lateral_displacement = displacement_coefficient * top_distance / total_distance

        # deformation in x, y and z axis
        easting_displacement = np.cos(angle_distance) * lateral_displacement
        northing_displacement = np.sin(angle_distance) * lateral_displacement
        uplifting = displacement_coefficient * self.depth / total_distance
        """

        position_diff = np.array([position[i] - self.position[i] for i in range(2)])

        top_distance = (position_diff[0] ** 2 + position_diff[1] ** 2) ** .5
        total_distance = (top_distance ** 2 + self.depth ** 2) ** .5
        angle_distance = np.arctan2(*position_diff.T[::-1].T)

        displacement_coefficient = self.delta_volume * (1 - POISSON_RATIO) / total_distance ** 3 / np.pi

        deformation_orientated = np.array([
            *[displacement_coefficient * f(angle_distance) * top_distance for f in [np.cos, np.sin]],
            displacement_coefficient * self.depth
        ])

        # deformation in the line of sight
        deformation_observed = (-observator.los_vector @ deformation_orientated.reshape(3,-1)) 
        
        _, *final_shape = deformation_orientated.shape
        
        return deformation_observed.reshape(*final_shape)
        
class Observator:
    """
    -incidence: (float) angle (IN RADIANS) between the z axis (vertical direction) and the line of sight
    -azimuth: (float) angle (IN RADIANS) between the x axis (east direction) and the line of sight
    """
    def __init__(self, azimuth, incidence, x_res: float, y_res: float) -> None:
        self.azimuth = azimuth
        self.incidence = incidence
        self.resolutions = np.array([x_res, y_res])
        self.los_vector = np.array([
            np.cos(self.azimuth) * np.sin(self.incidence),
            np.sin(self.azimuth) * np.sin(self.incidence), 
            -np.cos(self.incidence)
        ])

"""
if __name__ == "__main__":
    
    import time
    import matplotlib.pyplot as plt

    # for mogi only

    print("Plotting examples...")    

    variables_sets = {
        "azimuth": np.linspace(0, 6 * np.pi / 4, 4),
        "incidence": np.linspace(0, np.pi/2, 5),
        "depth": np.exp(0 + np.sqrt(2) * 1 * special.erfinv(2 * np.linspace(0,1,8) - 1)),
        "delta_volume": np.exp(-1 + np.sqrt(2) * 1 * special.erfinv(2 * np.linspace(0,1,8) - 1))
    }

    deformation_parameters = {
        "delta_volume": 1e-3,
        "depth": 1e-1,
        "x": 0.,
        "y": 0.
    }
    
    observator_parameters = {
        "azimuth": 0.,
        "incidence": 0.33,
        "x_res": 0.1,
        "y_res": 0.1
    }

    variables = "delta_volume", "depth"
    variable_1, variable_2 = variables

    variable_1_values = variables_sets[variable_1]
    variable_2_values = variables_sets[variable_2]

    fig = plt.figure()
    axs = fig.subplots(len(variable_1_values), len(variable_2_values))
    v_max = 0.01 # maximum displacement in cbar
    for axs_, variable_1_value in zip(axs, variable_1_values):
        for ax, variable_2_value in zip(axs_, variable_2_values):
            
            for variable, value in zip(variables, [variable_1_value, variable_2_value]):
                for parameters in [deformation_parameters, observator_parameters]:
                    if variable in parameters.keys():
                        parameters[variable] = value
            
            deformation = Mogi(**deformation_parameters)
            observator = Observator(**observator_parameters)
            
            extent = [[-4, 4], [-4, 4]]

            deformation.show(ax, observator, *extent)

    plt.tight_layout()
    plt.show()

    print("Testing the computing time...")

    n = 10
    tries = 10
    t = 0
    for i in range(tries):
        t0 = time.time()
        for j in range(n):
            deformation.grid(observator, *extent)
            t += time.time() - t0
            t0 = time.time()
    t_mean = t / n / tries

    print("Each sample takes roughly", t_mean, "seconds to be generated")
"""