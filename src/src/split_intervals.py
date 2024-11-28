from src.importation import numpy as np, operator, Tuple, List, Dict

def get_axis_intervals(
    proportions: List, 
    axis_intervals: List,
    r, 
    rng: np.random.Generator,
) -> Tuple:
    """
    -proportions: (list of floats) proportion of each non-fixed dataset. the sum must be equal to 1
    -fixed_intervals: (lists of lists of two floats) intervals for the fixed dataset
    """
    
    total_intervals = 8

    n_datasets = len(proportions) + 1
    
    axis_positions = axis_intervals
    gaps_positions = gaps_positions = [0.] + [position for positions in axis_intervals for position in positions] + [1.]
    gaps_positions = list(zip(gaps_positions[::2], gaps_positions[1::2]))
    
    axis_indexes = [n_datasets - 1] * len(axis_intervals)

    for i, gap_position in enumerate(gaps_positions):
        gap_size = operator.sub(*gap_position[::-1])

        intervals_sizes = []
        intervals_indexes = []
        for j, proportion in enumerate(proportions):
            n_intervals = 1 + rng.poisson(total_intervals * gap_size * r)
            sub_intervals_sizes = rng.lognormal(0, 0.1, n_intervals)
            sub_intervals_sizes = sub_intervals_sizes / np.sum(sub_intervals_sizes) * gap_size * proportion
            intervals_indexes += [j] * n_intervals
            intervals_sizes += list(sub_intervals_sizes)
    
        shuffled = list(range(len(intervals_indexes)))
        rng.shuffle(shuffled)
        
        intervals_indexes = [intervals_indexes[j] for j in shuffled]
        intervals_sizes = [intervals_sizes[j] for j in shuffled]

        intervals_positions = [0.] + list(np.cumsum(intervals_sizes))
        intervals_positions = np.array(intervals_positions) + gap_position[0]
        intervals_positions = list(zip(intervals_positions[:-1], intervals_positions[1:]))

        cut_index = len(axis_positions) - len(gaps_positions) + i + 1

        axis_positions = axis_positions[:cut_index] + intervals_positions + axis_positions[cut_index:]
        axis_indexes = axis_indexes[:cut_index] + intervals_indexes + axis_indexes[cut_index:]
    
    return axis_indexes, axis_positions

def excluding_intervals(axes_fixed_intervals, proportions, r, rng):
    
    axes_intervals = [
        get_axis_intervals(
            proportions, 
            axis_fixed_intervals,
            r,
            rng
        ) for axis_fixed_intervals in axes_fixed_intervals.values()
    ]

    axes_indexes, axes_positions = [[axis_interval[i] for axis_interval in axes_intervals] for i in range(2)]
    
    parameters = list(axes_fixed_intervals.keys())

    return [
        {
            parameter: [
                axes_positions[j][k] for k, axis_index in enumerate(axes_indexes[j]) if axis_index == i
            ] for j, parameter in enumerate(parameters)
        } for i in range(len(proportions) + 1)
    ]

def random_sample(dataset_intervals, rng):
    parameters = dataset_intervals.keys()
    sample = {}
    for parameter in parameters:
        p = np.diff(np.array(dataset_intervals[parameter])[:, ::-1], axis=1).flatten()
        p /= np.sum(p)
        interval = rng.choice(dataset_intervals[parameter], p=p)
        sample[parameter] = rng.uniform(*interval)
    return sample