import os
import glob
import numpy as np
import sys
import bisect
from bisect import bisect_left
from get_training_time import parse_args
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from collections import defaultdict

def sifmt(i):
    affix = iter(['', 'K', 'M', 'B', 'T', 'P', 'E'])
    while i > 1000:
        i /= 1000
        next(affix)
    return f'{i:.1f}{next(affix)}'


def extract_scalar_data(log_dir, tag='data params/total tokens',specific_run_number=None):
    """
    Extract scalar values from TensorBoard logs for a specific tag.
    
    Args:
        log_dir: Parent directory containing TensorBoard logs
        tag: The scalar tag to extract
        
    Returns:
        Dictionary with run names as keys and lists of (step, value) tuples as values
    """
    run_name = os.path.basename(log_dir)
    runs_data = defaultdict(list)
    event_files = glob.glob(f"{log_dir}/*/events.*")
    if specific_run_number is not None:
        if len(specific_run_number)>1:
            event_files = [f for f in event_files if int(os.path.dirname(f).split('-')[-1]) >= specific_run_number[0] <= specific_run_number[1]]
        else:    
            event_files = [f for f in event_files if int(os.path.dirname(f).split('-')[-1]) >= specific_run_number[0]]
        
    print(f"Extracting data from run {run_name}",flush=True)
    for i,f in enumerate(event_files,start=1):
        ea = EventAccumulator(f)
        ea.Reload()
        # pick any scalar tag that exists
        tags = ea.Tags().get("scalars", [])
        if not tags:
            continue
        # Check if the tag exists
        if tag in ea.scalars.Keys():
            # Extract all scalar events for this tag
            scalar_events = ea.Scalars(tag)
            # Store as (step, value) pairs
            runs_data[run_name].extend([(e.step, e.value) for e in scalar_events])
        if i % 10 == 0:
            print(f"Processed {(i/len(event_files))*100:.2f}% of event files",flush=True)
    return runs_data



def get_values_at_target_steps(data, target_steps):
    """
    Extract values at specific target steps or their nearest available steps.
    
    Args:
        data: Dictionary with run names and (step, value) lists
        target_steps: List of specific steps to extract values from
        
    Returns:
        Dictionary with run names and values at target steps
    """
    results = {}
    
    for run, step_values in data.items():
        # Sort by step
        step_values.sort(key=lambda x: x[0])
        
        # Extract steps and values into separate lists for easier processing
        steps = [sv[0] for sv in step_values]
        values = [sv[1] for sv in step_values]
        
        run_results = []
        for target in target_steps:
            # Find the index of the closest step
            if target <= steps[0]:
                # Target is before or at the first step
                closest_idx = 0
            elif target >= steps[-1]:
                # Target is at or after the last step
                closest_idx = len(steps) - 1
            else:
                # Find the insertion point
                idx = bisect.bisect_left(steps, target)
                # Determine if the previous or next step is closer
                if idx == len(steps):
                    closest_idx = idx - 1
                elif idx == 0:
                    closest_idx = idx
                else:
                    prev_diff = target - steps[idx-1]
                    next_diff = steps[idx] - target
                    closest_idx = idx if next_diff < prev_diff else idx-1
            
            actual_step = steps[closest_idx]
            value = values[closest_idx]
            run_results.append((target, actual_step, value))
        
        results[run] = run_results
    
    return results

def find_closest_step(step_values, target_step):
    """Find the closest step to the target step in the data."""
    steps = [s[0] for s in step_values]
    pos = bisect_left(steps, target_step)
    
    if pos == 0:
        return 0  # First step
    if pos == len(steps):
        return len(steps) - 1  # Last step
    
    # Check which is closer
    if abs(steps[pos] - target_step) < abs(steps[pos-1] - target_step):
        return pos
    else:
        return pos - 1


def calculate_mean_increase(data, window_size=10, step_ranges=None):
    """
    Calculate the mean increase between steps for each run.
    
    Args:
        data: Dictionary with run names and (step, value) lists
        window_size: Number of steps to use for calculating the mean increase
        step_ranges: List of tuples (start_step, end_step) to calculate increases for
        
    Returns:
        Dictionary with run names and mean increase values
    """
    results = {}
    
    for run, step_values in data.items():
        # Sort by step if not already sorted
        step_values.sort(key=lambda x: x[0])
        
        if step_ranges:
            # Calculate mean increase for specific step ranges
            range_results = []
            
            for start_step, end_step in step_ranges:
                # Find closest indices to the specified steps
                start_idx = find_closest_step(step_values, start_step)
                end_idx = find_closest_step(step_values, end_step)
                
                if start_idx == end_idx:
                    # Same point or too close together
                    range_results.append({
                        'range': f"{start_step}-{end_step}",
                        'actual_range': f"{step_values[start_idx][0]}-{step_values[end_idx][0]}",
                        'total_increase': 0,
                        'mean_increase_per_step': 0,
                        'steps': 0
                    })
                    continue
                
                start_step_actual, start_value = step_values[start_idx]
                end_step_actual, end_value = step_values[end_idx]
                
                total_increase = end_value - start_value
                step_diff = end_step_actual - start_step_actual
                
                if step_diff <= 0:
                    mean_increase_per_step = 0
                else:
                    mean_increase_per_step = total_increase / step_diff
                
                range_results.append({
                    'range': f"{start_step}-{end_step}",
                    'actual_range': f"{start_step_actual}-{end_step_actual}",
                    'total_increase': total_increase,
                    'mean_increase_per_step': mean_increase_per_step,
                    'steps': step_diff
                })
            
            results[run] = range_results
            
        else:
            # Calculate differences between consecutive values using window_size
            increases = []
            for i in range(1, len(step_values)):
                prev_step, prev_value = step_values[i-1]
                curr_step, curr_value = step_values[i]
                
                # For a cumulative metric, the increase is just the difference
                increase = curr_value - prev_value
                
                # Store the increase and the step difference
                step_diff = curr_step - prev_step
                if step_diff > 0:  # Avoid division by zero
                    increases.append((curr_step, increase, increase / step_diff))
            
            # Get the last window_size increases or all if less than window_size
            recent_increases = increases[-window_size:] if len(increases) > window_size else increases
            
            if recent_increases:
                # Calculate mean increase per step
                mean_increase = np.mean([inc for _, inc, _ in recent_increases])
                # Calculate mean increase normalized by step difference
                mean_increase_per_step = np.mean([inc_per_step for _, _, inc_per_step in recent_increases])
                
                results[run] = {
                    'mean_increase': mean_increase,
                    'mean_increase_per_step': mean_increase_per_step,
                    'steps_analyzed': window_size if len(increases) > window_size else len(increases)
                }
            else:
                results[run] = {
                    'mean_increase': 0,
                    'mean_increase_per_step': 0,
                    'steps_analyzed': 0
                }
    
    return results


# Example usage
if __name__ == "__main__":
    log_dir = "/path/to/your/tensorboard/logs"
    tag = "data params/total tokens"
    args = parse_args()

    # Extract data
    scalar_data = extract_scalar_data(args.log_dir, args.scalar_tag,args.specific_run_number)
    # Calculate mean increase using 10 steps
    specific_ranges = [(0,4002),(4002, 117300), (117300, 133860), (133860, 137999),(0,137999)] 
    # Calculate mean increase for specific step ranges
    range_increases = calculate_mean_increase(scalar_data, step_ranges=specific_ranges)
    # Display results for specific ranges
    for run, ranges in range_increases.items():
        print(f"Run: {run}")
        for r in ranges:
            print(f"  Range: {r['range']} (actual: {r['actual_range']})")
            print(f"    Total increase: {sifmt(r['total_increase'])}")
            print(f"    Mean increase per step: {sifmt(r['mean_increase_per_step'])}")
            print(f"    Step difference: {r['steps']}")
    # Option 2: Get values at specific target steps
    target_steps = [117300, 133860, 137999]
    target_values = get_values_at_target_steps(scalar_data, target_steps)
    
    # Display results for option 2
    print("\nValues at target steps:")
    for run, steps_data in target_values.items():
        print(f"Run: {run}")
        for target, actual_step, value in steps_data:
            print(f"  Target step: {target}, Actual step: {actual_step}, Value: {sifmt(value)}")
    