import glob
import argparse
import json
import os
from tensorboard.backend.event_processing import event_accumulator


def parse_args():
    parser = argparse.ArgumentParser(description='Process model evaluation results and find best learning rates.')
    parser.add_argument('--log_dir', type=str, required=True, help='Root directory containing the tensorboard logs')
    parser.add_argument('--output_dir',default=None ,type=str, help='Path to save the res')
    parser.add_argument('--scalar_tag',default='data params/total tokens',help="aggregation tag")
    parser.add_argument('--specific_run_number',action='append',default=None,type=int,help="wheter to filter event files")
    return parser.parse_args()

def get_training_hours(logdir):
    durations = []

    for f in glob.glob(f"{logdir}/*/events.*"):
        ea = event_accumulator.EventAccumulator(f)
        ea.Reload()

        # pick any scalar tag that exists
        tags = ea.Tags().get("scalars", [])
        if not tags:
            continue
        tag = tags[0]

        events = ea.Scalars(tag)
        if not events:
            continue

        rel_times = [e.wall_time - events[0].wall_time for e in events]
        durations.append(max(rel_times))

    total_hours = sum(durations) / 3600.0
    return total_hours, durations

if __name__ == "__main__":
    args = parse_args()
    total_hours, per_run = get_training_hours(args.log_dir)
    print(f"Total training time from logging directory {args.log_dir}: {total_hours:.2f} hours")
    print("Per-run durations (hours):", [round(d/3600, 2) for d in per_run])
    if args.output_dir:
        out_file = os.path.join(args.output_dir,f"{os.path.basename(args.log_dir)}-training-time.json")
        with open(out_file,"w") as f:
            d = {"total_hours":total_hours,"per_run":per_run}
            json.dump(d, f, indent=2)