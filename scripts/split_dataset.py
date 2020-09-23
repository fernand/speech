import os
import pickle
import random
import sys

TARGET_EVAL_DURATION = 10.0 * 3600

# Target eval set should be 10 hours.
def split_with_cer(dataset_dir, manifest, max_cer):
    clean_dataset = []
    for path, cer, duration in manifest:
        if cer == 0.0:
            clean_dataset.append((path, duration))
    random.shuffle(clean_dataset)
    eval_dataset = []
    train_dataset = []
    total_eval_duration = 0.0
    for path, duration in clean_dataset:
        if total_eval_duration < TARGET_EVAL_DURATION:
            eval_dataset.append((path, duration))
            total_eval_duration += duration
        else:
            train_dataset.append((path, duration))
    # Now add the less clean audio to the train set.
    for path, cer, duration in manifest:
        if cer != 0.0 and cer <= max_cer:
            train_dataset.append((path, duration))
    eval_dataset = sorted(eval_dataset, key=lambda t: t[1])
    train_dataset = sorted(train_dataset, key=lambda t: t[1])
    train_duration = round(sum([t[1] for t in train_dataset]) / 3600, 1)
    eval_duration = round(sum([t[1] for t in eval_dataset]) / 3600, 1)
    print(f"TOTAL TRAIN DURATION: {train_duration} hrs")
    print(f"TOTAL EVAL DURATION: {eval_duration} hrs")
    with open(os.path.join(dataset_dir, f"sorted_train_cer_{max_cer}.pkl"), "wb") as f:
        pickle.dump(train_dataset, f)
    with open(os.path.join(dataset_dir, f"sorted_eval_cer_{max_cer}.pkl"), "wb") as f:
        pickle.dump(eval_dataset, f)


if __name__ == "__main__":
    dataset_dir = sys.argv[1]
    max_cer = float(sys.argv[2])
    with open(os.path.join(dataset_dir, "cers.pkl"), "rb") as f:
        manifest = pickle.load(f)
    manifest = [t for chunk in manifest for t in chunk]
    split_with_cer(dataset_dir, manifest, max_cer)
