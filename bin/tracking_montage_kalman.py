#!/opt/conda/bin/python

import cv2
import numpy as np
import logging
import argparse
import os
from sql import Database
import pandas as pd
import collections
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.optimize import linear_sum_assignment


logger = logging.getLogger("TrackingDB")
logging.basicConfig(level=logging.INFO)

class Cell:
    """A class that makes cells from contours or masks."""
    def __init__(self, cnt):
        self.cnt = cnt

    def __repr__(self):
        center, _ = self.get_circle()
        return f"Cell instance at {center}"

    def get_circle(self):
        center, radius = cv2.minEnclosingCircle(self.cnt)
        return center, radius

    def evaluate_overlap(self, circle2):
        c2, r2 = circle2
        c1, r1 = self.get_circle()
        dist = np.hypot(c1[0] - c2[0], c1[1] - c2[1])
        return dist < (r1 + r2) * 0.8

    def evaluate_dist(self, circle2):
        c2, _ = circle2
        c1, _ = self.get_circle()
        return np.hypot(c1[0] - c2[0], c1[1] - c2[1])

class KalmanCellTracker:
    def __init__(self, initial_position):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0
        self.kf.F = np.array([[1, 0, dt, 0],
                              [0, 1, 0, dt],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])
        self.kf.P *= 10.
        self.kf.R *= 2.
        self.kf.Q = np.eye(4) * 0.01
        self.kf.x[:2] = np.array([[initial_position[0]], [initial_position[1]]])

    def predict(self):
        self.kf.predict()
        return self.kf.x[:2].flatten()

    def update(self, measurement):
        self.kf.update(measurement)

def sort_cell_info_by_index(time_dict, time_list):
    for tp in time_list:
        time_dict[tp] = sorted(
            [(int(idx), obj) for idx, obj in time_dict[tp]],
            key=lambda x: x[0]
        )
    return time_dict

def populate_cell_ind_overlap(time_dict, time_list):
    first = time_list[0]
    for i, rec in enumerate(time_dict[first], 1): rec[0] = i
    counter = len(time_dict[first]) + 1
    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i-1]
        for rec in time_dict[curr]:
            circ = rec[1].get_circle(); matched = False
            for p in time_dict[prev]:
                if p[1].evaluate_overlap(circ): rec[0] = p[0]; matched = True; break
            if not matched: rec[0] = counter; counter += 1
    return time_dict

def populate_cell_ind_closest(time_dict, time_list, max_dist=100):
    first = time_list[0]
    for i, rec in enumerate(time_dict[first],1): rec[0] = i
    counter = len(time_dict[first]) + 1
    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i-1]
        for rec in time_dict[curr]:
            circ = rec[1].get_circle(); best = float('inf')
            for p in time_dict[prev]:
                d = p[1].evaluate_dist(circ)
                if d < best: best, rec[0] = d, p[0]
            if best > max_dist: rec[0] = counter; counter += 1
    return time_dict

def populate_cell_ind_kalman(time_dict, time_list, max_dist=100):

    print(f"Kalman Filtering ...")
    
    first = time_list[0]
    trackers = {}
    for i, rec in enumerate(time_dict[first], 1):
        rec[0] = i
        c, _ = rec[1].get_circle()
        trackers[i] = KalmanCellTracker(c)

    counter = len(time_dict[first]) + 1

    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i - 1]

        predictions = {idx: trackers[idx].predict() for idx, _ in time_dict[prev]}
        unmatched = set(predictions.keys())

        for rec in time_dict[curr]:
            c, _ = rec[1].get_circle()
            dists = {idx: np.linalg.norm(pred - c) for idx, pred in predictions.items()}
            if dists:
                min_id = min(dists, key=dists.get)
                if dists[min_id] < max_dist:
                    rec[0] = min_id
                    trackers[min_id].update(c)
                    unmatched.discard(min_id)
                else:
                    rec[0] = counter
                    trackers[counter] = KalmanCellTracker(c)
                    counter += 1
            else:
                rec[0] = counter
                trackers[counter] = KalmanCellTracker(c)
                counter += 1

    return time_dict



def populate_cell_ind_kalman_hungarian(time_dict, time_list, max_dist=100):
    print(f"Kalman + Hungarian Filtering ...")
    
    first = time_list[0]
    trackers = {}
    for i, rec in enumerate(time_dict[first], 1):
        rec[0] = i
        c, _ = rec[1].get_circle()
        trackers[i] = KalmanCellTracker(c)

    counter = len(time_dict[first]) + 1

    for i in range(1, len(time_list)):
        curr, prev = time_list[i], time_list[i - 1]

        predictions = {idx: trackers[idx].predict() for idx, _ in time_dict[prev]}
        unmatched_trackers = set(predictions.keys())

        curr_positions = []
        for rec in time_dict[curr]:
            c, _ = rec[1].get_circle()
            curr_positions.append(c)

        if not predictions or not curr_positions:
            # No matching possible, assign new IDs to all current detections
            for rec in time_dict[curr]:
                rec[0] = counter
                c, _ = rec[1].get_circle()
                trackers[counter] = KalmanCellTracker(c)
                counter += 1
            continue

        pred_ids = list(predictions.keys())
        pred_positions = np.array([predictions[idx] for idx in pred_ids])
        curr_positions = np.array(curr_positions)

        # Build cost matrix
        cost_matrix = np.linalg.norm(pred_positions[:, np.newaxis, :] - curr_positions[np.newaxis, :, :], axis=2)

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        assigned_curr = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < max_dist:
                assigned_id = pred_ids[r]
                time_dict[curr][c][0] = assigned_id
                trackers[assigned_id].update(curr_positions[c])
                unmatched_trackers.discard(assigned_id)
                assigned_curr.add(c)
            else:
                # Distance too large, leave unmatched
                pass

        # Assign new IDs to unassigned detections
        for idx, rec in enumerate(time_dict[curr]):
            if idx not in assigned_curr:
                rec[0] = counter
                c, _ = rec[1].get_circle()
                trackers[counter] = KalmanCellTracker(c)
                counter += 1

    return time_dict

class MontageDBTracker:
    def __init__(self, experiment, track_type, max_dist):
        self.Db = Database()
        self.experiment = experiment
        self.track_type = track_type
        self.max_dist = max_dist
        logger.info(f"Initialized MontageDBTracker for experiment {experiment}")

    def gather_encoded_from_db(self, wells, channel_marker="_ENCODED_MONTAGE"):
        from db_util import Ops
        import argparse

        opt_inner = argparse.Namespace(
            experiment=self.experiment,
            chosen_wells=','.join(wells),
            wells_toggle='include',
            chosen_timepoints='',
            timepoints_toggle='include',
            chosen_channels='all',
            channels_toggle='include',
            tile=0
        )
        
        op = Ops(opt_inner)
        tiledata_df = op.get_tiledata_df()
        df = tiledata_df[
            tiledata_df['well'].isin(wells) &
            tiledata_df['newmaskmontage'].str.contains(channel_marker, na=False)
        ]
        df = df.groupby(['well','timepoint'], as_index=False).agg({'newmaskmontage':'first'})

        results = {}
        for well in wells:
            df_w = df[df['well'] == well]
            if df_w.empty:
                logger.warning(f"No encoded masks for well {well}")
                continue
            time_dict = collections.OrderedDict()
            for _, row in df_w.iterrows():
                mask_path = row['newmaskmontage']
                tp_label = os.path.basename(mask_path).split('_')[2]
                tp = int(tp_label.lstrip('T')) if tp_label.startswith('T') else int(tp_label)
                entries = time_dict.setdefault(tp, [])
                mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                labels = np.unique(mask); labels = labels[labels > 0]
                for lbl in labels:
                    bin_mask = (mask == lbl).astype(np.uint8) * 255
                    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        cnt = max(cnts, key=cv2.contourArea)
                        entries.append([lbl, Cell(cnt)])
            results[well] = time_dict
        return results, df, tiledata_df

    def run(self, wells):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        import numpy as np

        exp_id = self.Db.get_table_uuid('experimentdata', {'experiment': self.experiment})
        all_wells, df, tiledata_df = self.gather_encoded_from_db(wells)

        for well, time_dict in all_wells.items():
            welldata_id = self.Db.get_table_uuid('welldata', {'experimentdata_id': exp_id, 'well': well})
            tps = sorted(time_dict.keys())

            if self.track_type == 'overlap':
                tracked = populate_cell_ind_overlap(time_dict, tps)
            elif self.track_type == 'proximity':
                tracked = populate_cell_ind_closest(time_dict, tps, max_dist=self.max_dist)
            elif self.track_type == 'kalman':
                tracked = populate_cell_ind_kalman(time_dict, tps, max_dist=self.max_dist)
            elif self.track_type == 'kalman_hungarian':
                tracked = populate_cell_ind_kalman_hungarian(time_dict, tps, max_dist=self.max_dist)
            else:
                raise ValueError(f"Unknown track_type: {self.track_type}")

            sorted_td = sort_cell_info_by_index(tracked, tps)

            print(f"Well {well}:")
            for tp, recs in sorted_td.items():
                ids = [idx for idx, _ in recs]
                print(f"  T{tp}: {len(ids)} cells, IDs = {ids}")

            accuracy_history = []
            id_switch_history = []
            timepoints = []

            previous_ids = None
            id_persistence = {}

            for tp, recs in sorted_td.items():
                ids = [idx for idx, _ in recs]
                timepoints.append(int(tp))
                current_ids = set(ids)

                # Track persistence of each ID
                for idx in ids:
                    if idx not in id_persistence:
                        id_persistence[idx] = []
                    id_persistence[idx].append(tp)

                # Calculate accuracy
                if previous_ids is not None:
                    correct_matches = len(previous_ids & current_ids)
                    accuracy = (correct_matches / len(previous_ids)) * 100 if previous_ids else 0.0
                else:
                    accuracy = 100.0  # Assume 100% accuracy for the first frame

                accuracy_history.append(accuracy)

                # Calculate ID switches
                if previous_ids is not None:
                    id_switches = len(current_ids - previous_ids)
                else:
                    id_switches = 0

                id_switch_history.append(id_switches)
                previous_ids = current_ids

                # Save tracking results in DB and write tracked masks
                tiledata_id = self.Db.get_table_uuid('tiledata', {'experimentdata_id': exp_id, 'welldata_id': welldata_id, 'timepoint': tp})

                for new_id, _ in recs:
                    self.Db.update('celldata', update_dct={'cellid': int(new_id)}, kwargs={'tiledata_id': tiledata_id, 'randomcellid': int(new_id)})

                df_wtp = df[(df['well'] == well) & (df['timepoint'] == tp)]
                if df_wtp.empty:
                    logger.warning(f"No montage found for well {well} TP {tp}")
                    continue

                original_mask_path = df_wtp['newmaskmontage'].values[0]
                orig_mask = cv2.imread(original_mask_path, cv2.IMREAD_UNCHANGED)
                tracked_mask = np.zeros_like(orig_mask, dtype=np.uint16)

                for new_id, cell in recs:
                    bin_mask = np.zeros_like(orig_mask, dtype=np.uint8)
                    cv2.drawContours(bin_mask, [cell.cnt], -1, 255, -1)
                    tracked_mask[bin_mask > 0] = new_id

                tracked_mask_path = original_mask_path.replace('.tif', '_TRACKED.tif')
                cv2.imwrite(tracked_mask_path, tracked_mask)

                logger.warning(f"[TRACKED UPDATE] Well={well} TP={tp}")
                logger.warning(f"[TRACKED UPDATE] tracked_mask_path = {tracked_mask_path}")

                df_tiles_all = tiledata_df[(tiledata_df['welldata_id'] == welldata_id) & (tiledata_df['timepoint'] == tp)]

                if df_tiles_all.empty:
                    logger.warning(f"No tiledata rows found for {well} TP {tp}")
                else:
                    for tile_id in df_tiles_all['id']:
                        self.Db.update('tiledata', update_dct={'newtrackedmontage': tracked_mask_path}, kwargs={'id': tile_id})
                    logger.warning(f"Updated newtrackedmontage for {len(df_tiles_all)} tiles at {well} TP {tp}")

            # Summarize ID Persistence
            print("\n=== ID Persistence Summary ===")
            # Assume id_persistence is your dict: {cell_id: [list of timepoints]}
            appearances_counts = [len(tps_present) for tps_present in id_persistence.values()]

            plt.figure(figsize=(8, 5))
            plt.hist(appearances_counts, bins=range(1, max(appearances_counts)+2), align='left', color='skyblue', edgecolor='black')
            plt.xlabel('Number of Timepoints Cell Appeared In')
            plt.ylabel('Number of Cells')
            plt.title('Distribution of Cell ID Persistence Over Timepoints')
            plt.xticks(range(1, max(appearances_counts)+1))
            plt.grid(axis='y', alpha=0.75)
            plt.savefig('tracking_plots/hist.png')
            plt.show()

            # Plot accuracy and ID switches
            fig, axs = plt.subplots(1, 2, figsize=(14, 6))

            axs[0].plot(timepoints, accuracy_history, marker='o', color='green')
            axs[0].set_title('Tracking Accuracy Over Timepoints')
            axs[0].set_xlabel('Timepoint (Frame)')
            axs[0].set_ylabel('Accuracy (%)')
            axs[0].set_ylim(0, 110)

            axs[1].plot(timepoints, id_switch_history, marker='o', color='red')
            axs[1].set_title('New IDs Introduced Per Timepoint')
            axs[1].set_xlabel('Timepoint (Frame)')
            axs[1].set_ylabel('New IDs Introduced')

            plt.tight_layout()

            plot_dir = "tracking_plots"
            os.makedirs(plot_dir, exist_ok=True)
            plot_path = os.path.join(plot_dir, f"{well}_tracking_summary.png")
            plt.savefig(plot_path)
            print(f"Saved tracking summary plot at {plot_path}")

            plt.show()

            # Build ID Presence Heatmap
            all_ids = sorted(id_persistence.keys())
            tp_list = sorted(tps)
            id_matrix = np.zeros((len(all_ids), len(tp_list)))

            for i, idx in enumerate(all_ids):
                for tp in id_persistence[idx]:
                    j = tp_list.index(tp)
                    id_matrix[i, j] = 1  # ID is present

            plt.figure(figsize=(12, 8))
            sns.heatmap(id_matrix, cmap='viridis', cbar=True, xticklabels=tp_list, yticklabels=all_ids)
            plt.xlabel('Timepoints (Frames)')
            plt.ylabel('Cell IDs')
            plt.title(f'Cell ID Presence Heatmap: Well {well}')
            heatmap_path = os.path.join(plot_dir, f"{well}_ID_presence_heatmap.png")
            plt.savefig(heatmap_path)
            print(f"Saved ID presence heatmap at {heatmap_path}")
            plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Track montage masks stored in DB")
    parser.add_argument('--experiment', required=True, help='Experiment name')
    parser.add_argument('--track_type', choices=['overlap','proximity','kalman', 'kalman_hungarian'], default='overlap', help='Tracking method')
    parser.add_argument('--max_dist', type=int, default=100, help='Max distance for proximity or kalman')
    parser.add_argument('--wells', required=True, help='Comma-separated list of wells, e.g. A1,B1')
    args = parser.parse_args()
    wells = [w.strip() for w in args.wells.split(',')]
    tracker = MontageDBTracker(args.experiment, args.track_type, args.max_dist)
    tracker.run(wells)
