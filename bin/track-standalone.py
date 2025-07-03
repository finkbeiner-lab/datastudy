import os
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


class KalmanCellTracker:
    def __init__(self, initial_position):
        self.dt = 1.0
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]])
        self.Q = np.eye(4) * 1e-2
        self.R = np.eye(2) * 1e-1
        self.P = np.eye(4)
        self.x = np.array([initial_position[0], initial_position[1], 0, 0], dtype=float)
        self.last_position = initial_position

    def predict(self):
        self.x = self.A @ self.x
        self.P = self.A @ self.P @ self.A.T + self.Q
        self.last_position = self.x[:2]
        return self.last_position

    def update(self, measurement):
        z = np.array(measurement)
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P
        self.last_position = self.x[:2]

def load_masks(folder):
    masks = {}
    files = sorted(os.listdir(folder))
    for f in files:
        if not f.endswith('.tif'):
            continue
        parts = f.split('_')
        tp = None
        for part in parts:
            if part.startswith('T'):
                try:
                    tp = int(part[1:])
                    break
                except:
                    pass
        if tp is None:
            continue
        path = os.path.join(folder, f)
        mask = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if mask is None:
            continue
        if tp not in masks:
            masks[tp] = []
        masks[tp].append(mask)
    return masks

def merge_tiles(tiles):
    merged = np.zeros_like(tiles[0], dtype=np.int32)
    for tile in tiles:
        merged = np.maximum(merged, tile)
    return merged

def extract_cell_centers(mask):
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centers.append((cx, cy, cnt))
    return centers

def associate_detections_to_trackers(trackers, detections, max_dist=50):
    if len(trackers) == 0:
        return [], list(range(len(detections))), []

    cost = np.zeros((len(trackers), len(detections)), dtype=np.float32)
    for t, trk in enumerate(trackers):
        trk_pos = trk.predict()
        for d, det in enumerate(detections):
            det_pos = det[:2]
            cost[t, d] = np.linalg.norm(trk_pos - det_pos)

    row_ind, col_ind = linear_sum_assignment(cost)

    unmatched_trackers = list(range(len(trackers)))
    unmatched_detections = list(range(len(detections)))
    matches = []

    for r, c in zip(row_ind, col_ind):
        if cost[r, c] > max_dist:
            continue
        matches.append((r, c))
        unmatched_trackers.remove(r)
        unmatched_detections.remove(c)

    return matches, unmatched_detections, unmatched_trackers

def get_color_for_id(track_id, colormap):
    # Cycle through colormap based on track_id
    idx = (track_id - 1) % colormap.shape[0]
    color = tuple(int(c) for c in colormap[idx])
    return color

def draw_tracks(mask, tracks, frame_number):
    vis = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    colormap = plt.colormaps.get_cmap('tab20')
    colors = (colormap(np.linspace(0, 1, 50))[:, :3] * 255).astype(np.uint8)  # 50 discrete colors

    for track_id, tracker in tracks.items():
        x, y = tracker.last_position
        w, h = 20, 20
        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        idx = (track_id - 1) % colors.shape[0]
        color = tuple(int(c) for c in colors[idx])  # tuple for OpenCV

        cv2.rectangle(vis, top_left, bottom_right, color, 2)
        cv2.putText(vis, str(track_id), (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(vis, f"Frame {frame_number}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    return vis

def track_cells(folder):
    masks_by_tp = load_masks(folder)
    all_timepoints = sorted(masks_by_tp.keys())

    tracks = {}  # track_id -> KalmanCellTracker
    next_track_id = 1

    accuracy_history = []
    id_switch_history = []
    timepoints = []

    example_mask = merge_tiles(masks_by_tp[all_timepoints[0]])
    h, w = example_mask.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('tracked_cells.mp4', fourcc, 3, (w, h))

    prev_track_ids = set()

    for tp in all_timepoints:
        merged_mask = merge_tiles(masks_by_tp[tp])
        detections = extract_cell_centers(merged_mask)

        tracker_list = list(tracks.values())
        matches, unmatched_dets, unmatched_trks = associate_detections_to_trackers(tracker_list, detections)

        matched_track_ids = set()
        for trk_idx, det_idx in matches:
            det_pos = detections[det_idx][:2]
            tracker_list[trk_idx].update(det_pos)
            matched_track_ids.add(list(tracks.keys())[trk_idx])

        new_ids = []
        for det_idx in unmatched_dets:
            det_pos = detections[det_idx][:2]
            tracks[next_track_id] = KalmanCellTracker(det_pos)
            new_ids.append(next_track_id)
            next_track_id += 1

        if len(prev_track_ids) == 0:
            accuracy = 100.0
        else:
            accuracy = (len(prev_track_ids.intersection(matched_track_ids)) / len(prev_track_ids)) * 100
        accuracy_history.append(accuracy)

        id_switch_history.append(len(new_ids))

        prev_track_ids = set(tracks.keys())
        timepoints.append(tp)

        vis_frame = draw_tracks(merged_mask, tracks, tp)
        out.write(vis_frame)

    out.release()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(timepoints, accuracy_history, marker='o', color='green')
    plt.title('Tracking Accuracy Over Timepoints')
    plt.xlabel('Timepoint')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 110)

    plt.subplot(1, 2, 2)
    plt.plot(timepoints, id_switch_history, marker='o', color='red')
    plt.title('ID Switches (New IDs) Over Timepoints')
    plt.xlabel('Timepoint')
    plt.ylabel('Number of New IDs')

    plt.tight_layout()
    plt.savefig('tracking-stand.png')
    plt.show()




if __name__ == "__main__":
    folder_path = "/gladstone/finkbeiner/steve/WeiyiLiu/GXYTMP/WL-NGN2-TauEos-OPL-2/MontagedImages/D3"  # Path to your montage mask folder
    track_cells(folder_path)
