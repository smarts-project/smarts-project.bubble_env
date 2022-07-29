import math
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio import types as t

traffic_histories = [
    t.TrafficHistoryDataset(
        name=f"i80_{start}-{end}",
        source_type="NGSIM",
        input_path=f"../../xy-trajectories/i80/vehicle-trajectory-data/{start}pm-{end}pm/trajectories-{start}-{end}.txt",
        x_margin_px=60.0,
        swap_xy=True,
        flip_y=True,
        filter_off_map=True,
        speed_limit_mps=28,
        heading_inference_window=5,
        max_angular_velocity=4,
        default_heading=1.5 * math.pi,
    )
    for start, end in [("0400", "0415"), ("0500", "0515"), ("0515", "0530")]
]

gen_scenario(
    t.Scenario(traffic_histories=traffic_histories), output_dir=Path(__file__).parent
)
