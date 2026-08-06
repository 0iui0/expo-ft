import glob
import os

import h5py
import numpy as np
from tqdm import tqdm

def _discover_episode_dirs(base_path):
    dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    # Keep only numeric directory names (episode indices); skip e.g. action_videos, lerobot
    dirs = [d for d in dirs if d.isdigit()]
    dirs = sorted(dirs, key=lambda x: int(x))
    return [os.path.join(base_path, d) for d in dirs]

def process_droid_dataset(
    datapath, 
    task_config, 
    episode_indices = None,
    num_data = None):
    ep_dirs = _discover_episode_dirs(datapath)
    if episode_indices is not None:
        ep_dirs = [ep_dirs[i] for i in episode_indices if 0 <= i < len(ep_dirs)]
    elif num_data is not None and num_data > 0:
        ep_dirs = ep_dirs[:num_data]
    
    print(f"Find {len(_discover_episode_dirs(datapath))} episodes; using {len(ep_dirs)}")

    data = []
    for ep in tqdm(ep_dirs):
        with h5py.File(os.path.join(ep, "traj.hdf5"), "r") as f:
            def load_recursive(group):
                result = {}
                for k, v in group.items():
                    if isinstance(v, h5py.Group):
                        result[k] = load_recursive(v)
                    else:
                        arr = np.asarray(v)
                        # Decode bytes to strings (h5py stores strings as bytes)
                        if arr.dtype.kind == 'S':
                            result[k] = arr.astype('U')
                        elif arr.dtype == object and arr.size > 0 and isinstance(arr.flat[0], bytes):
                            result[k] = np.array([s.decode('utf-8') for s in arr.flat]).reshape(arr.shape)
                        else:
                            result[k] = arr
                return result
            
            ep_obs = load_recursive(f["saved_observation"])
            
            action_key = task_config.action_space
            gripper_key = f"gripper_{task_config.gripper_action_space}"
            a1 = np.asarray(f["action"][action_key])
            a2 = np.asarray(f["action"][gripper_key])
            ep_actions = np.concatenate([a1, a2[:, None] if len(a2.shape) == 1 else a2], axis=-1)
            
            T = len(ep_actions)
            ep_dones = np.pad(np.array([1.0], dtype=np.float32), (T-1, 0), constant_values=0)
            ep_rewards = np.pad(np.array([1.0], dtype=np.float32), (T-1, 0), constant_values=0)
            
            def extract_t(obs, t):
                return {k: extract_t(v, t) if isinstance(v, dict) else (v[t] if isinstance(v, np.ndarray) and len(v.shape) > 0 else v)
                       for k, v in obs.items()}
            
            for t in range(T):
                data.append({
                    "observations": extract_t(ep_obs, t),
                    "actions": ep_actions[t],
                    "rewards": ep_rewards[t],
                    "masks": 1 - ep_dones[t],
                    "dones": ep_dones[t]
                })

    return data


def process_cr5af_npz_pi05(
    datapath,
    prompt,
    episode_indices=None,
    num_data=None,
):
    """Load CR5AF npz demos as PI0.5 RL offline-seed transitions.

    Reads the SAME npz episodes used for PI0.5 SFT (``episode_*/data.npz`` with
    ``state`` (T,16)=[xyz_mm(3), rot6d(6), joint(6), grip(1)], ``images_table``,
    ``images_hand``, ``gripper_states``) and emits transitions whose observations
    match ``LeRobotDROIDDataConfig(use_cartesian_state=True)`` repack: flat
    ``cartesian_position`` (9), ``gripper_position`` (1), ``exterior_image_1_left``
    / ``exterior_image_2_left`` / ``wrist_image_left`` (RGB), and ``prompt``.

    Actions are the ABSOLUTE next-pose 10D target ``[cartesian[t+1], gripper[t+1]]``
    (the config-side DeltaActions converts the xyz channels to delta at load time,
    AbsoluteActions rebuilds them at inference), identical to
    scripts/convert_cr5af_npz_to_lerobot.py so the RL seed matches the SFT prior.
    """
    ep_dirs = sorted(
        d for d in glob.glob(os.path.join(datapath, "episode_*")) if os.path.isdir(d)
    )
    if episode_indices is not None:
        ep_dirs = [ep_dirs[i] for i in episode_indices if 0 <= i < len(ep_dirs)]
    elif num_data is not None and num_data > 0:
        ep_dirs = ep_dirs[:num_data]

    print(f"Find {len(glob.glob(os.path.join(datapath, 'episode_*')))} npz episodes; using {len(ep_dirs)}")

    data = []
    for ep in tqdm(ep_dirs):
        d = np.load(os.path.join(ep, "data.npz"))
        state = np.asarray(d["state"], dtype=np.float32)  # (T, 16)
        grip = np.asarray(d["gripper_states"], dtype=np.float32).reshape(-1, 1)
        if np.allclose(grip.squeeze(), state[..., 15]):
            grip = state[..., 15:16].astype(np.float32)
        # RealSense stores BGR8; convert to RGB for PaliGemma (matches SFT converter).
        imgs_table = np.asarray(d["images_table"], dtype=np.uint8)[..., ::-1]
        imgs_hand = np.asarray(d["images_hand"], dtype=np.uint8)[..., ::-1]
        # 16D state -> 9D cartesian [xyz_m(3), rot6d(6)]: mm->m, native rot6d passthrough.
        cartesian = np.concatenate(
            [state[..., :3] * 0.001, state[..., 3:9]], axis=-1
        ).astype(np.float32)

        T = cartesian.shape[0]
        # ABSOLUTE next-pose action; pad the last frame with itself.
        next_cart = np.concatenate([cartesian[1:], cartesian[-1:]], axis=0)
        next_grip = np.concatenate([grip[1:], grip[-1:]], axis=0)
        actions = np.concatenate([next_cart, next_grip], axis=-1).astype(np.float32)  # (T, 10)

        dones = np.pad(np.array([1.0], dtype=np.float32), (T - 1, 0), constant_values=0)
        rewards = np.pad(np.array([1.0], dtype=np.float32), (T - 1, 0), constant_values=0)

        for t in range(T):
            data.append({
                "observations": {
                    "exterior_image_1_left": np.ascontiguousarray(imgs_table[t]),
                    "exterior_image_2_left": np.ascontiguousarray(imgs_table[t]),
                    "wrist_image_left": np.ascontiguousarray(imgs_hand[t]),
                    "cartesian_position": cartesian[t],
                    "gripper_position": grip[t],
                    "prompt": prompt,
                },
                "actions": actions[t],
                "rewards": rewards[t],
                "masks": 1 - dones[t],
                "dones": dones[t],
            })

    return data
