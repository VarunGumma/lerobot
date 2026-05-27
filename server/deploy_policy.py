from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import requests
from scipy.spatial.transform import Rotation

ROBOTWIN_CAMERA_ORDER = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
ACTION_DIM = 14


def _as_xyzw(quat: np.ndarray, quat_order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).ravel()
    if quat.shape[0] != 4:
        raise ValueError(f"Expected quaternion with 4 values, got shape {quat.shape}")
    if quat_order == "xyzw":
        return quat
    if quat_order == "wxyz":
        return quat[[1, 2, 3, 0]]
    raise ValueError(f"Unsupported quaternion order '{quat_order}'. Use 'xyzw' or 'wxyz'.")


def _from_xyzw(quat: np.ndarray, quat_order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).ravel()
    if quat.shape[0] != 4:
        raise ValueError(f"Expected quaternion with 4 values, got shape {quat.shape}")
    if quat_order == "xyzw":
        return quat
    if quat_order == "wxyz":
        return quat[[3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion order '{quat_order}'. Use 'xyzw' or 'wxyz'.")


def _quat_to_euler_xyz(quat: np.ndarray, quat_order: str) -> np.ndarray:
    return Rotation.from_quat(_as_xyzw(quat, quat_order)).as_euler("xyz").astype(np.float32)


def _euler_xyz_to_quat(euler: np.ndarray, quat_order: str) -> np.ndarray:
    quat_xyzw = Rotation.from_euler("xyz", np.asarray(euler, dtype=np.float32)).as_quat()
    return _from_xyzw(quat_xyzw, quat_order).astype(np.float32)


def _scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float32).ravel()
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def encode_eef_state(observation: dict[str, Any], quat_order: str) -> np.ndarray:
    endpose = observation.get("endpose") or {}
    try:
        left_pose = np.asarray(endpose["left_endpose"], dtype=np.float32).ravel()
        right_pose = np.asarray(endpose["right_endpose"], dtype=np.float32).ravel()
    except KeyError as exc:
        raise KeyError(
            "EEF checkpoints require observation['endpose'] with left_endpose, "
            "left_gripper, right_endpose, and right_gripper."
        ) from exc

    if left_pose.size < 7 or right_pose.size < 7:
        raise ValueError(
            "RoboTwin endpose entries must contain xyz + quaternion "
            f"(got left={left_pose.shape}, right={right_pose.shape})."
        )

    left = np.concatenate(
        [
            left_pose[:3],
            _quat_to_euler_xyz(left_pose[3:7], quat_order),
            np.array([_scalar(endpose.get("left_gripper", 0.0))], dtype=np.float32),
        ]
    )
    right = np.concatenate(
        [
            right_pose[:3],
            _quat_to_euler_xyz(right_pose[3:7], quat_order),
            np.array([_scalar(endpose.get("right_gripper", 0.0))], dtype=np.float32),
        ]
    )
    return np.concatenate([left, right]).astype(np.float32)


def eef_action_to_robotwin_action(action: np.ndarray, quat_order: str) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).ravel()
    if action.shape[0] != ACTION_DIM:
        raise ValueError(f"Expected EEF action of shape ({ACTION_DIM},), got {action.shape}")

    left = action[:7]
    right = action[7:]
    left_quat = _euler_xyz_to_quat(left[3:6], quat_order)
    right_quat = _euler_xyz_to_quat(right[3:6], quat_order)
    return np.concatenate(
        [
            left[:3],
            left_quat,
            left[6:7],
            right[:3],
            right_quat,
            right[6:7],
        ]
    ).astype(np.float32)


def encode_obs(
    observation: dict[str, Any],
    *,
    action_type: str,
    eef_obs_quat_order: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]

    if action_type == "ee":
        input_state = encode_eef_state(observation, quat_order=eef_obs_quat_order)
    elif action_type == "qpos":
        input_state = np.asarray(observation["joint_action"]["vector"], dtype=np.float32).ravel()
    else:
        raise ValueError(f"Unsupported action_type '{action_type}'. Use 'qpos' or 'ee'.")

    return input_rgb_arr, input_state


class PI0LeRobotClient:
    def __init__(
        self,
        server_url: str,
        model_path: str | None = None,
        *,
        checkpoint_path: str | None = None,
        pi0_step: int = 50,
        device: str = "cuda",
        action_type: str = "qpos",
        eef_obs_quat_order: str = "xyzw",
        eef_action_quat_order: str = "xyzw",
        camera_order: list[str] | None = None,
    ) -> None:
        if action_type not in ("qpos", "ee"):
            raise ValueError(f"Unsupported action_type '{action_type}'. Use 'qpos' or 'ee'.")
        if model_path is None and checkpoint_path is None:
            raise ValueError("Specify either model_path or checkpoint_path.")

        self.server_url = server_url.rstrip("/")
        self.action_type = action_type
        self.eef_obs_quat_order = eef_obs_quat_order
        self.eef_action_quat_order = eef_action_quat_order
        self.camera_order = camera_order or ROBOTWIN_CAMERA_ORDER
        self._raw_obs = None
        self.session = requests.Session()

        print(f"[PI0LeRobotClient] Initializing model on server: {self.server_url}")
        init_payload = {
            "pi0_step": pi0_step,
            "device": device,
            "camera_order": self.camera_order,
        }
        if checkpoint_path is not None:
            init_payload["checkpoint_path"] = checkpoint_path
        else:
            init_payload["model_path"] = model_path

        resp = self.session.post(
            f"{self.server_url}/init",
            json=init_payload,
            timeout=None,
        )
        resp.raise_for_status()
        print("[PI0LeRobotClient] Connected and initialized.")

    def set_language(self, instruction: str) -> None:
        self.session.post(
            f"{self.server_url}/set_language",
            json={"instruction": instruction},
            timeout=None,
        ).raise_for_status()

    def update_observation_window(self, img_arr: list[np.ndarray], state: np.ndarray) -> None:
        data = pickle.dumps({"img_arr": img_arr, "state": state})
        self.session.post(f"{self.server_url}/update_obs", data=data, timeout=None).raise_for_status()
        self._raw_obs = True

    def get_action(self) -> np.ndarray:
        resp = self.session.post(f"{self.server_url}/get_action", timeout=None)
        resp.raise_for_status()
        return pickle.loads(resp.content)  # nosec: local RoboTwin deployment channel.

    def reset_obsrvationwindows(self) -> None:
        self.session.post(f"{self.server_url}/reset", timeout=None).raise_for_status()
        self._raw_obs = None


def get_model(usr_args: dict[str, Any]) -> PI0LeRobotClient:
    model_path = usr_args.get("model_path")
    checkpoint_path = usr_args.get("checkpoint_path")
    if model_path is None and checkpoint_path is None:
        raise ValueError("Specify either model_path or checkpoint_path.")

    pi0_step = int(usr_args.get("pi0_step", usr_args.get("actions_per_chunk", 50)))
    device = usr_args.get("device", "cuda")
    server_url = usr_args.get("server_url", "http://127.0.0.1:8000")
    action_type = usr_args.get("action_type", "qpos")
    eef_obs_quat_order = usr_args.get("eef_obs_quat_order", "xyzw")
    eef_action_quat_order = usr_args.get("eef_action_quat_order", "xyzw")
    camera_order = usr_args.get("camera_order", ROBOTWIN_CAMERA_ORDER)

    return PI0LeRobotClient(
        server_url=server_url,
        model_path=model_path,
        checkpoint_path=checkpoint_path,
        pi0_step=pi0_step,
        device=device,
        action_type=action_type,
        eef_obs_quat_order=eef_obs_quat_order,
        eef_action_quat_order=eef_action_quat_order,
        camera_order=camera_order,
    )


def _take_action(task_env, action: np.ndarray, model: PI0LeRobotClient) -> None:
    if model.action_type == "ee":
        robotwin_action = eef_action_to_robotwin_action(action, model.eef_action_quat_order)
        task_env.take_action(robotwin_action, action_type="ee")
        return

    try:
        task_env.take_action(action, action_type="qpos")
    except TypeError:
        task_env.take_action(action)


def eval(task_env, model: PI0LeRobotClient, observation: dict[str, Any]) -> None:
    if model._raw_obs is None:
        instruction = task_env.get_instruction()
        model.set_language(instruction)

    input_rgb_arr, input_state = encode_obs(
        observation,
        action_type=model.action_type,
        eef_obs_quat_order=model.eef_obs_quat_order,
    )
    model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()

    for action in actions:
        _take_action(task_env, action, model)
        observation = task_env.get_obs()
        input_rgb_arr, input_state = encode_obs(
            observation,
            action_type=model.action_type,
            eef_obs_quat_order=model.eef_obs_quat_order,
        )
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model: PI0LeRobotClient) -> None:
    model.reset_obsrvationwindows()
