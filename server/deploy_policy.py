from __future__ import annotations

import pickle
from typing import Any

import numpy as np
import requests

ROBOTWIN_CAMERA_ORDER = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
EEF_ACTION_DIM = 16


def _scalar(value: Any) -> float:
    arr = np.asarray(value, dtype=np.float32).ravel()
    if arr.size == 0:
        return 0.0
    return float(arr[0])


def encode_eef_state(observation: dict[str, Any]) -> np.ndarray:
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
        [left_pose[:7], np.array([_scalar(endpose.get("left_gripper", 0.0))], dtype=np.float32)]
    )
    right = np.concatenate(
        [right_pose[:7], np.array([_scalar(endpose.get("right_gripper", 0.0))], dtype=np.float32)]
    )
    return np.concatenate([left, right]).astype(np.float32)


def validate_eef_action(action: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).ravel()
    if action.shape[0] != EEF_ACTION_DIM:
        raise ValueError(f"Expected 16D EEF action, got {action.shape}")
    return action


def encode_obs(
    observation: dict[str, Any],
    *,
    action_type: str,
) -> tuple[list[np.ndarray], np.ndarray]:
    input_rgb_arr = [
        observation["observation"]["head_camera"]["rgb"],
        observation["observation"]["right_camera"]["rgb"],
        observation["observation"]["left_camera"]["rgb"],
    ]

    if action_type == "ee":
        input_state = encode_eef_state(observation)
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
            "eef_input_quat_order": eef_obs_quat_order,
            "eef_output_quat_order": eef_action_quat_order,
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
        task_env.take_action(validate_eef_action(action), action_type="ee")
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
    )
    model.update_observation_window(input_rgb_arr, input_state)

    actions = model.get_action()

    for action in actions:
        _take_action(task_env, action, model)
        observation = task_env.get_obs()
        input_rgb_arr, input_state = encode_obs(
            observation,
            action_type=model.action_type,
        )
        model.update_observation_window(input_rgb_arr, input_state)


def reset_model(model: PI0LeRobotClient) -> None:
    model.reset_obsrvationwindows()
