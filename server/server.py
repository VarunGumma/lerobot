from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from huggingface_hub.constants import CONFIG_NAME, SAFETENSORS_SINGLE_FILE
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.utils.constants import (
    ACTION,
    CHECKPOINTS_DIR,
    LAST_CHECKPOINT_LINK,
    OBS_STATE,
    PRETRAINED_MODEL_DIR,
)

DEFAULT_ROBOTWIN_CAMERA_ORDER = ("cam_high", "cam_right_wrist", "cam_left_wrist")
INTERNAL_EEF_DIM = 14
ROBOTWIN_EEF_DIM = 16
ROBOTWIN_CAMERA_ALIASES = {
    "head_camera": "cam_high",
    "right_camera": "cam_right_wrist",
    "left_camera": "cam_left_wrist",
}


class InitRequest(BaseModel):
    model_path: str | None = None
    checkpoint_path: str | None = None
    pi0_step: int | None = None
    actions_per_chunk: int | None = None
    device: str | None = None
    camera_order: list[str] | None = None
    eef_input_quat_order: str = "xyzw"
    eef_output_quat_order: str = "xyzw"


class LanguageRequest(BaseModel):
    instruction: str


def _camera_name_alias(name: str) -> str:
    return ROBOTWIN_CAMERA_ALIASES.get(name, name)


def _feature_suffix(feature_key: str) -> str:
    return feature_key.rsplit(".", maxsplit=1)[-1]


def _as_xyzw(quat: np.ndarray, quat_order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).ravel()
    if quat.shape[0] != 4:
        raise ValueError(f"Expected quaternion with 4 values, got shape {quat.shape}.")
    if quat_order == "xyzw":
        return quat
    if quat_order == "wxyz":
        return quat[[1, 2, 3, 0]]
    raise ValueError(f"Unsupported quaternion order '{quat_order}'. Use 'xyzw' or 'wxyz'.")


def _from_xyzw(quat: np.ndarray, quat_order: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).ravel()
    if quat.shape[0] != 4:
        raise ValueError(f"Expected quaternion with 4 values, got shape {quat.shape}.")
    if quat_order == "xyzw":
        return quat
    if quat_order == "wxyz":
        return quat[[3, 0, 1, 2]]
    raise ValueError(f"Unsupported quaternion order '{quat_order}'. Use 'xyzw' or 'wxyz'.")


def robotwin_eef16_to_policy_eef14(state: np.ndarray, quat_order: str) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32).ravel()
    if state.shape[0] != ROBOTWIN_EEF_DIM:
        raise ValueError(f"Expected 16D EEF state, got shape {state.shape}.")

    left = state[:8]
    right = state[8:]
    left_euler = Rotation.from_quat(_as_xyzw(left[3:7], quat_order)).as_euler("xyz").astype(np.float32)
    right_euler = Rotation.from_quat(_as_xyzw(right[3:7], quat_order)).as_euler("xyz").astype(np.float32)
    return np.concatenate(
        [
            left[:3],
            left_euler,
            left[7:8],
            right[:3],
            right_euler,
            right[7:8],
        ]
    ).astype(np.float32)


def policy_eef14_to_robotwin_eef16(action: np.ndarray, quat_order: str) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32).ravel()
    if action.shape[0] != INTERNAL_EEF_DIM:
        raise ValueError(f"Expected 14D EEF action, got shape {action.shape}.")

    left = action[:7]
    right = action[7:]
    left_quat = _from_xyzw(Rotation.from_euler("xyz", left[3:6]).as_quat(), quat_order)
    right_quat = _from_xyzw(Rotation.from_euler("xyz", right[3:6]).as_quat(), quat_order)
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


def _is_pretrained_model_dir(path: Path) -> bool:
    return (path / CONFIG_NAME).is_file() and (path / SAFETENSORS_SINGLE_FILE).is_file()


def resolve_pretrained_model_path(model_or_checkpoint_path: str) -> str:
    """Resolve a LeRobot run/checkpoint path to the pretrained_model directory.

    Accepted local forms:
    - /run/checkpoints/005000/pretrained_model
    - /run/checkpoints/005000
    - /run/checkpoints/last
    - /run

    Hub model ids are returned unchanged.
    """
    path = Path(model_or_checkpoint_path).expanduser()
    if not path.exists():
        return model_or_checkpoint_path

    path = path.resolve()
    if _is_pretrained_model_dir(path):
        return str(path)

    candidate = path / PRETRAINED_MODEL_DIR
    if _is_pretrained_model_dir(candidate):
        return str(candidate)

    candidate = path / CHECKPOINTS_DIR / LAST_CHECKPOINT_LINK / PRETRAINED_MODEL_DIR
    if _is_pretrained_model_dir(candidate):
        return str(candidate.resolve())

    checkpoint_root = path / CHECKPOINTS_DIR
    if checkpoint_root.is_dir():
        checkpoint_dirs = sorted(
            checkpoint
            for checkpoint in checkpoint_root.iterdir()
            if checkpoint.is_dir() and checkpoint.name != LAST_CHECKPOINT_LINK
        )
        for checkpoint in reversed(checkpoint_dirs):
            candidate = checkpoint / PRETRAINED_MODEL_DIR
            if _is_pretrained_model_dir(candidate):
                return str(candidate.resolve())

    raise FileNotFoundError(
        "Could not find a LeRobot pretrained model under "
        f"'{model_or_checkpoint_path}'. Expected either a directory containing "
        f"{CONFIG_NAME} and {SAFETENSORS_SINGLE_FILE}, a checkpoint directory "
        f"containing {PRETRAINED_MODEL_DIR}/, or a run directory containing "
        f"{CHECKPOINTS_DIR}/{LAST_CHECKPOINT_LINK}/{PRETRAINED_MODEL_DIR}/."
    )


def _image_to_chw_float01(image: np.ndarray) -> torch.Tensor:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.ndim != 3:
        raise ValueError(f"Expected image with shape (H, W, C), got {arr.shape}")
    if arr.shape[-1] < 3:
        raise ValueError(f"Expected at least 3 image channels, got {arr.shape}")
    if arr.shape[-1] > 3:
        arr = arr[..., :3]

    arr = np.ascontiguousarray(arr)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).to(dtype=torch.float32)
    if arr.dtype == np.uint8 or float(tensor.max()) > 1.5:
        tensor = tensor / 255.0
    return tensor


class LeRobotPolicyHost:
    def __init__(
        self,
        model_path: str,
        *,
        actions_per_chunk: int | None = None,
        device: str | None = None,
        camera_order: list[str] | None = None,
        eef_input_quat_order: str = "xyzw",
        eef_output_quat_order: str = "xyzw",
    ) -> None:
        self.requested_model_path = model_path
        self.model_path = resolve_pretrained_model_path(model_path)
        self.config = PreTrainedConfig.from_pretrained(self.model_path)
        if device is not None:
            self.config.device = device
        self.device = self.config.device

        policy_cls = get_policy_class(self.config.type)
        print(
            f"[LeRobotPolicyHost] Loading {self.config.type} policy from {self.model_path} "
            f"on {self.device}"
        )
        self.policy = policy_cls.from_pretrained(self.model_path, config=self.config)
        self.policy.eval()

        self.preprocess, self.postprocess = make_pre_post_processors(
            self.policy.config,
            pretrained_path=self.model_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        self.image_keys = list(self.policy.config.image_features.keys())
        if not self.image_keys:
            raise ValueError("Loaded policy has no image features; this server expects visual Pi0-family policies.")

        self.camera_order = tuple(camera_order or DEFAULT_ROBOTWIN_CAMERA_ORDER)
        self.eef_input_quat_order = eef_input_quat_order
        self.eef_output_quat_order = eef_output_quat_order
        self.actions_per_chunk = actions_per_chunk
        self.instruction: str | None = None
        self._raw_obs: dict[str, Any] | None = None
        self._return_robotwin_eef_actions = False
        self.policy.reset()

        print(f"[LeRobotPolicyHost] Policy type: {self.config.type}")
        print(f"[LeRobotPolicyHost] Image features: {self.image_keys}")
        print(f"[LeRobotPolicyHost] Incoming camera order: {self.camera_order}")
        print(
            "[LeRobotPolicyHost] 16D EEF quaternion orders: "
            f"input={self.eef_input_quat_order}, output={self.eef_output_quat_order}"
        )

    def set_language(self, instruction: str) -> None:
        self.instruction = instruction
        print(f"[LeRobotPolicyHost] Instruction set: {instruction}")

    def _model_key_for_camera(self, camera_name: str, fallback_index: int) -> str:
        camera_name = _camera_name_alias(camera_name)
        for key in self.image_keys:
            if key == camera_name or _feature_suffix(key) == camera_name:
                return key

        if fallback_index < len(self.image_keys):
            return self.image_keys[fallback_index]

        raise ValueError(
            f"Could not map incoming camera '{camera_name}' to model image keys {self.image_keys}"
        )

    def _state_for_policy(self, state: np.ndarray) -> np.ndarray:
        state_np = np.asarray(state, dtype=np.float32).ravel()
        expected_state_dim = self.policy.config.robot_state_feature.shape[0]
        self._return_robotwin_eef_actions = False

        if state_np.shape[0] == expected_state_dim:
            return state_np

        if expected_state_dim == INTERNAL_EEF_DIM and state_np.shape[0] == ROBOTWIN_EEF_DIM:
            self._return_robotwin_eef_actions = True
            return robotwin_eef16_to_policy_eef14(state_np, self.eef_input_quat_order)

        raise ValueError(
            f"Expected state dim {expected_state_dim} for {self.config.type}, or 16D EEF "
            f"for a 14D EEF checkpoint, got {state_np.shape[0]}."
        )

    def update_observation_window(self, img_arr: list[np.ndarray], state: np.ndarray) -> None:
        if self.instruction is None:
            raise RuntimeError("Call set_language() before update_observation_window().")
        if len(img_arr) != len(self.camera_order):
            raise ValueError(
                f"Expected {len(self.camera_order)} images for camera_order={self.camera_order}, "
                f"got {len(img_arr)}."
            )

        frame: dict[str, Any] = {}
        for index, (camera_name, image) in enumerate(zip(self.camera_order, img_arr, strict=True)):
            key = self._model_key_for_camera(camera_name, index)
            frame[key] = _image_to_chw_float01(image)

        state_np = self._state_for_policy(state)

        frame[OBS_STATE] = torch.from_numpy(state_np)
        frame["task"] = self.instruction
        self._raw_obs = frame

    def get_action(self) -> np.ndarray:
        if self._raw_obs is None:
            raise RuntimeError("Call update_observation_window() before get_action().")

        batch = self.preprocess(self._raw_obs)
        with torch.inference_mode():
            action_tensor = self.policy.predict_action_chunk(batch)

        if action_tensor.ndim == 2:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.ndim != 3:
            raise RuntimeError(f"Expected action chunk with shape (B, T, D), got {tuple(action_tensor.shape)}")

        available_chunk = action_tensor.shape[1]
        action_count = self.actions_per_chunk or getattr(self.policy.config, "n_action_steps", available_chunk)
        action_count = min(action_count, available_chunk)

        processed_actions = []
        for index in range(action_count):
            single_action = action_tensor[:, index, :]
            action = self.postprocess(single_action)
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            action = np.asarray(action, dtype=np.float32)
            if action.ndim == 2 and action.shape[0] == 1:
                action = action[0]
            processed_actions.append(action.reshape(-1))

        actions = np.stack(processed_actions, axis=0)
        expected_action_dim = self.policy.config.output_features[ACTION].shape[0]
        if actions.shape[-1] != expected_action_dim:
            raise RuntimeError(
                f"Expected postprocessed action dim {expected_action_dim}, got {actions.shape[-1]}."
            )
        if self._return_robotwin_eef_actions:
            actions = np.stack(
                [policy_eef14_to_robotwin_eef16(action, self.eef_output_quat_order) for action in actions],
                axis=0,
            )
        return actions

    def reset_observation_windows(self) -> None:
        self.instruction = None
        self._raw_obs = None
        self.policy.reset()
        print("[LeRobotPolicyHost] Reset complete.")


app = FastAPI()
model: LeRobotPolicyHost | None = None


def _require_model() -> LeRobotPolicyHost:
    if model is None:
        raise HTTPException(status_code=400, detail="Model is not initialized. Call /init first.")
    return model


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/init")
async def init_model(payload: InitRequest):
    global model
    requested_path = payload.model_path or payload.checkpoint_path
    if requested_path is None:
        raise HTTPException(status_code=400, detail="Specify either model_path or checkpoint_path.")

    action_count = payload.actions_per_chunk if payload.actions_per_chunk is not None else payload.pi0_step
    try:
        model = LeRobotPolicyHost(
            model_path=requested_path,
            actions_per_chunk=action_count,
            device=payload.device,
            camera_order=payload.camera_order,
            eef_input_quat_order=payload.eef_input_quat_order,
            eef_output_quat_order=payload.eef_output_quat_order,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "ok",
        "policy_type": model.config.type,
        "device": model.device,
        "model_path": model.model_path,
        "requested_model_path": model.requested_model_path,
        "actions_per_chunk": model.actions_per_chunk,
        "eef_input_quat_order": model.eef_input_quat_order,
        "eef_output_quat_order": model.eef_output_quat_order,
    }


@app.post("/set_language")
async def set_language(payload: LanguageRequest):
    _require_model().set_language(payload.instruction)
    return {"status": "ok"}


@app.post("/update_obs")
async def update_obs(request: Request):
    raw_data = await request.body()
    data = pickle.loads(raw_data)  # nosec: local RoboTwin deployment channel.
    host = _require_model()
    if "instruction" in data and data["instruction"] is not None:
        host.set_language(data["instruction"])
    host.update_observation_window(data["img_arr"], data["state"])
    return Response(content=pickle.dumps({"status": "ok"}))


@app.post("/get_action")
async def get_action():
    actions = _require_model().get_action()
    return Response(content=pickle.dumps(actions))


@app.post("/reset")
async def reset_model():
    _require_model().reset_observation_windows()
    return {"status": "ok"}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    print(f"Starting LeRobot policy server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
