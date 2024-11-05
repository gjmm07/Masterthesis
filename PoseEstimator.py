import asyncio
import typing
import numpy.typing as npt
from operator import itemgetter
import cv2
import numpy as np
import torch
from collections import deque
from scipy.spatial.transform import Rotation as R


import mmcv
from mmdet.apis import inference_detector, init_detector
from mmhuman3d.utils.demo_utils import process_mmdet_results, convert_verts_to_cam_coord
from mmhuman3d.apis import inference_image_based_model, init_model
from mmhuman3d.utils.transforms import rotmat_to_aa, rotmat_to_ee, rotmat_to_quat
from mmhuman3d.models.body_models.builder import build_body_model
from mmhuman3d.core.renderer.mpr_renderer.smpl_realrender import VisualizerMeshSMPL


det_model = init_detector(
    "mmdetection/ssdlite_mobilenetv2_scratch_600e_coco.py",
    "mmdetection/ssdlite_mobilenetv2_scratch_600e_coco_20210629_110627-974d9307.pth",
    device="cuda:0"
)


mesh_model, extractor = init_model(
    "configs/expose/expose.py",
    "data/checkpoints/expose.pth",
    device="cuda:0")

body_model = build_body_model(
        dict(
            type="smplx",
            gender='neutral',
            num_betas=10,
            model_path="data/body_models/smplx"))


JOINT_NAMES = ['pelvis',
               'left_hip',
               'right_hip',
               'spine1',
               'left_knee',
               'right_knee',
               'spine2',
               'left_ankle',
               'right_ankle',
               'spine3',
               'left_foot',
               'right_foot',
               'neck',
               'left_collar',
               'right_collar',
               'head',
               'left_shoulder',
               'right_shoulder',
               'left_elbow',
               'right_elbow',
               'left_wrist',
               'right_wrist',
               'jaw',
               'left_eye',
               'right_eye',
               'left_index1',
               'left_index2',
               'left_index3',
               'left_middle1',
               'left_middle2',
               'left_middle3',
               'left_pinky1',
               'left_pinky2',
               'left_pinky3',
               'left_ring1',
               'left_ring2',
               'left_ring3',
               'left_thumb1',
               'left_thumb2',
               'left_thumb3',
               'right_index1',
               'right_index2',
               'right_index3',
               'right_middle1',
               'right_middle2',
               'right_middle3',
               'right_pinky1',
               'right_pinky2',
               'right_pinky3',
               'right_ring1',
               'right_ring2',
               'right_ring3',
               'right_thumb1',
               'right_thumb2',
               'right_thumb3']


def _calc_elbow_angle(rm_upper, rm_lower):
    r_rel = np.dot(rm_upper.T, rm_lower)
    x_axis_elbow = r_rel[:, 0]
    x_axis_shoulder = np.array([1, 0, 0])
    angle_rad = np.arccos(np.dot(x_axis_shoulder, x_axis_elbow) / (np.linalg.norm(x_axis_shoulder) * np.linalg.norm(x_axis_elbow)))
    return np.degrees(angle_rad)


class HumanPoseEstimator:

    def __init__(self,
                 current_position: deque,
                 stop_event: asyncio.Event,
                 cam_id: int = 0,
                 ):
        self._frame_queue: asyncio.Queue[np.ndarray] = asyncio.Queue(maxsize=1)
        self._result_queue: asyncio.Queue[tuple[np.ndarray, dict]] = asyncio.Queue(maxsize=1)
        self._cam_id = cam_id
        resolution = self._get_cam_resolution()
        self._renderer = VisualizerMeshSMPL(
            device="cuda:0", body_models=body_model, resolution=resolution)
        # self._renderer.load_vertices_filter("rightHand", "rightForeArm", "rightArm")
        self._stop_event: asyncio.Event = stop_event
        self._current_position = current_position
        self._vid_cap = cv2.VideoCapture(self._cam_id)

    def _get_cam_resolution(self):
        vid_cap = cv2.VideoCapture(self._cam_id)
        _, frame = vid_cap.read()
        vid_cap.release()
        return frame.shape[:2]

    async def _cam_reader(self):
        if not self._vid_cap.isOpened():
            return
        while not self._stop_event.is_set():
            ret, frame = self._vid_cap.read()
            if not ret:
                break
            await asyncio.sleep(0.05)
            await self._frame_queue.put(frame)
        self._vid_cap.release()

    async def _inference_detection(self):
        while not self._stop_event.is_set():
            frame = await self._frame_queue.get()
            det_results = inference_detector(det_model, frame)
            det_results = process_mmdet_results(det_results, cat_id=1, bbox_thr=0.6)
            mesh_results = inference_image_based_model(
                mesh_model,
                frame,
                det_results,
                bbox_thr=0.6,
                format='xyxy')
            try:
                await self._result_queue.put((frame, mesh_results[0]))
            except IndexError:
                print("No human detected!")
                continue
            mesh_results = mesh_results[0]["param"]
            full_pose = np.concatenate((
                (mesh_results["global_orient"]).cpu().numpy(),
                (mesh_results["body_pose"]).cpu().numpy(),
                (mesh_results["jaw_pose"]).cpu().numpy(),
                np.zeros((2, 3, 3)),
                (mesh_results["left_hand_pose"]).cpu().numpy(),
                (mesh_results["right_hand_pose"].cpu()).numpy()
            ))
            self._current_position.append(_calc_elbow_angle(full_pose[17], full_pose[19]))
            # self._current_position.append(calc_elbow_angle(full_pose[21], full_pose[19]))


    async def _display(self):
        while not self._stop_event.is_set():
            frame, result = await self._result_queue.get()
            pred_cams = result['camera']
            verts = result['vertices']
            bboxes_xyxy = result['bbox']
            verts, _ = convert_verts_to_cam_coord(
                verts, pred_cams, bboxes_xyxy, focal_length=5000.)

            mmcv.imshow_bboxes(
                frame,
                bboxes_xyxy[None],
                colors='green',
                top_k=-1,
                thickness=2,
                show=False)

            # visualize smpl
            if isinstance(verts, np.ndarray):
                verts = torch.tensor(verts).to("cuda:0").squeeze()
            frame = self._renderer(verts, frame, alpha=0.9)
            cv2.imshow('mmhuman3d webcam', frame)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

    def stop(self):
        cv2.destroyAllWindows()
        self._stop_event.set()
        self._vid_cap.release()

    def __await__(self):
        yield from asyncio.gather(
            self._cam_reader(),
            self._inference_detection(),
            self._display()).__await__()




