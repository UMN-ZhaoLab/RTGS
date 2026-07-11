import torch
import torch.nn.functional as F
from torch import nn

from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from utils.slam_utils import image_gradient, image_gradient_mask


class ScaledCameraView:
    """Low-resolution view for tracking; shares pose parameters with base camera."""

    def __init__(self, base_camera, scale: float):
        self._base = base_camera
        H = max(2, int(round(base_camera.image_height * scale)))
        W = max(2, int(round(base_camera.image_width * scale)))
        H -= H % 2
        W -= W % 2
        s = H / base_camera.image_height

        self.uid = base_camera.uid
        self.device = base_camera.device
        self.image_height = H
        self.image_width = W
        self.fx = base_camera.fx * s
        self.fy = base_camera.fy * s
        self.cx = base_camera.cx * s
        self.cy = base_camera.cy * s
        self.FoVx = base_camera.FoVx
        self.FoVy = base_camera.FoVy

        self.projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            W=W,
            H=H,
        ).transpose(0, 1).to(device=base_camera.device)

        self.original_image = F.interpolate(
            base_camera.original_image.unsqueeze(0),
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if base_camera.depth is not None:
            depth_t = (
                torch.from_numpy(base_camera.depth)
                .float()
                .to(base_camera.device)
                .unsqueeze(0)
                .unsqueeze(0)
            )
            self.depth = (
                F.interpolate(depth_t, size=(H, W), mode="nearest")
                .squeeze()
                .cpu()
                .numpy()
            )
        else:
            self.depth = None

        if base_camera.grad_mask is not None:
            grad_mask = base_camera.grad_mask.float()
            if grad_mask.ndim == 2:
                grad_mask = grad_mask.unsqueeze(0)
            self.grad_mask = (
                F.interpolate(
                    grad_mask.unsqueeze(0),
                    size=(H, W),
                    mode="nearest",
                )
                .squeeze(0)
                .bool()
            )
        else:
            self.grad_mask = None

    @property
    def R(self):
        return self._base.R

    @property
    def T(self):
        return self._base.T

    @property
    def cam_rot_delta(self):
        return self._base.cam_rot_delta

    @property
    def cam_trans_delta(self):
        return self._base.cam_trans_delta

    @property
    def exposure_a(self):
        return self._base.exposure_a

    @property
    def exposure_b(self):
        return self._base.exposure_b

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]


class Camera(nn.Module):
    def __init__(
        self,
        uid,
        color,
        depth,
        gt_T,
        projection_matrix,
        fx,
        fy,
        cx,
        cy,
        fovx,
        fovy,
        image_height,
        image_width,
        device="cuda:0",
    ):
        super(Camera, self).__init__()
        self.uid = uid
        self.device = device

        T = torch.eye(4, device=device)
        self.R = T[:3, :3]
        self.T = T[:3, 3]
        self.R_gt = gt_T[:3, :3]
        self.T_gt = gt_T[:3, 3]

        self.original_image = color
        self.depth = depth
        self.grad_mask = None

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.FoVx = fovx
        self.FoVy = fovy
        self.image_height = image_height
        self.image_width = image_width

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=device)
        )

        self.projection_matrix = projection_matrix.to(device=device)

    @staticmethod
    def init_from_dataset(dataset, idx, projection_matrix):
        gt_color, gt_depth, gt_pose = dataset[idx]
        return Camera(
            idx,
            gt_color,
            gt_depth,
            gt_pose,
            projection_matrix,
            dataset.fx,
            dataset.fy,
            dataset.cx,
            dataset.cy,
            dataset.fovx,
            dataset.fovy,
            dataset.height,
            dataset.width,
            device=dataset.device,
        )

    @staticmethod
    def init_from_gui(uid, T, FoVx, FoVy, fx, fy, cx, cy, H, W):
        projection_matrix = getProjectionMatrix2(
            znear=0.01, zfar=100.0, fx=fx, fy=fy, cx=cx, cy=cy, W=W, H=H
        ).transpose(0, 1)
        return Camera(
            uid, None, None, T, projection_matrix, fx, fy, cx, cy, FoVx, FoVy, H, W
        )

    @property
    def world_view_transform(self):
        return getWorld2View2(self.R, self.T).transpose(0, 1)

    @property
    def full_proj_transform(self):
        return (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)

    @property
    def camera_center(self):
        return self.world_view_transform.inverse()[3, :3]

    def update_RT(self, R, t):
        self.R = R.to(device=self.device)
        self.T = t.to(device=self.device)

    def make_scaled_view(self, scale: float):
        if scale >= 1.0:
            return self
        if not hasattr(self, "_scaled_view_cache"):
            self._scaled_view_cache = {}
        key = round(scale, 3)
        if key not in self._scaled_view_cache:
            self._scaled_view_cache[key] = ScaledCameraView(self, scale)
        return self._scaled_view_cache[key]

    def compute_grad_mask(self, config):
        edge_threshold = config["Training"]["edge_threshold"]

        gray_img = self.original_image.mean(dim=0, keepdim=True)
        gray_grad_v, gray_grad_h = image_gradient(gray_img)
        mask_v, mask_h = image_gradient_mask(gray_img)
        gray_grad_v = gray_grad_v * mask_v
        gray_grad_h = gray_grad_h * mask_h
        img_grad_intensity = torch.sqrt(gray_grad_v**2 + gray_grad_h**2)

        if config["Dataset"]["type"] == "replica":
            row, col = 32, 32
            multiplier = edge_threshold
            _, h, w = self.original_image.shape
            for r in range(row):
                for c in range(col):
                    block = img_grad_intensity[
                        :,
                        r * int(h / row) : (r + 1) * int(h / row),
                        c * int(w / col) : (c + 1) * int(w / col),
                    ]
                    th_median = block.median()
                    block[block > (th_median * multiplier)] = 1
                    block[block <= (th_median * multiplier)] = 0
            self.grad_mask = img_grad_intensity
        else:
            median_img_grad_intensity = img_grad_intensity.median()
            self.grad_mask = (
                img_grad_intensity > median_img_grad_intensity * edge_threshold
            )

    def clean(self):
        self.original_image = None
        self.depth = None
        self.grad_mask = None

        self.cam_rot_delta = None
        self.cam_trans_delta = None

        self.exposure_a = None
        self.exposure_b = None
