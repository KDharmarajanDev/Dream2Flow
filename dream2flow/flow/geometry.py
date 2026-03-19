import torch


def depth_to_world_points(
    depth: torch.Tensor,
    camera_intrinsics: torch.Tensor,
    camera_extrinsics: torch.Tensor,
    pixel_coords: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convert depth values to 3D world coordinates.

    This mirrors the helper used by the reference video-particles pipeline.
    """
    if camera_intrinsics.dim() == 2:
        camera_intrinsics = camera_intrinsics.unsqueeze(0)
    if camera_extrinsics.dim() == 2:
        camera_extrinsics = camera_extrinsics.unsqueeze(0)

    batch_size = depth.shape[0]
    if len(depth.shape) == 3:
        height, width = depth.shape[1:]
        depth = depth.reshape(batch_size, height * width)
    else:
        width = None
        height = None

    num_points = depth.shape[1]

    if pixel_coords is None:
        if height is None or width is None:
            raise ValueError("pixel_coords must be provided when depth has shape (B, N)")
        ys, xs = torch.meshgrid(
            torch.arange(height, device=depth.device, dtype=torch.float32),
            torch.arange(width, device=depth.device, dtype=torch.float32),
            indexing="ij",
        )
        pixel_coords = torch.stack((xs, ys), dim=-1).reshape(1, height * width, 2).expand(batch_size, -1, -1)

    pixel_coords_homogeneous = torch.cat(
        (
            pixel_coords,
            torch.ones((batch_size, num_points, 1), device=depth.device, dtype=depth.dtype),
        ),
        dim=-1,
    )

    camera_coords = torch.linalg.inv(camera_intrinsics) @ pixel_coords_homogeneous.transpose(1, 2)
    camera_coords = camera_coords * depth.unsqueeze(1)
    camera_coords_homogeneous = torch.cat(
        (
            camera_coords,
            torch.ones((batch_size, 1, num_points), device=depth.device, dtype=depth.dtype),
        ),
        dim=1,
    )
    world_coords = camera_extrinsics @ camera_coords_homogeneous
    return world_coords.transpose(1, 2)[..., :3]
