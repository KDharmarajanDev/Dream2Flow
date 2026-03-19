# Data

This directory stores downloaded scene folders.

Download the scene archive from the Hugging Face link listed in the main [README](/scr/karthikd/Documents/video_particle/Dream2Flow/README.md), then extract or move the downloaded scene folder directly into this `data/` directory.

Expected layout:

```text
data/
  put_bread/
    camera_rgb.png
    scene_data.yaml
    object_flow_result.pt
    ...
```

For example, after downloading the `put_bread` scene, the final path should be [`data/put_bread/`](/scr/karthikd/Documents/video_particle/Dream2Flow/data/put_bread).
