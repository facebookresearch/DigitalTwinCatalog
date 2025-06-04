# Dataset pre-processing

## Download
- Link to our real-world dataset TBA
- [MII dataset](https://github.com/zju3dv/InvRender)
- [DTU](https://github.com/Totoro97/NeuS)

## Our data format

    {scene_name}
    ├── cameras.json
    └── images/*png

The `cameras.json` detail the camera parameters and file path of each frame and also some additional scene information. See below for an example:
```json
{
    "fx": 723.3623,
    "fy": 723.3623,
    "cx": 450.0,
    "cy": 337.5,
    "aabb": [[0.0000,  0.0000, -0.3655], [0.2580, 0.2000, 0.0000]],
    "split": {
        "train": [1, 2, 3, 5, 6, 7, 9],
        "test": [0, 4, 8],
    },
    "frames": [
        {
            "path": "images/im_0001.png",
            "mask": "optional path to mask.png",
            "mask_from_alpha": "optional boolean to indicate image alpha is mask",
            "to_world": [
                [0.0646, 0.3667, 0.9280, -0.2012],
                [-0.0425, 0.9301, -0.3645, 0.2044],
                [-0.9970, -0.0159, 0.0757, -0.2122],
                [0.0, 0.0, 0.0, 1.0]
            ]
        },
        ...
    ]
}
```
- `fx, fy, cx, cy`: Intrinsic parameters.
- `aabb`: An axis-aligned bonding box that enclose the foreground object of interest. The scene outside aabb will be modeled by the background model.
- `frames`: A list of frames meta data each of which include the relative file path and the camera-to-world matrix of the frame.


## Pre-process MII dataset
Our raw data folder structure is:

    data
    └── Synthetic4Relight
        └── [air_baloons|chair|hotdog|jugs]
            ├── transforms_train.json
            ├── transforms_test.json
            ├── test_rli/
            ├── train/
            └── test/

To generate the meta json file, run:
```
python scripts/preprocess/mii.py data/Synthetic4Relight/air_baloons/
```
It will produce a `data/Synthetic4Relight/air_baloons/cameras.json`. After processing all the 4 scenes, you can now run `./scripts/mii/mii.sh`.


## Pre-process DTU dataset
You will need [DTUeval-python](https://github.com/jzhangbs/DTUeval-python) and the [DTU official ground-truth](https://roboimagedata.compute.dtu.dk/?page_id=36) to run evaluation. Our raw data folder structure is:

    data
    ├── DTU
    │   ├── dtu_scan24
    │   │   ├── cameras_large.npz
    │   │   ├── cameras_sphere.npz
    │   │   ├── image/*png
    │   │   └── mask/*png
    │   └── ...
    └── DTU_official
        ├── DTUeval-python/
        ├── ObsMask/
        ├── Points/
        └── SampleSet/

To generate the meta json file, run:
```
python scripts/preprocess/dtu.py data/DTU/dtu_scan24/
```
It will produce a `data/DTU/dtu_scan24/cameras.json`. After processing all the 15 scenes, you can now run `./scripts/dtu/run_all.sh`.
