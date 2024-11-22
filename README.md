# View Consistent Purification for Accurate Cross-View Localization

View Consistent Purification for Accurate Cross-View Localization, Shan Wang, Yanhao Zhang, Akhil Perincherry, Ankit Vora and Hongdong Li, ICCV 2023 [Paper](https://arxiv.org/abs/2308.08110)

## Abstract
This paper proposes a fine-grained self-localization method for outdoor robotics that utilizes a flexible number of onboard cameras and readily accessible satellite images. The proposed method addresses limitations in existing cross-view localization methods that struggle to handle noise sources such as moving objects and seasonal variations. It is the first sparse visual-only method that enhances perception in dynamic environments by detecting view-consistent key points and their corresponding deep features from ground and satellite views, while removing off-the-ground objects and establishing homography transformation between the two views. Moreover, the proposed method incorporates a spatial embedding approach that leverages camera intrinsic and extrinsic information to reduce the ambiguity of purely visual matching, leading to improved feature matching and overall pose estimation accuracy. The method exhibits strong generalization and is robust to environmental changes, requiring only geo-poses as ground truth. Extensive experiments on the KITTI and Ford Multi-AV Seasonal datasets demonstrate that our proposed method outperforms existing state-of-the-art methods, achieving median spatial accuracy errors below $0.5$ meters along the lateral and longitudinal directions, and a median orientation accuracy error below $2^\circ$.

<p align="center">
  <a href="https://github.com/ShanWang-Shan/PureACL.git"><img src="architecture.jpg" width="100%"/></a>
</p>

## Installation

PureACL is built with Python >=3.6 and PyTorch. The package includes code for both training and evaluation. Installing the package locally also installs the minimal dependencies listed in `requirements.txt`:

``` bash
git clone https://github.com/ShanWang-Shan/PureACL.git
cd PureACL/
pip install -e .
```

## Datasets

We construct our Ford-CVL datasets by correcting the spatial-consistent satellite counterparts from Google Map \cite{google} according to these GPS tags. More specifically, we find the large region covering the vehicle trajectory and uniformly partition the region into overlapping satellite image patches. Each satellite image patch has a resolution of $1280\times 1280$ pixels. A script to download the latest satellite images is provided in (ford_data_process/downloading_satellite_iamges.py). For access to our collected satellite images, please send an inquiry email to shan.wang@anu.edu.au from an academic institution email address. The data is strictly for academic purposes. we will provide you with the download link.


Ford-CVL: Please first download the raw data (ground images) from [https://avdata.ford.com/](https://avdata.ford.com/). We provide the script(ford_data_process/raw_data_downloader.sh) for raw data download and the script(ford_data_process/other_data_downloader.sh) for processed data download. Your dataset folder structure should be like this. If the link in the script file has expired or lacks the necessary permissions, please contact us.
```
FordAV/
├─ 2017-08-04-V2-Log*/
│  ├─ 2017-08-04-V2-Log*-FL/
│  │  └─ *******.png
│  ├─ 2017-08-04-V2-Log*-RR/
│  ├─ 2017-08-04-V2-Log*-SL/
│  ├─ 2017-08-04-V2-Log*-SR/
│  ├─ info_files/
│  │  ├─ gps.csv
│  │  ├─ gps_time.csv
│  │  ├─ imu.csv
│  │  ├─ pose_ground_truth.csv
│  │  ├─ pose_localized.csv
│  │  ├─ pose_raw.csv
│  │  ├─ pose_tf.csv
│  │  ├─ velocity_raw.csv
│  │  ├─ groundview_gps.npy
│  │  ├─ groundview_NED_pose_gt.npy
│  │  ├─ groundview_pitchs_pose_gt.npy
│  │  ├─ groundview_yaws_pose_gt.npy
│  │  ├─ groundview_satellite_pair.npy
│  │  ├─ satellite_gps_center.npy
│  │  ├─ 2017-08-04-V2-Log*-FL-names.txt
│  │  ├─ 2017-08-04-V2-Log*-RR-names.txt
│  │  ├─ 2017-08-04-V2-Log*-SL-names.txt
│  │  └─2017-08-04-V2-Log*-SR-names.txt
│  ├─ Satellit_Image_18
│  │  └─ satellite_*_lat_*_long_*_zoom_18_size_640x640_scale_2.png 
├─ 2017-10-26-V2-Log*/
└─ V2/
```
To update your dataset path, you can modify the "default_conf.dataset_dir" in the following files:  "PureACL/pixlib/dataset/ford.py" or in your training/evaluation script. Additionally, if you wish to change the trajectory for the Ford-CVL dataset, you can adjust the "log_id_train/val/test" in the "PureACL/pixlib/dataset/ford.py" file.


## Models
Weights of the model trained on *Ford-CVL*, hosted [here](https://drive.google.com/drive/folders/1X8pPmBYfLSYwiklQM_f67rInXAGVkTSQ?usp=sharing).


## Evaluation

To perform the PureACL, simply launch the corresponding run script:

```
python -m PureACL.evaluation
```

## Training

To train the PureACL, simply launch the corresponding run script:

```
python -m PureACL.pixlib.train
```

## BibTex Citation

Please consider citing our work if you use any of the ideas presented in the paper or code from this repo:

```
@misc{wang2023view,
      title={View Consistent Purification for Accurate Cross-View Localization}, 
      author={Shan Wang and Yanhao Zhang and Akhil Perincherry and Ankit Vora and Hongdong Li},
      year={2023},
      eprint={2308.08110},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Thanks to the work of [Paul-Edouard Sarlin](psarlin.com/) et al., the code of this repository borrows heavily from their [psarlin.com/pixloc](https://psarlin.com/pixloc), and we follow the same pipeline to verify the effectiveness of our solution.
