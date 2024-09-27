# Dataset Description

## Overview
This dataset is part of the Cyber2A project, derived from a study led by Yili Yang, as described in the citation paper below. It is intended for demonstration or teaching purposes and contains data related to retrogressive thaw slumps (RTS) mapping. 

## Contents
The dataset includes the following folders and files:
```
- rts/
  - images/  # Contains the RTS images.
  - masks/   # Contains the RTS masks.
  - data_split.json  # A dictionary with "train" and "valtest" keys, listing the corresponding image file names.
  - coco_rts_train.json  # Contains the training set in COCO format, with instance-based annotations.
  - coco_rts_valtest.json  # Contains the validation and test sets in COCO format, with instance-based annotations.
```

## Contact
For any questions or further information, please contact Chia-Yu Hsu at chsu53@asu.edu.

## Citation
If you use this dataset in the workshop, please cite the following paper:

```
@article{yang2023mapping,
  title={Mapping retrogressive thaw slumps using deep neural networks},
  author={Yang, Yili and Rogers, Brendan M and Fiske, Greg and Watts, Jennifer and Potter, Stefano and Windholz, Tiffany and Mullen, Andrew and Nitze, Ingmar and Natali, Susan M},
  journal={Remote Sensing of Environment},
  volume={288},
  pages={113495},
  year={2023},
  publisher={Elsevier}
}
```