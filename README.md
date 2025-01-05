## This is the Repository for MA-SAM
MA-SAM: A Multi-atlas Guided SAM Using Pseudo Mask Prompts without Manual Annotation for Spine Image Segmentation
[pdf](https://doi.org/10.1109/TMI.2024.3524570)

## Notice
We only give the framework of the training procedure, please fulfill relevant parts (e.g., the spine_dataset.py, the properties.py, and the relevant codes in train.py) before actually conducting the training process. 

## Step
1. Train the coarse segmentation sub\-network first to obtain coarse segmentation results of input image.
2. Use registration.py to obtain warped label maps of atlas as prompts.
3. Train segmentation sub\-network using input image and warped label maps of atlas.

## BibTex
```
  @article{fan2025ma,
  title={MA-SAM: A Multi-atlas Guided SAM Using Pseudo Mask Prompts without Manual Annotation for Spine Image Segmentation},
  author={Fan, Dingwei and Zhao, Junyong and Li, Chunlin and Wang, Xinlong and Zhang, Ronghan and Zhu, Qi and Wang, Mingliang and Si, Haipeng and Zhang, Daoqiang and Sun, Liang},
  journal={IEEE Transactions on Medical Imaging},
  year={2025},
  publisher={IEEE}
}
```
