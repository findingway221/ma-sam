## This is the Repository for MA-SAM
MA-SAM: A Multi-atlas Guided SAM Using Pseudo Mask Prompts without Manual Annotation for Spine Image Segmentation

## Notice
We only give the framework of training procedure, please fulfill relevant parts (e.g., the spine_dataset.py, the properties.py, and the relevant codes in train.py) before actually conduct training process. 

## Step
1. Train the coarse segmentation sub\-network first to obtain coarse segmentation results of input image.
2. Use registration.py to obtain warped label maps of atlas as prompts.
3. Train segmentation sub\-network using input image and warped label maps of atlas.