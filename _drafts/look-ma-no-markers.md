---
layout: post
title: "Notes on 'Look Ma, no markers: holistic performance capture without the hassle'"
date: "2024-02-10"
tags: ["paper", "machine learning", "pose estimation", "graphics", "motion capture"]
---

[Paper Link](https://arxiv.org/pdf/2410.11520)

# Outline
- 'Single shot' performance-capture pipeline for generating motion capture data from video images
- No need for specialized equipment, multiple takes, or manual editing
- Uses a pipeline which combines a DNN for parametric pose estimation, followed by a refinement and optimization process based on a *parametric* human model
- DNN is trained using *synthetic* data from this parametric model

# Synthetic dataset

The DNN is fit on a synthetic dataset generated using a parametric model of the human body and face. This has a number of advantages:
- Direct access to ground-truth 3D point locations of the mesh generated from the model (no need for manual labeling)
- Ability to easily generate a large dataset

The full synthetic data pipeline proceeds in 3 stages:

Parametric Human Model --> Texture/Hair/Clothing addition --> Realistic 3D render with different lighting + camera conditions

3 separate training datasets of images are created in this manner: face, hand, body (~100,000 images each).

# Performance Capture

Goal is to capture body motion data from video data. This is done in 2 stages:
1. DNN is trained on the still 2D images to generate 2D landmark data, and estimates of the pose parameters
2. The parametric model is fit to the 2D landmark data by minimizing the "projection error" of the 3D mesh onto the predicted 2D landmark data which also incorporates certain priors. It is also initialized using the predicted pose parameters.

## DNN details

Two options - directly learn the model parameters, or use DNN to predict 2D landmark data (or other such latent). Directly learning parameters leads to strong inductive bias and is robust, but lacks precision. Alternative of generating landmark estimate to fit the parametric model works better.

Single DNN also struggles to fit detailed areas like hands and face, so train 3 separate models for hand, face, body. This requires detecting region of interest (ROI) for each, which is done using ground truth during training, and then using the estimated landmark locations during inference, while the whole-body ROI uses standard detector model (i.e. whole body prediction used to generate hand ROI, then hand DNN is used to refine).

### Targets/Losses

For landmark data, predict a gaussian $(\mu_i, \sigma_i)$ of 2D location of point, and log likelihood loss with per-landmark weights (how did they determine per-landmark weights?).

For pose estimation, there is some kind of pose reconstruction loss where certain rotations are predicted. Note that there is a pose loss, joint loss, and shape loss (all generated from human model parameters).

These losses are combined using weights and used for each DNN (face, hand, body). Face does not use pose or shape heads since variation is small and can be fully reconstructed from the final model optimization step.

## Model fitting

Model is initialized using predicted shapes and poses, and we try to find optimal parameters to describe the given landmark data (using LBFGS).

Solving has 3 steps:
1. Generate global translation + rotation by solving Perspective-n-Point problem (PnP) (Fischer & Bolles 1981) for camera perspective guessing.
2. Fit the parametric human model by optimizing the parameters.
3. Optionally refining camera details using calibration data or focal length if available.

### Details

This is a highly *underdetermined* problem (single view from 3D) so use a lot of regularization and priors to ensure a good solution (and initializing with the DNN prediction helps too).

- The uncertainty of the landmark data can be used to weight the loss contribution of each point
- Use a number of additional regularizers
    - GMM prior on body/face shapes generated from dataset
    - L1 loss for sparse model
    - Frame-to-frame 3D movement loss
    - Non-physical loss e.g. avoid eyes or tongue intersecting face
    - More can be added if we have additional information...
- Neural pose prior to ensure reasonable poses
    - Uses a normalizing flow trained on the pose dataset (RealNVP)

# Conclusions

Robust, generalizable performance due to combination of DNN + strong inductive bias in parametrized model. Number of limitations:

- Synthetic dataset coverage lacking for unusual poses, challenging hairstyles or loose clothing
- Constrained by texture/asset library used for generation
- Might struggle with challenging camera effects?
- Relies on good landmark prediction, struggles with heavily occluded areas such as interlaced fingers.
- Requires heavy use of different priors which must be tuned - perhaps in the future could be more end-to-end learning
- Static camera assumption - in theory could jointly optimize camera location and body location!