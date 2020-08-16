# TFBS model
The source code of TFBS model
TFBS model contains four module including input module, temporal feature extraction module, segmentation module and output module. It employs an LSTM model matrix to learn masses of temporal features from raw time-series satellite image and produces an image consists of these temporal features. The temporal feature image is then input to an UNET module to extract spatial context information form temporal features and produce a segmentation image
