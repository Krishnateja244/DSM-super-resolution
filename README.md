# Deep Learning based Super Resolution of Urban Digital Surface Models

This Repo will be soon updated with all the necessary training/ testing instructions 

## Abstract 
  Digital Surface Model (DSM), characterized by their ability to represent both natural and man-made features with precision, plays an indispensable role in diverse fields such as urban planning, environmental monitoring, disaster management, and infrastructure development. However, the intrinsic limitations of traditionally collected DSMs, which capture only a fraction of the Earth's complexity, necessitate the development of super-resolution techniques. This research is dedicated to advancing the field of urban DSM super-resolution through deep learning. The primary objective is the generation of highly accurate high-resolution DSMs, an area of study that remains relatively underexplored in comparison to image super-resolution or DTM super-resolution. The complexity of urban topography, the continuous data, and the presence of high-frequency features in DSMs compared to Digital Terrain Models (DTMs) pose unique challenges. Prior works in super-resolution, typically designed for DTMs, may not fully address the nuances of high-resolution DSM reconstruction.

  To bridge this gap, this research investigated the performance of state-of-the-art Generative Adversarial Network (GAN)-based deep learning algorithms such as D-SRGAN, ESRGAN, Real-ESRGAN, Pix2Pix(U-Net), and EfficientNetv2 for super-resolving DSM. These algorithms serve as the foundation for establishing a baseline model for DSM super-resolution. Comprehensive qualitative and quantitative analyses conducted in this research reveal that D-SRGAN stands out as the promising baseline model by performing better than other deep-learning models and classical bicubic upsampling. However, the model couldn't reconstruct the fine details present in the urban environment. Therefore the research focused on the development of a deep learning model with D-SRGAN as a base to improve the baseline model performance, which includes D-SRGAN with multi-head attention layers, channel attention, co-learning architecture, and Encoder-decoder style D-SRGAN. These models do not yield significant improvements over the baseline. This outcome is attributed to the distinctive attributes of DSM, including the absence of high-frequency features in 4x low-resolution DSM and the presence of complex high-level semantics. Moreover, these models demonstrate limited feasibility in enhancing resolution beyond a 4X scale.
