# Occlusion-aware-Amodal-Depth-Estimation-for-Enhancing-3D-Reconstruction-from-a-Single-Image
In various fields, such as robotics navigation, autonomous driving, and augmented reality,
the demand for the reconstructing of three-dimensional (3D) scenes from two-dimensional (2D) images
captured by a camera is growing. With advancements in deep learning, monocular depth prediction research
has gained momentum, leading to the exploration of 3D reconstruction from single images. While previous
studies have attempted to restore occluded regions by training deep networks on high-resolution 3D data or
with jointly learned 3D segmentation, achieving perfect restoration of occluded objects remains challenging.
Such mesh generation methods often result in unrealistic interactions with graphic objects, limiting their
applicability. To address this, this paper introduces an amodal depth estimation approach to enhance the
completeness of 3D reconstruction. By utilizing amodal masks that recover occluded regions, the method
predicts the depths of obscured areas. Employing an iterative amodal depth estimation framework allows this
approach to work even with scenes containing deep occlusions. Incorporating a SPADE fusion block within
the amodal depth estimation model effectively combines amodal mask features and image features to improve
the accuracy of depth estimation for occluded regions. The proposed system exhibits superior performance on
occluded region depth estimation tasks compared to conventional depth inpainting networks. Unlike models
that explicitly rely on multiple RGB or depth images to handle instances of occlusion, the proposed model
implicitly extracts amodal depth information from a single image. Consequently, it significantly enhances
the quality of 3D reconstruction even when single images serve as input.
 
![amodal](https://github.com/Seonguke/Occlusion-aware-Amodal-Depth-Estimation-for-Enhancing-3D-Reconstruction-from-a-Single-Image/assets/57488386/850037b1-3543-4a5a-953e-41c418ba3a18)
