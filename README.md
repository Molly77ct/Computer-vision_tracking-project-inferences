# Final Results and Requirements

Final Results

After multiple months of experimentation, debugging, environment migration, dependency fixing, and framework adaptation, successful inference pipelines were achieved for:

- DeepSORT
- Deep OC-SORT

The final implementation was capable of running multi-object tracking inference successfully on modern GPU hardware under Ubuntu 24.04.
, using yolov8 model. Weight can be find here:
https://drive.google.com/file/d/1hnmL09jbtm48KJHf3GMLFUh1jiTVfbsD/view?usp=sharing

In addition to the successful inference setup, the project also achieved:

- generation and evaluation of tracking metrics using Deep OC-SORT,
- development and adaptation of inference scripts for Deep OC-SORT,
- development and adaptation of inference scripts for DeepSORT,
- successful execution of tracking pipelines on video sequences,
- integration and testing of detector-tracker workflows,
- and validation of the tracking frameworks under modern Python and CUDA environments.

The implemented scripts allow running inference experiments, processing tracking outputs, and evaluating tracking performance across multiple datasets and video sources.

Additionally, the project provided valuable insights into:

- modern MOT architectures,
- motion-based vs appearance-based tracking,
- moving-camera tracking challenges,
- drone-based object tracking,
- adaptive re-identification methods,
- and compatibility adaptation of legacy research code for modern AI environments.

---

# 9. Requirements and Environment Setup

## Deep OC-SORT Requirements

The final working implementation of Deep OC-SORT was executed under the following environment:

### Operating System
- Ubuntu 24.04 LTS

### Python Environment
- Python 3.10.20
- Conda environment recommended

### Hardware
- NVIDIA RTX 5080 GPUs
- CUDA-compatible NVIDIA drivers

### Main Dependencies
- PyTorch
- TorchVision
- OpenCV
- NumPy
- SciPy
- FilterPy
- YOLOX
- FastReID
- lap
- cython_bbox
- motmetrics
- scikit-learn
- pandas
- matplotlib

### Additional Notes
Deep OC-SORT required several compatibility adjustments due to:
- newer GPU architectures,
- updated CUDA versions,
- newer Python releases,
- deprecated dependencies,
- and incompatibilities between older MOT libraries and recent PyTorch versions.

The framework is modular and detector-agnostic, allowing integration with:
- YOLOv5,
- YOLOv8,
- YOLOX,
- and custom object detectors.

### Required Resources
The following datasets and pretrained weights were required:
- MOT17
- MOT20
- DanceTrack
- pretrained `.pth.tar` ReID weights
- detector checkpoints

### Reference Repository
https://github.com/gerardmaggiolino/deep-oc-sort

---

## DeepSORT Requirements

The DeepSORT inference pipeline was also successfully implemented and tested.

### Operating System
- Ubuntu 24.04 LTS
- Windows 11 (partial development and testing)

### Python Environment
- Python 3.10
- Conda environment recommended

### Main Dependencies
- PyTorch
- OpenCV
- NumPy
- SciPy
- filterpy
- ultralytics
- deep-sort-realtime
- torchvision

### Detector Support
DeepSORT was tested with:
- YOLOv8
- Ultralytics framework

### Additional Notes
DeepSORT is easier to deploy compared to Deep OC-SORT because:
- it has fewer dependency conflicts,
- it is actively maintained in newer libraries,
- and it integrates directly with Ultralytics.

However, DeepSORT is more dependent on appearance embeddings and may be less robust in:
- heavy occlusion scenarios,
- moving-camera environments,
- and drone-based tracking applications.

### Final Outcome
Inference scripts for both DeepSORT and Deep OC-SORT were successfully developed and executed, including:
- video inference,
- trajectory visualization,
- tracking metric generation,
- and detector-tracker integration workflows.
