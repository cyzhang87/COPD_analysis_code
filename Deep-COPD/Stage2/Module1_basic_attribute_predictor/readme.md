**Module 1: Basic Attribute Predictor**

Module 1 serves as an attribute predictor, estimating the probabilities of three demographic attributes: user type, gender, and age. This prediction is based on the analysis of various user profile elements, including profile images, screen names, names, and biographies. The module is implemented using an open-source package of the M3 (Multimodal, Multilingual, and Multi-Attribute) model, which is a deep neural system trained on a vast dataset.

The M3 model primarily consists of the following components:
- **DenseNet Layer**: image processing.
- **Three Character-based Neural Networks**: text processing.
- **Two Fully Connected Dense Layers**: result classification.

The original code is at: https://github.com/euagendas/m3inference.
We further improved a part of the code to update the profile information, in update_profile_info.py.