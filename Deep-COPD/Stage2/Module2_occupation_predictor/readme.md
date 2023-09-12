**Module 2: Occupation Predictor**

Module 2 is a custom deep learning model designed to determine the occupation categories of Twitter users based on their profile information and tweet texts. This module employs a word-based convolutional neural network architecture, comprising several key components:

- **Encoding Layer**: This layer processes the input data.
- **Embedding Layer**: It converts words into numerical vectors.
- **Parallel Convolution Blocks**: These blocks feature various kernel sizes to capture different text patterns.
- **Dense Layer**: Responsible for making predictions.

In our research, we simplified the nine occupation categories (OCs) from the Standard Occupation Classification (SOC) into two categories, denoted as OC1 and OC2 (refer to Appendix p 2). OC1 represents individuals with a relatively higher socioeconomic status, such as managers, directors, and senior officials, while OC2 represents those with a relatively lower socioeconomic status, including process, plant, and machine operatives.
