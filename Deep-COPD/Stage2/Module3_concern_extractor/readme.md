**Module 3: Concern Extractor**

Module 3 serves as a concern extractor designed to capture hashtags and Latent Dirichlet Allocation (LDA) topics from the tweet data. Here's how it operates:

- **Hashtag Extraction**: Hashtags are identified within the tweet texts through the use of regular expressions. Subsequently, the frequencies of these hashtags are computed.

- **LDA Topic Modeling**: This module utilizes Latent Dirichlet Allocation (LDA) for topic modeling, aiming to identify the primary topics conveyed in the tweets. The original tweets undergo several preprocessing steps, including n-gram processing and document-term matrix construction, resulting in a bag-of-words (BoW) representation. These BoWs are then input into the LDA models.

To determine the optimal number of topics for the LDA model, a grid search is performed within a range of 2 to 20, with a step size of 1, for each year.