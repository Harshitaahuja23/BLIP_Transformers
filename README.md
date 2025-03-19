# BLIP: Bootstrapping Language-Image Pretraining for Unified Vision-Language Understanding and Generation

**Authors: Junnan Li, Dongxu Li, Caiming Xiong, Steven Hoi**

 ## Overview
### Context & Problem Statement

Vision-language pretraining (VLP) has driven significant progress in multimodal learning, enabling models to connect images and text. However, existing pre-trained models face key limitations:
1. **Task Specialization Problem**: Most models excel at either understanding-based tasks (e.g., image-text retrieval) or generation-based tasks (e.g., image captioning), but not both.
2.	**Noisy Data Problem**: Many VLP models rely on web-crawled image-text pairs, which are often inaccurate or misaligned, making learning inefficient.

These challenges limit the effectiveness of vision-language models in real-world applications, where both understanding and generation capabilities are crucial.

### BLIP’s Approach

BLIP (Bootstrapping Language-Image Pretraining) tackles these problems using two key innovations:
- Multimodal Mixture of Encoder-Decoder (MED):
  - A flexible vision-language model that can function as:
    - A unimodal encoder for retrieval tasks.
    - An image-grounded text encoder for multimodal understanding.
    - An image-grounded text decoder for text generation.
  - Jointly pre-trained on three key objectives: 
    - Image-Text Contrastive Loss (ITC): Aligns image & text embeddings.
    - Image-Text Matching Loss (ITM): Distinguishes matched vs. mismatched image-text pairs.
    - Language Modeling Loss (LM): Enables image captioning.
- CapFilt (Captioning + Filtering) for Noisy Data:
  - A data bootstrapping technique that improves dataset quality by: 
    - Generating synthetic captions for web images using a Captioner.
    - Filtering out noisy captions from both web-sourced and synthetic captions using a Filter.
  - This results in a cleaner and more informative training dataset.

### How BLIP Addresses the Problem
- Unifies understanding & generation: The MED architecture allows BLIP to handle both image-text retrieval and text generation effectively.
- Cleans training data: CapFilt removes noise from web datasets, ensuring higher-quality supervision.
- Achieves state-of-the-art (SOTA) performance on multiple tasks: 
  - +2.7% Recall@1 in image-text retrieval (COCO dataset).
  - +2.8% CIDEr in image captioning.
  - +1.6% VQA Score in Visual Question Answering.
  - Zero-shot generalization to video-language tasks without additional training.

## Research Questions

1. Why is it important for a vision-language model to handle both understanding and generation tasks instead of focusing on just one?
2. How does CapFilt improve training data quality, and why is filtering noisy web captions necessary for vision-language models?
























