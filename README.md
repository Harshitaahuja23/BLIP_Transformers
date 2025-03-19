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

## Architecture Overview & Methodology

### BLIP Model Architecture

BLIP is built on a Multimodal Mixture of Encoder-Decoder (MED), a flexible vision-language model that supports three different modes:
- Unimodal Encoder (ITC Loss - Contrastive Learning):
  - Separately encodes images and text.
  - Image encoder is a Vision Transformer (ViT) that converts images into patch-based embeddings.
  - Text encoder is BERT-based, using a [CLS] token to summarize text.
- Image-Grounded Text Encoder (ITM Loss - Matching Learning):
  - Introduces cross-attention (CA) layers between self-attention layers and feed-forward networks.
  - Uses an [Encode] token to capture fine-grained image-text alignment.
  - Helps improve retrieval tasks by distinguishing matched vs. unmatched pairs.
- Image-Grounded Text Decoder (LM Loss - Language Modeling):
  - Replaces bi-directional self-attention with causal self-attention.
  - Uses [Decode] token to generate descriptive captions for images.
  - Enables better generalization to captioning and text generation tasks.

**Key Advantage**: Unlike previous models, BLIP’s MED architecture can switch between all three modes, making it highly flexible for both understanding and generation tasks.

<img width="478" alt="image" src="https://github.com/user-attachments/assets/3f50f488-107e-4089-b952-db178591ac84" />
<img width="960" alt="Screenshot 2025-03-18 at 11 52 47 PM" src="https://github.com/user-attachments/assets/f6623e2e-585b-4c89-9290-f6a0369a1ec2" />

### Pre-training Objectives
BLIP is optimized using three key training objectives:
- Image-Text Contrastive Loss (ITC) → Aligns image and text representations to distinguish positive vs. negative pairs.
- Image-Text Matching Loss (ITM) → Fine-grained classification of matched vs. unmatched image-text pairs.
- Language Modeling Loss (LM) → Trains the decoder to generate captions from images.

Each image-text pair only requires one forward pass through the image encoder but three passes through the text encoder, ensuring efficient computation.

### CapFilt: Improving Data Quality
**Problem**: Web-crawled datasets contain noisy captions that are often misaligned with image content, making training inefficient.
**Solution**: CapFilt (Captioning + Filtering)
- **Captioner (Text Decoder - LM Fine-tuned)**: Generates synthetic captions for web images.
- **Filter (Text Encoder - ITC & ITM Fine-tuned)**: Removes noisy captions that don’t align with images.
- **Final dataset**: Combination of filtered synthetic captions + human-annotated captions, leading to higher-quality training data.

**Impact**: Cleaner training data leads to better model generalization and performance across multiple vision-language tasks.






