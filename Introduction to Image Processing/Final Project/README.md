 # Image Super-Resolution with HAT, Filtering, and Codeformer
Implement existing open-source methods and pre-trained models to improve the resolution of pictures.
## Method
Our project focuses on enhancing image quality using the Hybrid Attention Transformer (HAT) method, integrated with filtering and the Codeformer model for further improvements.

### Super Resolution—HAT
HAT utilizes both channel attention and self-attention mechanisms to improve image reconstruction. It consists of two key features:

### Hybrid Attention Block (HAB): Combines channel attention with window-based multi-head self-attention (WMSA), enhancing global and local feature extraction.
Overlapping Cross-Attention Module (OCAB): Enhances interaction between neighboring window features for better integration of cross-window information.
### HAT-GAN
HAT-GAN is a GAN model using the HAT architecture for real-world image super-resolution. It is trained using large-scale datasets (DIV2k, Flick2K, OST) and a pre-training strategy with MSE-based models.

### Filtering
To improve image quality, we performed filtering to reduce noise. Filters were applied after super-resolution to retain more details while removing unwanted noise. Various filters (mean, max, min, median, and contraharmonic) were tested, with the contraharmonic filter showing the best results for different images.

### Codeformer
Codeformer, a Transformer-based face restoration network, was used to improve image quality after filtering. It addresses challenges in blind face restoration by utilizing a learned discrete codebook prior, enhancing image clarity and detail restoration, particularly for complex images like human faces and animals.

## Results
Best Performance: The combination of HAT + Contraharmonic Filter + Codeformer performed exceptionally well on human faces, dogs, and birds, showing significant improvements over using HAT alone.
Challenges: Some images, such as pandas and fruits, still showed poor restoration with rough edges and unnatural margins.
## References
### 1. HAT: 
Paper: https://arxiv.org/pdf/2205.04437 / https://arxiv.org/pdf/2309.05239 
Github: https://github.com/XPixelGroup/HAT?tab=readme-ov-file 
### 2. Codeformer: 
https://github.com/sczhou/CodeFormer/tree/master
