# Data-generation
This repository can be used to generate refraction and re   flection data for a reference pattern, given a depth map of the fluid. 

- Make sure that images in reference_patterns are 128x128, because simulation generates a depthmap of 128x128 pixel.

The depth maps of each frame should be order in such fashion: 
```
├── wave_sequences
│   ├── wave_x
│   │   ├── Seq_0
|   │   │   ├── frame_0.npy
|   │   │   ├── frame_1.npy
|   │   │   ├── ...
|   │   │   ├── frame_N.npy
│   │   ├── Seq_1
│   │   ├── ...
│   │   ├── Seq_N
│   ├── wave_y
│   ├── ...
│   ├── wave_N
```

and they are currently generated in matlab