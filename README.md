# Data-generation
This repository can be used to generate refraction and reflection data for a given reference pattern. This data consits of a depthmap, normal, warp map and a warped image. A refrence pattern should be given as input. The warp map is created for both reflection and refraction, such that both the refracted and reflected warped image can be created. The data can be generated for training, testing and validation. 
Normal, depth/warp maps have a sequence number and frame number.
- Sequece number is the ID for a specific simulation 
- Frame number is the index for a frame within a sequence

The warped images has also an image name that corresponds to the referenece image used.

NB
- Make sure that images in reference_patterns are bigger than 256x256, because the depth maps are created with a size of 256x256 pixel.