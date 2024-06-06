# Data-generation
This repository can be used to generate refraction and/or reflection data for a given reference pattern. This data consits of a depth map, normal map, warp map and a warped image. The depthmap is an array of the fluid height. The fluid height is created with the application of the air puff rheometer in mind. 

A refrence pattern should be given as input. The warp map is created for both reflection and refraction, such that both the refracted and reflected warped image can be created. 
Normal, depth and warp maps have a sequence number and frame number.
- Sequece number is the ID for a specific simulation 
- Frame number is the index for a frame within a sequence

The depth, normal and warps maps can be created by running

    python main.py --create_maps

and reference images can be created by running

    python main.py --create_images --ref_image "<reference image>"

Both commands can be extended with ```--reflection``` flag, such that the whole process is also repeated for reflection.

The script ```create_checkered_patterns.py``` can be used to create a black and white checkered pattern to be used as reference pattern. The script ```preprocess_experimental_data.py``` can be used to preprocess experimentally obtained data. 

NB
- Make sure that images in reference_patterns are larger or equal size of 128x128 pixels, because the depth maps are created with a size of 128x128 pixel.