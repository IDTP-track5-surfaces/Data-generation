import os
import numpy as np
from utils import deform_image, raytracing_im_generator_ST

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_PATTERN_DIR = os.path.join(ROOT_DIR, "reference_patterns")
TRAINING_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "training_images")
# TEST_IMAGE_DIR = os.path.join(ROOT_DIR, "data_sets", "test_images")
WAVE_SEQUENCE_DIR = os.path.join(ROOT_DIR, "wave_sequences")


def generaet_warp_map(n1 = 1, n2 = 1.33):
    # n1 = 1.0 air
    # n2 = 1.33 water
    
    for waves in os.listdir(WAVE_SEQUENCE_DIR):
        wave_path = os.path.join(WAVE_SEQUENCE_DIR, waves)
        
        for seq in os.listdir(wave_path):
            seq_path = os.path.join(wave_path, seq)
            
            if not os.path.exists(os.path.join(seq_path, "warp")):
                os.makedirs(os.path.join(seq_path, "warp", "reflection"))
                os.makedirs(os.path.join(seq_path, "warp", "refraction"))
                
            for file in os.listdir(os.path.join(seq_path, "depth")):
                depth_map = np.load(os.path.join(seq_path, "depth", file))
                
                for i in ["refraction", "reflection"]:
                    warp_map = raytracing_im_generator_ST(depth_map, n1, n2, i)
                    np.save(os.path.join(seq_path, "warp", i, file), warp_map)


def generate_warped_image(image, name):
    rgb = image / 255.0
        
    for waves in os.listdir(WAVE_SEQUENCE_DIR):
        wave_path = os.path.join(WAVE_SEQUENCE_DIR, waves)
        
        for seq in os.listdir(wave_path):
            seq_path = os.path.join(wave_path, seq)
            
            for file in os.listdir(os.path.join(seq_path, "warp")):
                warp_map = np.load(os.path.join(seq_path, "warp", file))
                
                for i in ["refraction", "reflection"]:
                    rgb_deformation = deform_image(rgb, warp_map)
                    
                    if not os.path.exists(os.path.join(TRAINING_IMAGE_DIR, i, waves, seq, name)):
                        os.makedirs(os.path.join(TRAINING_IMAGE_DIR, i, waves, seq, name))
                        
                    np.save(os.path.join(TRAINING_IMAGE_DIR, i, waves, seq, name, file), rgb_deformation)



# if __name__ == "__main__":
#     __main__()


# if __name__ == "__main__":
#     generate_warped_image()
