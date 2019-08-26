import Augmentor

p = Augmentor.Pipeline("C:/Users/Henrique/Documents/teia/teia/Binario/data/train")

#p.rotate(probability=0.5, max_left_rotation=12, max_right_rotation=12)

p.flip_left_right(probability=0.3)

p.flip_top_bottom(0.6)

p.random_brightness(0.7, 0.3, 0.9)

p.random_distortion(0.6, 10, 30, 5)

p.skew_corner(0.3, 1)

p.skew_top_bottom(0.4, 1)


#p.crop_by_size(0.6, 1052, 1052, True)

p.sample(280)

p.process()
