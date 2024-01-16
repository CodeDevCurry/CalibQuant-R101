import os
import random
import shutil

src_dir = "train"
dst_dir = "sel_1024_1000classes"

selected_images = []
for class_dir in os.listdir(src_dir):
    class_path = os.path.join(src_dir, class_dir)
    if os.path.isdir(class_path):
        image = random.choice(os.listdir(class_path))
        selected_images.append((class_dir, image))

remaining_images = 1024 - 1000
all_images = [(dp.split('/')[-1], f) for dp, dn, filenames in os.walk(src_dir)
              for f in filenames if f not in [img[1] for img in selected_images]]
selected_images.extend(random.sample(all_images, remaining_images))

for class_dir, image_name in selected_images:
    src_image_path = os.path.join(src_dir, class_dir, image_name)
    dst_class_dir = os.path.join(dst_dir, class_dir)
    if not os.path.exists(dst_class_dir):
        os.makedirs(dst_class_dir)
    shutil.copy(src_image_path, dst_class_dir)
