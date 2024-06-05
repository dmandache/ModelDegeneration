import random
import torch
import glob
import numpy as np
from PIL import Image, ImageDraw


def sample_indices(vector, k, max_classes=None, seed=None):
    # # Example usage:
    # vector = torch.tensor([0, 0, 1, 1, 2, 2, 2, 3, 3, 3, 3])  # Example tensor with 4 labels
    # k = 3  # Number of points to sample from each label
    # seed = 42  # Seed for reproducibility
    # sampled_indices = sample_indices(vector, k, seed)
    # print("Sampled indices:", sampled_indices)

    if seed is not None:
        random.seed(seed)

    indices = []
    label_dict = {}

    # Group indices by label
    for i, label in enumerate(vector):
        label = label.item() if torch.is_tensor(label) else label
        if label not in label_dict:
            label_dict[label] = [i]
        else:
            label_dict[label].append(i)

    # Subsample number of classes
    if max_classes is not None and max_classes < len(label_dict.keys()):
        classes = random.sample(label_dict.keys(), max_classes)
        label_dict = {c: label_dict[c] for c in classes}
        
    # Sample K points from each label
    for label, label_indices in label_dict.items():
        sampled_indices = random.sample(label_indices, min(k, len(label_indices)))
        indices.extend(sampled_indices)

    random.shuffle(indices)
    return indices


# Shapes Dataset containing Cricles, Triangles and Squares
def load_shapes_dataset(dirpath='data/shapes'):
    x = []
    y = []

    label_dict = {
        'circles'   : 0,
        'squares'   : 1,
        'triangles' : 2
    }

    files = glob.glob(dirpath + '/**/*.png', recursive=True)

    for f in files:
        img=Image.open(f)
        img=img.resize(size=(28,28))
        img=img.convert('L')
        x.append(np.array(img))
        label = f.split('/')[-2]
        y.append(label_dict[label]) #y.append(label)
        del img

    return np.array(x), np.array(y)


# Disks and Circles Synthethic Dataset
# centered, black and white, of different diameter and thickness
def gen_circles_dataset(num_images=100, image_size=28, seed=None):
    
    if seed is not None:
        random.seed(seed)

    x = []
    y = []

    label_dict = {
        'circle' : 0,
        'disk'   : 1,
    }

    def create_shape_image(image_size, shape, diameter, thickness):
        image = Image.new('L', (image_size, image_size), 0)  # Create a black image
        draw = ImageDraw.Draw(image)
        center = image_size // 2
        radius = diameter // 2
        
        if shape == 'disk':
            # Draw a filled circle (disk)
            draw.ellipse((center - radius, center - radius, center + radius, center + radius), fill='white')
        elif shape == 'circle':
            # Draw a circle with given thickness
            for t, fill in ( 0, 'white' ), ( thickness, 'black' ):
                draw.ellipse((center - radius + t, center - radius + t, center + radius - t, center + radius - t), fill=fill)
        
        return np.array(image)
        
    for _ in range(num_images):
        shape = random.choice(['disk', 'circle'])
        diameter = random.randint(image_size // 5, image_size // 1.1)  # Diameter between 5 and 25
        thickness = random.randint(1, diameter // 2.5) if shape == 'circle' else None  # Thickness for circle, between 1 and 10
        
        image = create_shape_image(image_size, shape, diameter, thickness)
        
        x.append(image)
        y.append(label_dict[shape])

    return np.array(x), np.array(y)
    