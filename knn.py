import os
import math


def load_pgm_bin(filename):
    """
    Reads a binary PGM (P5) file and returns a 2D list of pixel values.
    Compatible with 8-bit grayscale images.
    """
    with open(filename, 'rb') as f:
        # Read magic number (P5)
        magic_number = f.readline().strip()
        if magic_number != b'P5':
            raise ValueError("Unsupported format — not a binary PGM (P5).")

        # Skip comments
        line = f.readline()
        while line.startswith(b'#'):
            line = f.readline()

        # Read image size
        width, height = [int(i) for i in line.split()]

        # Read maximum gray value
        maxval = int(f.readline().strip())
        if maxval > 255:
            raise ValueError("Only 8-bit PGM files are supported.")

        # Read pixel data
        pixel_data = f.read(width * height)
        if len(pixel_data) != width * height:
            raise ValueError("File ended unexpectedly while reading pixels.")

        # Convert to 2D list
        image = [list(pixel_data[row * width:(row + 1) * width]) for row in range(height)]
        return image

# --- Function to load a PGM (Portable Graymap) image ---
def load_pgm(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    
    # Ignore comments 
    lines = [l.strip() for l in lines if not l.startswith('#') and l.strip() != '']

    assert lines[0] == 'P2', f"Invalid PGM format: {path}"

    # Read dimensions (width, height)
    width, height = map(int, lines[1].split())
    maxval = int(lines[2])

    # Flatten pixel values
    pixels = list(map(int, " ".join(lines[3:]).split()))

    # Convert to 2D matrix
    matrix = [pixels[i * width:(i + 1) * width] for i in range(height)]
    return matrix

# --- Euclidean distance between two images ---
def euclidean_distance(img1, img2):
    dist = 0.0
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            diff = img1[i][j] - img2[i][j]
            dist += diff * diff
    return math.sqrt(dist)

# --- kNN classifier ---
def knn_classify(train_data, test_image, k=3):
    distances = []
    for (img, label) in train_data:
        d = euclidean_distance(test_image, img)
        distances.append((d, label))
    distances.sort(key=lambda x: x[0])
    neighbors = distances[:k]

    # Majority vote
    votes = {}
    for _, label in neighbors:
        votes[label] = votes.get(label, 0) + 1
    return max(votes, key=votes.get)

# --- Load training images ---
def load_dataset(dataset_dir):
    data = []
    label_path = dataset_dir
    for file in os.listdir(label_path):
        label = file[0]
        if file.endswith(".pgm"):
            path = os.path.join(label_path, file)
            img = load_pgm_bin(path)
            data.append((img, int(label)))
    return data

# --- Main test ---
train_dir = "./yymnist-master/minst_s/train/"
test_image_path = "yymnist-master/mnist/test/4_00006.pgm"

print("Loading training dataset...")
train_data = load_dataset(train_dir)

print(f"Loaded {len(train_data)} training samples.")

print("Loading test image...")
test_img = load_pgm_bin(test_image_path)

predicted_label = knn_classify(train_data, test_img, k=3)
print(f"Predicted class: {predicted_label}")
