import torch

# %matplotlib
import matplotlib.pyplot as plt


def compute_mode(block):
    """
    Computes the mode of a 2D tensor block.

    Args:
        block (torch.Tensor): 2D tensor of shape (H_block, W_block) containing integer labels.

    Returns:
        torch.Tensor: A scalar tensor containing the mode label.
    """
    # Flatten the block to 1D
    block_flat = block.flatten()
    # Compute mode
    mode_label, _ = torch.mode(block_flat)
    return mode_label


def get_block_sizes(total_size, num_blocks):
    """
    Computes the sizes of each block along one dimension, distributing extra pixels to initial blocks.

    Args:
        total_size (int): Total number of pixels along one dimension (e.g., 512).
        num_blocks (int): Number of blocks along that dimension (e.g., 10).

    Returns:
        list: A list containing the size of each block.
    """
    base_size = total_size // num_blocks  # Floor division
    remainder = total_size % num_blocks  # Extra pixels to distribute

    # Distribute the extra pixels to the first 'remainder' blocks
    block_sizes = [
        base_size + 1 if i < remainder else base_size for i in range(num_blocks)
    ]
    return block_sizes


block_sizes_h = get_block_sizes(512, 10)  # For height
block_sizes_w = get_block_sizes(512, 10)  # For width

print(block_sizes_h)
# Output: [52, 52, 51, 51, 51, 51, 51, 51, 51, 51]

print(block_sizes_w)
# Output: [52, 52, 51, 51, 51, 51, 51, 51, 51, 51]


def get_cumulative_indices(block_sizes):
    """
    Computes the cumulative indices for splitting based on block sizes.

    Args:
        block_sizes (list): List of block sizes along one dimension.

    Returns:
        list: Cumulative start indices.
    """
    cumulative = [0]
    for size in block_sizes:
        cumulative.append(cumulative[-1] + size)
    return cumulative


cumulative_h = get_cumulative_indices(block_sizes_h)
cumulative_w = get_cumulative_indices(block_sizes_w)

print(cumulative_h)
# Output: [0, 52, 104, 155, 206, 257, 308, 359, 410, 461, 512]

print(cumulative_w)
# Output: [0, 52, 104, 155, 206, 257, 308, 359, 410, 461, 512]


def downsample_label_map_majority_vote(label_map, output_size=(10, 10)):
    """
    Downsamples a multi-class label map to a specified output size using majority voting.

    Args:
        label_map (torch.Tensor): 2D tensor of shape (H, W) containing integer labels.
        output_size (tuple): Desired output size as (H_out, W_out).

    Returns:
        torch.Tensor: Downsampled label map of shape (H_out, W_out).
    """
    H, W = label_map.shape
    H_out, W_out = output_size

    # Compute block sizes
    block_sizes_h = get_block_sizes(H, H_out)
    block_sizes_w = get_block_sizes(W, W_out)

    # Compute cumulative indices
    cumulative_h = get_cumulative_indices(block_sizes_h)
    cumulative_w = get_cumulative_indices(block_sizes_w)

    # Initialize the output tensor
    downsampled = torch.zeros((H_out, W_out), dtype=label_map.dtype)

    # Iterate over each block and compute the mode
    for i in range(H_out):
        for j in range(W_out):
            start_h = cumulative_h[i]
            end_h = cumulative_h[i + 1]
            start_w = cumulative_w[j]
            end_w = cumulative_w[j + 1]

            block = label_map[start_h:end_h, start_w:end_w]
            mode_label = compute_mode(block)
            downsampled[i, j] = mode_label

    return downsampled


# Create a synthetic 512x512 label map with 3 classes: 0, 1, 2
# For demonstration, create distinct regions for each class
label_map = torch.zeros((512, 512), dtype=torch.int64)

# Assign class 0 to top-left quadrant
label_map[:256, :256] = 0

# Assign class 1 to top-right quadrant
label_map[:256, 256:] = 1

# Assign class 2 to bottom half
label_map[256:, :] = 2

# Introduce some noise
noise = torch.randint(0, 3, (512, 512))
label_map = torch.where(noise > 1, noise, label_map)

# Downsample to 10x10 using majority vote
downsampled_map = downsample_label_map_majority_vote(label_map, output_size=(10, 10))

print("Downsampled Label Map (10x10):")
print(downsampled_map)

# Visualize the original and downsampled label maps
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title("Original Label Map (512x512)")
plt.imshow(label_map, cmap="jet")
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Downsampled Label Map (10x10)")
plt.imshow(downsampled_map, cmap="jet")
plt.colorbar()

plt.show()
