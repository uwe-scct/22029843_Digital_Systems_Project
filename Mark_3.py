import numpy as np
import matplotlib.pyplot as plt

# Room setup
room_length = 7.0   # meters X
room_width = 5.0    # meters Y
speaker_height = 1.2  # meters Z
grid_res = 0.01  # grid resolution in meters

# Wall reflection parameters
reflection_coeff = 0.24  # 0 = fully absorptive, 1 = fully reflective, brick wall is between 0.18 and 0.36
max_order = 3        # number of reflections (order)

x = np.arange(0, room_length, grid_res)
y = np.arange(0, room_width, grid_res)
X, Y = np.meshgrid(x, y)

# --- Speaker setup ---
speaker_pos = np.array([2.5, 2.0, speaker_height])  # x, y, z
direction_angle = np.deg2rad(45)  # 0° faces +x

def compute_pressure(source_pos, source_angle, attenuation=1.0):
    dx = X - source_pos[0]
    dy = Y - source_pos[1]
    dist = np.sqrt(dx**2 + dy**2) + 1e-2

    theta = np.arctan2(dy, dx) - source_angle
    directivity = np.cos(theta)
    directivity[directivity < 0] = 0.05

    return attenuation * directivity / dist

# Initialize pressure map with directed sound
p_total = compute_pressure(speaker_pos, direction_angle)

# Function that generates image sources (pixels) recursively
def generate_image_sources(pos, angle, order):
    if order == 0:
        return [(pos, angle, 1.0)]
    
    images = []
    # Reflect across left/right walls (x)
    images.append((np.array([-pos[0], pos[1], pos[2]]), np.pi - angle, reflection_coeff))
    images.append((np.array([2*room_length - pos[0], pos[1], pos[2]]), np.pi - angle, reflection_coeff))
    
    # Reflect across bottom/top walls (y)
    images.append((np.array([pos[0], -pos[1], pos[2]]), -angle, reflection_coeff))
    images.append((np.array([pos[0], 2*room_width - pos[1], pos[2]]), -angle, reflection_coeff))
    
    # Recursively generate higher-order images
    if order > 1:
        higher_order_images = []
        for img_pos, img_angle, atten in images:
            higher_order_images += generate_image_sources(img_pos, img_angle, order-1)
        images += higher_order_images
    
    return images

# --- Generate and sum all reflections ---
all_images = generate_image_sources(speaker_pos, direction_angle, max_order)

for img_pos, img_angle, atten in all_images[1:]:  # skip the first entry (direct sound)
    p_total += compute_pressure(img_pos, img_angle, attenuation=atten)

# Normalize and convert to SPL
p_total /= np.max(p_total)
spl = 20 * np.log10(p_total + 1e-6)

# --- Plot heat map ---
plt.figure(figsize=(8, 5))
plt.imshow(spl, extent=(0, room_length, 0, room_width), origin='lower', cmap='inferno')
plt.colorbar(label='Relative SPL (dB)')
plt.scatter(speaker_pos[0], speaker_pos[1], c='cyan', s=60, label='Speaker')
plt.title('Simulated SPL Heat Map (Up to 3rd-Order Reflections)')
plt.xlabel('Room length (m)')
plt.ylabel('Room width (m)')
plt.legend()
plt.show()