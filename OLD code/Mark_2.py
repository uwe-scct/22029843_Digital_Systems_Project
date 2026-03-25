import numpy as np
import matplotlib.pyplot as plt

# Room setup
room_length = 7.0   # meters X
room_width = 5.0    # meters Y
speaker_height = 1.2  # meters Z
grid_res = 0.05  # grid resolution in meters

# Wall reflection parameters
reflection_coeff = 0.7  # 0 = fully absorptive, 1 = fully reflective
max_order = 1           # first-order reflections

x = np.arange(0, room_length, grid_res)
y = np.arange(0, room_width, grid_res)
X, Y = np.meshgrid(x, y)

# --- Speaker setup ---
speaker_pos = np.array([2.5, 2.0, speaker_height])  # (x, y, z)
direction_angle = np.deg2rad(45)  # degrees -> radians, 0° faces +x

def compute_pressure(source_pos, source_angle, attenuation=1.0):
    dx = X - source_pos[0]
    dy = Y - source_pos[1]
    dist = np.sqrt(dx**2 + dy**2) + 1e-2

    theta = np.arctan2(dy, dx) - source_angle
    directivity = np.cos(theta)
    directivity[directivity < 0] = 0.025

    return attenuation * directivity / dist

p_total = compute_pressure(speaker_pos, direction_angle)

image_sources = [
    # left wall
    (np.array([-speaker_pos[0], speaker_pos[1], speaker_height]), np.pi - direction_angle),
    # right wall
    (np.array([2 * room_length - speaker_pos[0], speaker_pos[1], speaker_height]), np.pi - direction_angle),
    # bottom wall 
    (np.array([speaker_pos[0], -speaker_pos[1], speaker_height]), -direction_angle),
    # top wall
    (np.array([speaker_pos[0], 2 * room_width - speaker_pos[1], speaker_height]), -direction_angle)
]

for img_pos, img_angle in image_sources:
    p_total += compute_pressure(img_pos, img_angle, attenuation=reflection_coeff)

p_total /= np.max(p_total)
# Convert to SPL (0 dB = max pressure)
spl = 20 * np.log10(p_total + 1e-6)

# --- Plot heat map ---
plt.figure(figsize=(8, 5))
plt.imshow(spl, extent=(0, room_length, 0, room_width), origin='lower', cmap='inferno')
plt.colorbar(label='Relative SPL (dB)')
plt.scatter(speaker_pos[0], speaker_pos[1], c='cyan', s=60, label='Speaker')
plt.title('Simulated SPL Heat Map (Direct + First-Order Reflections)')
plt.xlabel('Room length (m)')
plt.ylabel('Room width (m)')
plt.legend()
plt.show()