import numpy as np
import matplotlib.pyplot as plt

# --- Room setup ---
room_length = 7.0   # meters (X)
room_width = 5.0    # meters (Y)
speaker_height = 1.2  # meters
grid_res = 0.05  # grid resolution in meters

x = np.arange(0, room_length, grid_res)
y = np.arange(0, room_width, grid_res)
X, Y = np.meshgrid(x, y)

# --- Speaker setup ---
speaker_pos = np.array([2.5, 2.0, speaker_height])  # (x, y, z)
direction_angle = np.deg2rad(45)  # degrees -> radians, 0° faces +x

# --- Compute directionality ---
dx = X - speaker_pos[0]
dy = Y - speaker_pos[1]
dist = np.sqrt(dx**2 + dy**2) + 1e-2 # avoid division by zero

# Angle between speaker direction and point
theta = np.arctan2(dy, dx) - direction_angle
directivity = np.cos(theta)
directivity[directivity < 0] = 0.05  # back of speaker = no sound

# --- Simple sound pressure model ---
p = directivity / dist  # 1/r decay
p /= np.max(p)  # normalize to max = 1

# Convert to SPL (0 dB = max pressure)
spl = 20 * np.log10(p + 1e-6)

# --- Plot heat map ---
plt.figure(figsize=(8, 5))
plt.imshow(spl, extent=(0, room_length, 0, room_width), origin='lower', cmap='inferno')
plt.colorbar(label='Relative SPL (dB)')
plt.scatter(speaker_pos[0], speaker_pos[1], c='cyan', s=60, label='Speaker')
plt.title('Simulated Sound Pressure Heat Map (Simplified)')
plt.xlabel('Room length (m)')
plt.ylabel('Room width (m)')
plt.legend()
plt.show()