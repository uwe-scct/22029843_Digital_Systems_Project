import numpy as np
import matplotlib.pyplot as plt

# Room setup
room_length = 7.0   # meters X
room_width = 5.0    # meters Y
room_height = 3.0   # meters Z
speaker_height = 1.2
grid_res = 0.2      # coarser resolution for 3D in meters

# Reflection parameters
reflection_coeff = 0.24 # 0 = totaly absorbant, 1 = totaly reflective, brick wall is 0.18 - 0.36

# --- 3D Grid ---
x = np.arange(0, room_length, grid_res)
y = np.arange(0, room_width, grid_res)
z = np.arange(0, room_height, grid_res)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # shape (Nx, Ny, Nz)

# --- Speaker ---
speaker_pos = np.array([2.5, 2.0, speaker_height])
direction_angle = np.deg2rad(45)  # 0° faces +x

# --- Pressure computation ---
def compute_pressure_3d(source_pos, source_angle, attenuation=1.0):
    dx = X - source_pos[0]
    dy = Y - source_pos[1]
    dz = Z - source_pos[2]
    dist = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-2  # full 3D distance

    # simplified planar directivity
    theta = np.arctan2(dy, dx) - source_angle
    directivity = np.cos(theta)
    directivity[directivity < 0] = 0.0

    return attenuation * directivity / dist

# Function that generates image sources (pixels) recursively
def generate_image_sources_3d(pos, angle, order):
    if order == 0:
        return [(pos, angle, 1.0)]
    
    images = []
    # Reflect walls x
    images.append((np.array([-pos[0], pos[1], pos[2]]), np.pi - angle, reflection_coeff))
    images.append((np.array([2*room_length - pos[0], pos[1], pos[2]]), np.pi - angle, reflection_coeff))
    # Reflect walls y
    images.append((np.array([pos[0], -pos[1], pos[2]]), -angle, reflection_coeff))
    images.append((np.array([pos[0], 2*room_width - pos[1], pos[2]]), -angle, reflection_coeff))
    # Reflect floor/ceiling z
    images.append((np.array([pos[0], pos[1], -pos[2]]), angle, reflection_coeff))
    images.append((np.array([pos[0], pos[1], 2*room_height - pos[2]]), angle, reflection_coeff))

    if order > 1:
        higher_order_images = []
        for img_pos, img_angle, atten in images:
            higher_order_images += generate_image_sources_3d(img_pos, img_angle, order-1)
        images += higher_order_images
    
    return images

# --- Compute SPL ---
p_total = compute_pressure_3d(speaker_pos, direction_angle)  # direct sound
all_images = generate_image_sources_3d(speaker_pos, direction_angle, max_order)

for img_pos, img_angle, atten in all_images[1:]:  # skip direct
    p_total += compute_pressure_3d(img_pos, img_angle, attenuation=atten)

# Normalize and convert to SPL
p_total /= np.max(p_total)
spl = 20 * np.log10(p_total + 1e-6)

# --- Visualization: horizontal slice at listener height ---
listener_height = 1.2
z_index = np.argmin(np.abs(z - listener_height))

plt.figure(figsize=(8, 5))
plt.imshow(spl[:, :, z_index].T, extent=(0, room_length, 0, room_width),
           origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(label='Relative SPL (dB)')
plt.scatter(speaker_pos[0], speaker_pos[1], c='cyan', s=60, label='Speaker')
plt.title(f'3D SPL Heat Map at z={listener_height} m (order {max_order})')
plt.xlabel('Room length (m)')
plt.ylabel('Room width (m)')
plt.legend()
plt.show()

# --- Optional: vertical slice along x=2.5 m ---
x_slice = 2.5
x_index = np.argmin(np.abs(x - x_slice))
plt.figure(figsize=(5, 5))
plt.imshow(spl[x_index, :, :].T, extent=(0, room_width, 0, room_height),
           origin='lower', cmap='inferno', aspect='auto')
plt.colorbar(label='Relative SPL (dB)')
plt.title(f'Vertical SPL Slice at x={x_slice} m')
plt.xlabel('Room width (m)')
plt.ylabel('Height (m)')
plt.show()

X_flat = X.flatten()
Y_flat = Y.flatten()
Z_flat = Z.flatten()
spl_flat = spl.flatten()

# Normalize SPL for color mapping
norm_spl = (spl_flat - spl_flat.min()) / (spl_flat.max() - spl_flat.min())

# --- 3D Scatter Plot ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(X_flat, Y_flat, Z_flat, c=spl_flat, cmap='inferno', marker='o', s=5, alpha=0.5)
ax.scatter(speaker_pos[0], speaker_pos[1], speaker_pos[2], c='cyan', s=100, label='Speaker')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D SPL Distribution in Room')
fig.colorbar(sc, label='Relative SPL (dB)')
ax.legend()
plt.show()