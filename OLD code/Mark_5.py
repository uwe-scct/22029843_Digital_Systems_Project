import numpy as np
import matplotlib.pyplot as plt

# wavelength constants
c = 343.0 # speed of sound in m/s
frequency = 200.0 # Hz
wavelength = c / frequency
k = 2 * np.pi / wavelength # wave number

# Room setup
room_length = 7.0   # meters X
room_width = 5.0    # meters Y
room_height = 3.0   # meters Z
speaker_height = 1.2
grid_res = 0.2     # coarser resolution for 3D in meters

# Reflection parameters
reflection_coeff = 0.7 # 0 = totaly absorbant, 1 = totaly reflective
max_order = 2 # number of reflections (3 works best)

# --- 3D Grid ---
x = np.arange(0, room_length, grid_res)
y = np.arange(0, room_width, grid_res)
z = np.arange(0, room_height, grid_res)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # shape (Nx, Ny, Nz)

# --- Speaker ---
speaker_pos = np.array([2.5, 2.0, speaker_height])
direction_angle = np.deg2rad(45)  # 0° faces +x

# --- Pressure computation --- NEW!!!
def compute_pressure_3d_complex(source_pos, source_angle, attenuation=1.0):
    dx = X - source_pos[0]
    dy = Y - source_pos[1]
    dz = Z - source_pos[2]

    distance = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6  # full 3D distance

    # Horizontal only directionality
    theta = np.arctan2(dy, dx) - source_angle
    directivity = np.cos(theta)
    directivity[directivity < 0] = 0.0

    # Phase term
    phase = np.exp(1j * k * distance)

    # Complex pressure field
    return attenuation * directivity * phase / distance

# Function that generates image sources (pixels) recursively with keeping attentuation
def generate_image_sources_3d(pos, angle, order, current_attentuation=1.0):
    if order == 0:
        return [(pos, angle, current_attentuation)]
    
    images = []

    def reflect(new_pos, new_angle):
        return (new_pos, new_angle, current_attentuation * reflection_coeff)

    images.extend([
        reflect(np.array([-pos[0], pos[1], pos[2]]), np.pi - angle),
        reflect(np.array([2*room_length - pos[0], pos[1], pos[2]]), np.pi - angle),
        reflect(np.array([pos[0], -pos[1], pos[2]]), -angle),
        reflect(np.array([pos[0], 2*room_width - pos[1], pos[2]]), -angle),
        reflect(np.array([pos[0], pos[1], -pos[2]]), angle),
        reflect(np.array([pos[0], pos[1], 2*room_height - pos[2]]), angle)
    ])

    if order > 1:
        higher = []
        for p, a, att in images:
            higher += generate_image_sources_3d(p, a, order-1, att)
        images += higher

    return images

# --- Compute total complex pressure ---
p_total = np.zeros_like(X, dtype=complex)

all_sources = generate_image_sources_3d(speaker_pos, direction_angle, max_order)

for pos, angle, atten in all_sources:
    p_total += compute_pressure_3d_complex(pos, angle, atten)

# --- Convert physical pressure ---
p_mag = np.abs(p_total)

# Normalize for visualization
p_mag /= np.max(p_mag)

# Convert to SPL-like dB
spl = 20 * np.log10(p_mag + 1e-6)

# --- Plot horizontal slice ---
listener_height = 1.2
z_index = np.argmin(np.abs(z - listener_height))

plt.figure(figsize=(8, 5))
plt.imshow(spl[:, :, z_index].T,
           extent=(0, room_length, 0, room_width),
           origin='lower',
           cmap='inferno',
           aspect='auto')

plt.colorbar(label='Relative SPL (dB)')
plt.scatter(speaker_pos[0], speaker_pos[1], c='cyan', s=60, label='Speaker')

plt.title(f'Wave Interference SPL (f={frequency} Hz, order={max_order})')
plt.xlabel('Room length (m)')
plt.ylabel('Room width (m)')
plt.legend()
plt.show()