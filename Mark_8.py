import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time 

dragging = False
last_update = 0
update_interval = 0.05 # 20fps max

# constants
c = 343.0 # speed of sound in m/s

# Room setup
room_length = 10.0   # meters X
room_width = 10.0    # meters Y
room_height = 3.0   # meters Z
speaker_height = 1.2
grid_res = 0.2   # coarser resolution for performance

# Reflection parameters
reflection_coeff = 0.7 # 0 = totaly absorbant, 1 = totaly reflective
max_order = 1 # number of reflections (2 works best)

# --- 3D Grid ---
x = np.arange(0, room_length, grid_res)
y = np.arange(0, room_width, grid_res)
z = np.arange(0, room_height, grid_res)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')  # shape (Nx, Ny, Nz)

# --- Speaker ---
speaker_pos = np.array([2.0, 2.0, speaker_height])
direction_angle = np.deg2rad(45)  # 0° faces +x

# --- Listener slice ---
listener_height = 1.2
z_index = np.argmin(np.abs(z - listener_height))

# --- Pressure computation --- (updated for visible SPL)
def compute_pressure_3d_complex(source_pos, source_angle, k, attenuation=1.0):
    dx = X - source_pos[0]
    dy = Y - source_pos[1]
    dz = Z - source_pos[2]

    distance = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6  # full 3D distance

    dir_vector = np.stack((dx, dy, dz), axis=-1)
    dir_unit = dir_vector / distance[..., np.newaxis]

    forward = np.array([np.cos(source_angle), # x
                        np.sin(source_angle), # y
                        0.0]) # z - for tilt implementation later

    cos_angle = (dir_unit[..., 0] * forward[0] +
                 dir_unit[..., 1] * forward[1] +
                 dir_unit[..., 2] * forward[2] )

    # --- Adjusted directivity to avoid near-zero everywhere ---
    #directivity = np.clip(cos_angle, 0.0, 1.0)**0.5
    directivity = ((1.0 + cos_angle) / 2.0) ** 0.5

    # Phase term
    phase = np.exp(1j * k * distance)

    # air absorption
    air_absorption = np.exp(-0.01 * distance)

    # Complex pressure field
    return attenuation * directivity * phase * air_absorption / distance


# --- Image source generator --- (updated to preserve attenuation)
def generate_image_sources_3d(position, angle, order, current_attentuation=1.0):
    if order == 0:
        return [(position, angle, current_attentuation)]

    images = []

    # reflect helper (keeps attenuation positive)
    def reflect(new_position, new_angle, current_attentuation):
        return (new_position, new_angle, current_attentuation * reflection_coeff)

    # Generate first-order image sources
    images.extend([
        reflect(np.array([-position[0], position[1], position[2]]), np.pi - angle, current_attentuation),
        reflect(np.array([2*room_length - position[0], position[1], position[2]]), np.pi - angle, current_attentuation),
        reflect(np.array([position[0], -position[1], position[2]]), -angle, current_attentuation),
        reflect(np.array([position[0], 2*room_width - position[1], position[2]]), -angle, current_attentuation),
        reflect(np.array([position[0], position[1], -position[2]]), angle, current_attentuation),
        reflect(np.array([position[0], position[1], 2*room_height - position[2]]), angle, current_attentuation)
    ])

    # Recursively add higher-order reflections
    if order > 1:
        higher = []
        for pos, ang, atten in images:
            # Keep all meaningful reflections, lower pruning threshold
            if atten > 0.05:
                higher += generate_image_sources_3d(pos, ang, order-1, atten)
        images += higher

    return images

# --- Compute field ---
def compute_field (frequency, direction_angle, max_order):
    wavelength = c / frequency
    k = 2 * np.pi / wavelength

    p_total = np.zeros_like(X, dtype=complex)

    all_sources = generate_image_sources_3d(speaker_pos, direction_angle, max_order)

    for pos, angle, atten in all_sources:
        if atten > 0.01:
            p_total += compute_pressure_3d_complex(pos, angle, k, atten)

    # Convert physical pressure
    p_mag = np.abs(p_total)

    # Convert to SPL-like dB
    spl = 20 * np.log10(p_mag + 1e-12)
    spl = np.clip(spl, -60, 0)

    return spl

# --- Initial values for variables ---
frequency = 200.0
direction_angle = np.deg2rad(45)#
max_order = 1

spl = compute_field(frequency, direction_angle, max_order)

# --- Plot ---
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.3)

img = ax.imshow(spl[:, :, z_index].T,
                extent=(0, room_length, 0, room_width),
                origin='lower',
                cmap='inferno',
                aspect='auto',)

plt.colorbar(img, ax=ax, label='Relative SPL (dB)')

speaker_plot = ax.scatter(speaker_pos[0], speaker_pos[1], c='cyan', s=60, label='Speaker', picker=True)

ax.set_title(f'Wave Interference SPL')
ax.set_xlabel('Room length (m)')
ax.set_ylabel('Room width (m)')
ax.legend()

# --- Sliders ---
ax_freq = plt.axes([0.2, 0.2, 0.6, 0.03])
ax_angle = plt.axes([0.2, 0.15, 0.6, 0.03])
ax_order = plt.axes([0.2, 0.1, 0.6, 0.03])

s_freq = Slider(ax_freq, 'Freq (Hz)', 50, 500, valinit=frequency)
s_angle = Slider(ax_angle, 'Angle (deg)', 0, 360, valinit=45)
s_order = Slider(ax_order, 'Reflections', 0, 3, valinit=max_order, valstep=1)

# --- Update ---
def update(val):
    global frequency, direction_angle, max_order

    frequency = s_freq.val
    direction_angle = np.deg2rad(s_angle.val)
    max_order = int(s_order.val)

    new_spl = compute_field(frequency, direction_angle, max_order)

    img.set_data(new_spl[:, :, z_index].T)
    ax.set_title(f'Wave Interference SPL (f={frequency:.1f} Hz, order={max_order})')

    fig.canvas.draw_idle()

s_freq.on_changed(update)
s_angle.on_changed(update)
s_order.on_changed(update)

plt.show()