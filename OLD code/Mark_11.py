import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox, RadioButtons  ### EDITED
import time 

# --- Physics functions remain outside the GUI class ---

# constants
c = 343.0 # speed of sound in m/s

# Room setup
room_length = 10.0   # meters X
room_width = 10.0    # meters Y
room_height = 3.0    # meters Z
speaker_height = 1.2
grid_res = 0.2   # coarser resolution for performance

# Reflection parameters
reflection_coeff = 0.7 # 0 = totally absorbent, 1 = totally reflective
max_order_default = 1 # number of reflections (2 works best)

# 3D Grid 
def generate_grid():  # New
    global x, y, z, X, Y, Z
    x = np.arange(0, room_length, grid_res)
    y = np.arange(0, room_width, grid_res)
    z = np.arange(0, room_height, grid_res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

generate_grid()  

# Speaker positioning class
class Speaker:
    def __init__(self, position, angle, arrow_length=0.4):
        self.position = position
        self.angle = angle
        self.arrow_length = arrow_length
        self.plot = None

    def draw(self, ax):
        self.plot = ax.arrow(self.position[0], self.position[1], 
                             self.arrow_length * np.cos(self.angle), 
                             self.arrow_length * np.sin(self.angle), 
                             color='cyan', width=0.1, label='Speaker')

    def update_plot(self, ax):
        if self.plot:
            self.plot.remove()
        self.draw(ax)

    def set_position(self, x, y):
        x = np.clip(x, 0, room_length)
        y = np.clip(y, 0, room_width)
        self.position[0] = x
        self.position[1] = y

    def set_angle(self, angle):
        self.angle = angle

# Physics Functions

def compute_pressure_3d_complex(source_pos, source_angle, k, attenuation=1.0):
    dx = X - source_pos[0]
    dy = Y - source_pos[1]
    dz = Z - source_pos[2]

    distance = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
    dir_vector = np.stack((dx, dy, dz), axis=-1)
    dir_unit = dir_vector / distance[..., np.newaxis]

    forward = np.array([np.cos(source_angle), np.sin(source_angle), 0.0])
    cos_angle = (dir_unit[..., 0] * forward[0] +
                 dir_unit[..., 1] * forward[1] +
                 dir_unit[..., 2] * forward[2] )

    directivity = ((1.0 + cos_angle) / 2.0) ** 0.5
    phase = np.exp(1j * k * distance)
    air_absorption = np.exp(-0.01 * distance)

    return attenuation * directivity * phase * air_absorption / distance

def generate_image_sources_3d(position, angle, order, current_attentuation=1.0):
    if order == 0:
        return [(position, angle, current_attentuation)]

    images = []

    def reflect(new_position, new_angle, current_attentuation):
        return (new_position, new_angle, current_attentuation * reflection_coeff)

    images.extend([
        reflect(np.array([-position[0], position[1], position[2]]), np.pi - angle, current_attentuation),
        reflect(np.array([2*room_length - position[0], position[1], position[2]]), np.pi - angle, current_attentuation),
        reflect(np.array([position[0], -position[1], position[2]]), -angle, current_attentuation),
        reflect(np.array([position[0], 2*room_width - position[1], position[2]]), -angle, current_attentuation),
        reflect(np.array([position[0], position[1], -position[2]]), angle, current_attentuation),
        reflect(np.array([position[0], position[1], 2*room_height - position[2]]), angle, current_attentuation)
    ])

    if order > 1:
        higher = []
        for pos, ang, atten in images:
            if atten > 0.05:
                higher += generate_image_sources_3d(pos, ang, order-1, atten)
        images += higher

    return images

def compute_field(frequency, direction_angle, max_order, speaker_pos):

    wavelength = c / frequency
    k = 2 * np.pi / wavelength

    p_total = np.zeros_like(X, dtype=complex)
    all_sources = generate_image_sources_3d(speaker_pos, direction_angle, max_order)

    for pos, angle, atten in all_sources:
        if atten > 0.01:
            p_total += compute_pressure_3d_complex(pos, angle, k, atten)

    p_mag = np.abs(p_total)
    spl = 20 * np.log10(p_mag + 1e-12)
    spl = np.clip(spl, -60, 0)
    return spl

# GUI class encapsulating all state 
class AcousticGUI:
    def __init__(self):
        self.dragging = False
        self.last_update = 0
        self.update_interval = 0.05

        self.frequency = 200.0
        self.direction_angle = np.deg2rad(45)
        self.max_order = max_order_default

        self.speaker = Speaker(np.array([2.0, 2.0, speaker_height]), self.direction_angle)

        self.listener_height = 1.2
        self.z_index = np.argmin(np.abs(z - self.listener_height))

        self.build_gui()

    def build_gui(self):
        if hasattr(self, 'fig'):  
            plt.close(self.fig)   

        self.fig, self.ax = plt.subplots(figsize=(8, 6)) 
        self.resize_figure() 
        plt.subplots_adjust(left=0.25, bottom=0.45) 

        self.spl = self.compute_field()
        self.img = self.ax.imshow(self.spl[:, :, self.z_index].T,
                                  extent=(0, room_length, 0, room_width),
                                  origin='lower',
                                  cmap='inferno',
                                  aspect='equal',)
        plt.colorbar(self.img, ax=self.ax, label='Relative SPL (dB)')
        self.speaker.draw(self.ax)

        self.ax.set_title(f'Wave Interference SPL')
        self.ax.set_xlabel('Room length (m)')
        self.ax.set_ylabel('Room width (m)')
        self.ax.legend()

        # sliders
        ax_freq = plt.axes([0.25, 0.35, 0.5, 0.02])
        ax_angle = plt.axes([0.25, 0.31, 0.5, 0.02])
        ax_order = plt.axes([0.25, 0.27, 0.5, 0.02])
        ax_listener = plt.axes([0.25, 0.23, 0.5, 0.02]) 

        self.s_freq = Slider(ax_freq, 'Freq (Hz)', 20, 20000, valinit=self.frequency)
        self.s_angle = Slider(ax_angle, 'Angle (deg)', 0, 360, valinit=np.rad2deg(self.direction_angle))
        self.s_order = Slider(ax_order, 'Reflections', 0, 3, valinit=self.max_order, valstep=1)
        self.s_listener = Slider(ax_listener, 'Listener Height', 0.1, room_height, valinit=self.listener_height)  

        # Spinboxes (TextBox)
        ax_len = plt.axes([0.05, 0.32, 0.12, 0.035])
        ax_wid = plt.axes([0.05, 0.26, 0.12, 0.035])
        ax_hei = plt.axes([0.05, 0.20, 0.12, 0.035])

        self.tb_len = TextBox(ax_len, 'Length', initial=str(room_length))  
        self.tb_wid = TextBox(ax_wid, 'Width', initial=str(room_width))
        self.tb_hei = TextBox(ax_hei, 'Height', initial=str(room_height))

        # Reflection material selector
        ax_reflect = plt.axes([0.05, 0.05, 0.12, 0.07])  
        self.radio = RadioButtons(ax_reflect,
                                 ('plywood', 'Brick wall', 'corrugated iron wall', 'Plaster board')) 

        # events
        self.s_freq.on_changed(self.update)
        self.s_angle.on_changed(self.update)
        self.s_order.on_changed(self.update)
        self.s_listener.on_changed(self.update) 

        self.tb_len.on_submit(self.update_room)  
        self.tb_wid.on_submit(self.update_room)
        self.tb_hei.on_submit(self.update_room)

        self.radio.on_clicked(self.update_material)  

        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def resize_figure(self):  # New
        base_size = 10  # controls overall size
        aspect_ratio = room_length / room_width

        if aspect_ratio >= 1:
            width = base_size * aspect_ratio
            height = base_size
        else:
            width = base_size
            height = base_size / aspect_ratio

        self.fig.set_size_inches(width, height, forward=True)

    def update_room(self, text):  # New
        global room_length, room_width, room_height
        try:
            room_length = np.clip(float(self.tb_len.text), 2, 100)
            room_width = np.clip(float(self.tb_wid.text), 2, 100)
            room_height = np.clip(float(self.tb_hei.text), 2, 30)
            generate_grid()
            self.z_index = np.argmin(np.abs(z - self.listener_height))

            # keep speaker inside new room
            self.speaker.set_position(self.speaker.position[0], self.speaker.position[1])

            # update plot extent
            self.img.set_extent((0, room_length, 0, room_width))
            self.ax.set_xlim(0, room_length)
            self.ax.set_ylim(0, room_width)

            # recompute field
            new_spl = compute_field(self.frequency, self.direction_angle, self.max_order, self.speaker.position)
            self.img.set_data(new_spl[:, :, self.z_index].T)
            self.resize_figure()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events() 
        except:
            pass

    def update_material(self, label):  # New
        global reflection_coeff
        if label == 'plywood':
            reflection_coeff = 0.7
        elif label == 'Brick wall':
            reflection_coeff = 0.95
        elif label == 'corrugated iron wall':
            reflection_coeff = 1.0
        elif label == 'Plaster board':
            reflection_coeff = 0.9
        self.update(None)

    def compute_field(self):
        return compute_field(self.frequency, self.direction_angle, self.max_order, self.speaker.position)

    def update(self, val):
        self.frequency = self.s_freq.val
        self.direction_angle = np.deg2rad(self.s_angle.val)
        self.max_order = int(self.s_order.val)
        self.listener_height = self.s_listener.val  

        self.z_index = np.argmin(np.abs(z - self.listener_height))  

        self.speaker.set_angle(self.direction_angle)
        self.speaker.update_plot(self.ax)

        new_spl = compute_field(self.frequency, self.direction_angle, self.max_order, self.speaker.position)
        self.img.set_data(new_spl[:, :, self.z_index].T)

        self.ax.set_title(f'Wave Interference SPL (f={self.frequency:.1f} Hz, order={self.max_order})')
        self.fig.canvas.draw_idle()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        dist = np.hypot(event.xdata - self.speaker.position[0], event.ydata - self.speaker.position[1])
        if dist < 0.5:
            self.dragging = True

    def on_release(self, event):
        self.dragging = False

    def on_motion(self, event):
        if not self.dragging or event.inaxes != self.ax:
            return

        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return

        self.last_update = current_time
        self.speaker.set_position(event.xdata, event.ydata)
        self.speaker.update_plot(self.ax)

        new_spl = compute_field(self.frequency, self.direction_angle, self.max_order, self.speaker.position)
        self.img.set_data(new_spl[:, :, self.z_index].T)
        self.fig.canvas.draw_idle()

# Launch the GUI
gui = AcousticGUI()
plt.show()