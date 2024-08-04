import numpy as np
import vispy
from vispy import scene, app
from vispy.visuals.transforms import MatrixTransform
from torus import Torus
import colorsys
from vispy.color import Color
from vispy.app import Timer
from vispy.visuals.mesh import MeshVisual
from vispy.scene.visuals import create_visual_node
from vispy.scene.visuals import Line

# Create a custom visual that can show both filled and wireframe
TorusMeshVisual = create_visual_node(MeshVisual)

vispy.use('glfw')


def smooth_random_field(size):
    X, Y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    result = np.zeros((size, size))
    for i in range(5):
        freq = 2 ** i
        phase_x, phase_y = np.random.random(2) * 2 * np.pi
        result += np.sin(freq * 2 * np.pi * X + phase_x) * np.sin(freq * 2 * np.pi * Y + phase_y)
    result = (result - result.min()) / (result.max() - result.min())
    return result


def value_to_rgb(value):
    hue = 0.6 - 0.6 * value  # blue->teal->green->yellow transition
    return colorsys.hsv_to_rgb(hue, 0.8, 0.9)


class TorusViewer:
    def __init__(self, color_dim=100, vertex_dim=None):
        vertex_dim = color_dim if vertex_dim is None else vertex_dim
        self.color_dim = color_dim
        self.vertex_dim = vertex_dim

        # Initialize torus color data
        self.color_data = Torus(smooth_random_field(color_dim))

        # Set up scene
        self.canvas = scene.SceneCanvas(keys='interactive', size=(800, 600))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'arcball'
        self.view.camera.distance = 15
        self.view.bgcolor = Color('#303030')

        # Initially start with surface mode
        self.wireframe_mode = False

        # Generate torus vertices
        self.generate_geometry(int(vertex_dim))

        # Create mesh for surface
        self.mesh = TorusMeshVisual(vertices=self.vertices, faces=self.faces,
                                    shading='smooth', parent=self.view.scene)

        # Create lines for wireframe
        edges = self.get_edges_from_faces()
        self.wireframe = Line(pos=self.vertices[edges].reshape((-1, 3)),
                              color=(1, 1, 1, 0.5), parent=self.view.scene,
                              connect='segments')

        # Node for group transforms
        self.mesh_parent = scene.Node(parent=self.view.scene)
        self.mesh.parent = self.mesh_parent
        self.wireframe.parent = self.mesh_parent

        # Apply colors (white wireframe or surface with colors from the torus array data)
        self.update_colors()

        # Set up rotation
        self.theta, self.phi = 0, 0
        self.theta_speed, self.phi_speed = 0.001, 0.00025

        self.timer = Timer(interval=1/60, connect=self.rotate, start=True)

        self.mesh.transform = MatrixTransform()

        # Text instructions
        self.text = scene.visuals.Text(
            "Mouse: Left-drag rotates view | Scroll zooms\n"
            "Space: Toggle wireframe/surface | <-->: Horizontal rotation | ^v: Vertical rotation",
            color='white', font_size=10, pos=(self.canvas.size[0] // 2, 20), parent=self.canvas.scene)

        self.status_text = scene.visuals.Text(f"θ-speed: {self.theta_speed:.4f} | φ-speed: {self.phi_speed:.4f}", color='white', font_size=10,
                                             pos=(self.canvas.size[0] - 120, self.canvas.size[1] - 30), parent=self.canvas.scene)

        # Connect key press event
        self.canvas.events.key_press.connect(self.on_key_press)

    def generate_geometry(self, vertex_dim):
        R, r = 2, 1  # Major and minor radii
        phi, theta = np.mgrid[0:2 * np.pi:vertex_dim * 1j, 0:2 * np.pi:vertex_dim * 1j]
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        self.vertices = np.dstack((x, y, z)).reshape(-1, 3)

        # Generate faces
        idx = np.arange(vertex_dim ** 2).reshape(vertex_dim, vertex_dim)
        # Pre-allocate faces array
        self.faces = np.zeros((vertex_dim * vertex_dim * 2, 3), dtype=np.uint32)
        face_idx = 0
        for i in range(vertex_dim):
            for j in range(vertex_dim):
                sq = [idx[i, j], idx[(i + 1) % vertex_dim, j],
                      idx[(i + 1) % vertex_dim, (j + 1) % vertex_dim], idx[i, (j + 1) % vertex_dim]]
                self.faces[face_idx] = [sq[0], sq[1], sq[2]]
                self.faces[face_idx + 1] = [sq[0], sq[2], sq[3]]
                face_idx += 2

    def regenerate_geometry(self, vertex_dim):
        old_transform = self.mesh_parent.transform  # Note: changed from mesh to mesh_parent
        self.vertex_dim = vertex_dim
        self.generate_geometry(vertex_dim)

        # Update mesh
        self.mesh.set_data(vertices=self.vertices, faces=self.faces)

        # Update wireframe
        edges = self.get_edges_from_faces()
        self.wireframe.set_data(pos=self.vertices[edges].reshape((-1, 3)))

        self.mesh_parent.transform = old_transform
        self.update_colors()

    def get_edges_from_faces(self):
        edges = set()
        for face in self.faces:
            # The way Line with connect='segments' works, this is all we need for the wireframe
            # (the horizontal and vertical lines)
            edges.add(tuple(sorted([face[1], face[2]])))
        return np.array(list(edges))

    def update_colors(self):
        color_rows, color_cols = self.color_data.shape
        theta_idxs, phi_idxs = np.meshgrid(np.arange(color_rows), np.arange(color_cols), indexing='ij')
        flat_idx = theta_idxs.flatten()
        flat_phi = phi_idxs.flatten()

        raw_colors = np.array([value_to_rgb(self.color_data[t, p]) for t, p in zip(flat_idx, flat_phi)])

        # Interpolate colors to match vertex count
        old_idxs = np.linspace(0, raw_colors.shape[0], raw_colors.shape[0])
        new_idxs = np.linspace(0, raw_colors.shape[0], self.vertices.shape[0])
        colors = np.array([np.interp(new_idxs, old_idxs, raw_colors[:, i]) for i in range(3)]).T
        vertex_colors = np.c_[colors, np.ones(colors.shape[0])]

        if self.wireframe_mode:
            self.mesh.visible = False
            self.wireframe.set_data(color=(1, 1, 1, 1))  # Full opacity wireframe
        else:
            self.mesh.visible = True
            self.mesh.set_data(vertices=self.vertices, faces=self.faces, vertex_colors=vertex_colors)
            self.wireframe.set_data(color=(1, 1, 1, 0.3))  # Semi-transparent wireframe

        # Always keep wireframe visible
        self.wireframe.visible = True

    def on_key_press(self, event):
        if event.key == ' ':  # Spacebar
            self.wireframe_mode = not self.wireframe_mode
            self.update_colors()
        elif event.key in ['Left', 'Right', 'Up', 'Down']:
            speed_change = 0.00025
            if event.key == 'Left':
                self.theta_speed -= speed_change
            elif event.key == 'Right':
                self.theta_speed += speed_change
            elif event.key == 'Up':
                self.phi_speed += speed_change
            elif event.key == 'Down':
                self.phi_speed -= speed_change
        elif event.key in ['+', '=']:  # Both keys often share same button
            new_dim = min(int(self.vertex_dim * 1.25), self.color_dim * 10)
            self.regenerate_geometry(new_dim)
        elif event.key == '-':
            new_dim = max(int(self.vertex_dim * 0.8), self.color_dim // 10)
            self.regenerate_geometry(new_dim)

    def rotate(self, event):
        self.theta += self.theta_speed
        self.phi += self.phi_speed

        # Create a single transform
        transform = MatrixTransform()

        # Apply both rotations
        transform.rotate(np.degrees(self.theta), (0, 1, 0))  # Around Y-axis for better visual
        transform.rotate(np.degrees(self.phi), (1, 0, 0))  # Around X-axis

        # Set the transform to the parent node
        self.mesh_parent.transform = transform

        # Update rotation speed text
        self.status_text.text = (f"θ: {self.theta:.2f} (speed: {self.theta_speed:.4f})\n"
                                 f"φ: {self.phi:.2f} (speed: {self.phi_speed:.4f})")
        v_ratio = (self.vertex_dim ** 2) / (self.color_dim ** 2)
        self.status_text.text += f"\nQuality: {v_ratio:.2f}x (use +/- to change)"

    def run(self):
        self.canvas.show()
        app.run()


if __name__ == '__main__':
    TorusViewer().run()