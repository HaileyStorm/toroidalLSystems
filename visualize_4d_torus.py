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

#vispy.use('glfw')
vispy.use('PyQt6')


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
        self.view.camera.distance = 3
        self.view.bgcolor = Color('#303030')

        # Initially start with surface mode
        self.wireframe_mode = False

        # 4D rotation angles
        self.theta1, self.theta2, self.theta3, self.theta4 = 0, 0, 0, 0
        self.theta1_speed, self.theta2_speed = 0.00005, 0.00005

        # Projection parameters
        self.w_scale = 1.0  # Scale factor for w dimension before projection

        # Generate torus vertices
        self.generate_geometry(int(vertex_dim))

        # Create mesh for surface
        self.mesh = TorusMeshVisual(vertices=self.vertices, faces=self.faces,
                                    shading='flat', parent=self.view.scene)

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
        self.theta_speed, self.phi_speed = 0.0005, 0.0001

        self.timer = Timer(interval=1/60, connect=self.rotate, start=True)

        self.mesh.transform = MatrixTransform()

        # Text instructions
        self.text = scene.visuals.Text(
            "Mouse: Left-drag rotates view | Scroll zooms\n"
            "Space: Toggle wireframe/surface | <-->: Horizontal rotation | ^v: Vertical rotation",
            color='white', font_size=10, pos=(self.canvas.size[0] // 2, 60), parent=self.canvas.scene)

        self.status_text = scene.visuals.Text(f"θ-speed: {self.theta_speed:.6f} | φ-speed: {self.phi_speed:.6f}", color='white', font_size=10,
                                             pos=(self.canvas.size[0] - 120, self.canvas.size[1] - 60), parent=self.canvas.scene)

        # Connect key press event
        self.canvas.events.key_press.connect(self.on_key_press)

        # Update text instructions
        self.text.text = (
            "Mouse: Left-drag rotates view | Scroll zooms\n"
            "Space: Toggle wireframe/surface\n"
            "<-->: θ1 rotation | ^v: θ2 rotation\n"
            "Q/E: Rotate θ3 | A/D: Rotate θ4\n"
            "W/S: Adjust w-scale\n"
            "+/-: Adjust quality"
        )

    def generate_geometry(self, vertex_dim):
        R, r = 2, 1  # Major and minor radii
        alpha, beta = np.mgrid[0:2 * np.pi:vertex_dim * 1j, 0:2 * np.pi:vertex_dim * 1j]
        gamma, delta = np.mgrid[0:2 * np.pi:vertex_dim * 1j, 0:2 * np.pi:vertex_dim * 1j]

        # Generate 4D coordinates
        x = (R + r * np.cos(alpha)) * np.cos(beta)
        y = (R + r * np.cos(alpha)) * np.sin(beta)
        z = r * np.sin(alpha) * np.cos(gamma)
        w = r * np.sin(alpha) * np.sin(gamma)

        self.vertices_4d = np.dstack((x, y, z, w)).reshape(-1, 4)

        # Generate 3D vertices through projection
        self.project_4d_to_3d()

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

    def project_4d_to_3d(self):
        # Simple perspective projection from 4D to 3D
        w_adjusted = self.vertices_4d[:, 3] * self.w_scale + 5  # Add offset to avoid division by zero
        self.vertices = self.vertices_4d[:, :3] / w_adjusted[:, np.newaxis]

    def get_edges_from_faces(self):
        edges = set()
        for face in self.faces:
            # The way Line with connect='segments' works, this is all we need for the wireframe
            # (the horizontal and vertical lines)
            edges.add(tuple(sorted([face[1], face[2]])))
        return np.array(list(edges))

    def update_colors(self):
        color_rows, color_cols = self.color_data.shape
        vertex_rows, vertex_cols = self.vertex_dim, self.vertex_dim

        # Map 4D coordinates to 2D color indices
        # We'll use alpha and beta angles for this
        alpha = np.arctan2(self.vertices_4d[:, 2], self.vertices_4d[:, 0]).reshape(vertex_rows, vertex_cols)
        beta = np.arctan2(self.vertices_4d[:, 3], self.vertices_4d[:, 1]).reshape(vertex_rows, vertex_cols)

        # Normalize to [0, 1]
        alpha_norm = (alpha + np.pi) / (2 * np.pi)
        beta_norm = (beta + np.pi) / (2 * np.pi)

        # Map to color indices
        color_x = (alpha_norm * color_cols).astype(int) % color_cols
        color_y = (beta_norm * color_rows).astype(int) % color_rows

        # Generate colors for each grid cell
        raw_colors = np.array([[value_to_rgb(self.color_data[color_y[i, j], color_x[i, j]])
                                for j in range(vertex_cols)]
                               for i in range(vertex_rows)])

        # Reshape colors to match face count (2 faces per grid cell)
        face_colors = np.repeat(raw_colors.reshape(-1, 3), 2, axis=0)
        face_colors = np.c_[face_colors, np.ones(face_colors.shape[0])]

        if self.wireframe_mode:
            self.mesh.visible = False
            self.wireframe.set_data(color=(1, 1, 1, 1))  # Full opacity wireframe
        else:
            self.mesh.visible = True
            self.mesh.set_data(vertices=self.vertices, faces=self.faces, face_colors=face_colors)
            self.wireframe.set_data(color=(1, 1, 1, 0.3))  # Semi-transparent wireframe

        # Always keep wireframe visible
        self.wireframe.visible = True

    def on_key_press(self, event):
        if event.key == ' ':  # Spacebar
            self.wireframe_mode = not self.wireframe_mode
            self.update_colors()
        elif event.key in ['Left', 'Right', 'Up', 'Down']:
            speed_change = 0.00001
            if event.key == 'Left':
                self.theta1_speed -= speed_change
            elif event.key == 'Right':
                self.theta1_speed += speed_change
            elif event.key == 'Up':
                self.theta2_speed += speed_change
            elif event.key == 'Down':
                self.theta2_speed -= speed_change
        elif event.key in ['+', '=']:  # Both keys often share same button
            new_dim = min(int(round(self.vertex_dim * 1.25)), self.color_dim * 10)
            self.regenerate_geometry(new_dim)
        elif event.key == '-':
            new_dim = max(int(round(self.vertex_dim * 0.8)), self.color_dim // 10)
            self.regenerate_geometry(new_dim)
        elif event.key == 'W':
            self.w_scale *= 1.1
        elif event.key == 'S':
            self.w_scale /= 1.1
        elif event.key == 'Q':
            self.theta3 += 0.001
        elif event.key == 'E':
            self.theta3 -= 0.001
        elif event.key == 'A':
            self.theta4 += 0.0005
        elif event.key == 'D':
            self.theta4 -= 0.0005

    def rotate_4d(self):
        # 4D rotation matrices
        def R4_xy(theta): return np.array([
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        def R4_xz(theta): return np.array([
            [np.cos(theta), 0, -np.sin(theta), 0],
            [0, 1, 0, 0],
            [np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

        def R4_xw(theta): return np.array([
            [np.cos(theta), 0, 0, -np.sin(theta)],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [np.sin(theta), 0, 0, np.cos(theta)]
        ])

        def R4_yz(theta): return np.array([
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ])

        # Apply rotations
        vertices = self.vertices_4d.copy()
        vertices = vertices @ R4_xy(self.theta1).T
        vertices = vertices @ R4_xz(self.theta2).T
        vertices = vertices @ R4_yz(self.theta3).T
        vertices = vertices @ R4_xw(self.theta4).T

        self.vertices_4d = vertices
        self.project_4d_to_3d()

        # Update mesh and wireframe
        self.mesh.set_data(vertices=self.vertices, faces=self.faces)
        edges = self.get_edges_from_faces()
        self.wireframe.set_data(pos=self.vertices[edges].reshape((-1, 3)))

        self.update_colors()

    def rotate(self, event):
        # Update rotation angles
        self.theta1 += self.theta1_speed
        self.theta2 += self.theta2_speed

        self.rotate_4d()

        # Update status text
        self.status_text.text = (
            f"θ1: {self.theta1:.6f} ({self.theta1_speed:.6f})\n"
            f"θ2: {self.theta2:.6f} ({self.theta2_speed:.6f})\n"
            f"θ3: {self.theta3:.6f}\n"
            f"θ4: {self.theta4:.6f}\n" 
            f"w-scale: {self.w_scale:.4f}"
        )
        v_ratio = (self.vertex_dim ** 2) / (self.color_dim ** 2)
        self.status_text.text += f"\nQuality: {v_ratio:.2f}x"

    def run(self):
        self.canvas.show()
        app.run()


if __name__ == '__main__':
    TorusViewer().run()