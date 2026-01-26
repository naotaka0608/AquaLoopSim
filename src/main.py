import taichi as ti
import numpy as np
from src.config import *
from src.solver import FluidSolver

# Initialize Taichi
try:
    ti.init(arch=ti.gpu)
except:
    ti.init(arch=ti.cpu)

# Global State for GUI
tank_width = 1000.0
tank_height = 500.0
tank_depth = 1000.0

inlet_y_mm = 100.0
inlet_z_mm = 500.0
inlet_radius_mm = 60.0

outlet_y_mm = 400.0
outlet_radius_mm = 60.0
outlet_z_mm = 500.0

# Resolution Scale
SCALE = 10.0 

# Scene Data
box_vertices = ti.Vector.field(3, dtype=float, shape=8)
box_indices = ti.field(dtype=int, shape=24)

# Pipe Data
MAX_PIPE_VERTS = 1000
pipe_v_field = ti.Vector.field(3, dtype=float, shape=MAX_PIPE_VERTS)
pipe_i_field = ti.field(dtype=int, shape=MAX_PIPE_VERTS * 2)
num_pipe_indices = ti.field(dtype=int, shape=())

def update_box_geometry(res_x, res_y, res_z):
    corners = np.array([
        [0, 0, 0], [res_x, 0, 0], [res_x, 0, res_z], [0, 0, res_z],
        [0, res_y, 0], [res_x, res_y, 0], [res_x, res_y, res_z], [0, res_y, res_z]
    ], dtype=np.float32)
    
    indices = np.array([
        [0, 1], [1, 2], [2, 3], [3, 0],
        [4, 5], [5, 6], [6, 7], [7, 4],
        [0, 4], [1, 5], [2, 6], [3, 7]
    ], dtype=np.int32)
    
    for i in range(8):
        box_vertices[i] = ti.Vector(corners[i])
    for i in range(12):
        box_indices[2*i] = indices[i, 0]
        box_indices[2*i+1] = indices[i, 1]

def update_pipe_geometry(in_y, out_y, in_rad, out_rad, in_z, out_z, res_y, res_z):
    verts = []
    inds = []
    num_pipe_segs = 16
    
    def add_pipe_mesh(start_pos, end_pos, radius):
        start_idx = len(verts)
        for i in range(num_pipe_segs):
            theta = (i / num_pipe_segs) * 2 * np.pi
            y = np.cos(theta) * radius
            z = np.sin(theta) * radius
            verts.append([start_pos[0], start_pos[1] + y, start_pos[2] + z])
            verts.append([end_pos[0], end_pos[1] + y, end_pos[2] + z])
            
            curr = start_idx + 2*i
            next_seg = start_idx + 2*((i+1)%num_pipe_segs)
            
            inds.append([curr, next_seg])
            inds.append([curr+1, next_seg+1])
            inds.append([curr, curr+1])

    add_pipe_mesh([0, in_y, in_z], [-20, in_y, in_z], in_rad)
    add_pipe_mesh([0, out_y, out_z], [-20, out_y, out_z], out_rad)
    
    # Update Fields
    count = min(len(verts), MAX_PIPE_VERTS)
    for i in range(count):
        pipe_v_field[i] = ti.Vector(verts[i])
        
    ind_count = min(len(inds), MAX_PIPE_VERTS)
    for i in range(ind_count):
        pipe_i_field[2*i] = inds[i][0]
        pipe_i_field[2*i+1] = inds[i][1]
        
    num_pipe_indices[None] = ind_count * 2

def main():
    global tank_width, tank_height, tank_depth
    global inlet_y_mm, inlet_radius_mm, inlet_z_mm
    global outlet_y_mm, outlet_radius_mm, outlet_z_mm
    
    # Physics Init
    res_x = int(tank_width / SCALE)
    res_y = int(tank_height / SCALE)
    res_z = int(tank_depth / SCALE)
    
    # Defaults
    inlet_flow = 500.0 # L/min
    outlet_flow = 500.0
    
    solver = FluidSolver(res_x, res_y, res_z)
    update_box_geometry(res_x, res_y, res_z)
    update_pipe_geometry(
        inlet_y_mm/SCALE, outlet_y_mm/SCALE, 
        inlet_radius_mm/SCALE, outlet_radius_mm/SCALE, 
        inlet_z_mm/SCALE, outlet_z_mm/SCALE, 
        res_y, res_z
    )
    
    solver.update_params(
        inlet_y_mm/SCALE, outlet_y_mm/SCALE, 
        inlet_radius_mm/SCALE, outlet_radius_mm/SCALE,
        inlet_z_mm/SCALE, outlet_z_mm/SCALE,
        inlet_flow, outlet_flow
    )

    # GUI Setup
    window = ti.ui.Window("Fluid Simulation", (1400, 900))
    canvas = window.get_canvas()
    scene = ti.ui.Scene()
    camera = ti.ui.Camera()
    
    # Camera State
    camera_pos = np.array([res_x * 1.5, res_y * 1.5, res_z * 2.0], dtype=np.float32)
    camera_target = np.array([res_x / 2.0, res_y / 2.0, res_z / 2.0], dtype=np.float32)
    camera_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    last_mouse_pos = None
    
    # GUI State
    is_sync = True
    
    while window.running:
        solver.step()
        
        # UI Layout: Sidebar on Left (Fixed width ~350px)
        sidebar_width_px = 350
        sidebar_width_ratio = sidebar_width_px / 1400.0
        
        # Camera Control (Restricted to Right side)
        curr_mouse_pos = np.array(window.get_cursor_pos())
        in_sidebar = curr_mouse_pos[0] < sidebar_width_ratio
        
        if last_mouse_pos is not None and not in_sidebar:
            delta = curr_mouse_pos - last_mouse_pos
            is_shift = window.is_pressed(ti.ui.SHIFT)
            
            if window.is_pressed(ti.ui.RMB):
                if is_shift: # Zoom
                    zoom_speed = 5.0
                    diff = camera_pos - camera_target
                    dist = np.linalg.norm(diff)
                    dir_vec = diff / dist
                    new_dist = max(1.0, dist + delta[1] * zoom_speed)
                    camera_pos = camera_target + dir_vec * new_dist
                else: # Orbit
                    diff = camera_pos - camera_target
                    radius = np.linalg.norm(diff)
                    theta = np.arctan2(diff[2], diff[0])
                    phi = np.arccos(diff[1] / radius)
                    # Invert Y logic for intuitiveness if needed, but keeping consistent
                    theta += delta[0] * 5.0 
                    phi -= delta[1] * 5.0
                    phi = np.clip(phi, 0.01, 3.14)
                    camera_pos[0] = camera_target[0] + radius * np.sin(phi) * np.cos(theta)
                    camera_pos[1] = camera_target[1] + radius * np.cos(phi)
                    camera_pos[2] = camera_target[2] + radius * np.sin(phi) * np.sin(theta)
            
            elif window.is_pressed(ti.ui.MMB): # Pan
                forward = (camera_target - camera_pos) / np.linalg.norm(camera_target - camera_pos)
                right = np.cross(forward, camera_up) / np.linalg.norm(np.cross(forward, camera_up))
                up = np.cross(right, forward)
                displacement = (right * -delta[0] + up * -delta[1]) * 100.0
                camera_pos += displacement
                camera_target += displacement
                
        last_mouse_pos = curr_mouse_pos
        
        # Rendering
        camera.position(camera_pos[0], camera_pos[1], camera_pos[2])
        camera.lookat(camera_target[0], camera_target[1], camera_target[2])
        camera.up(0, 1, 0)
        scene.set_camera(camera)
        
        scene.point_light(pos=(res_x/2, res_y*1.5, res_z/2), color=(1, 1, 1))
        scene.ambient_light((0.5, 0.5, 0.5))
        
        scene.particles(solver.particle_pos, radius=0.5, per_vertex_color=solver.particle_color)
        scene.lines(box_vertices, indices=box_indices, color=(1, 1, 1), width=2.0)
        scene.lines(pipe_v_field, indices=pipe_i_field, color=(0.8, 0.8, 0.8), width=2.0, vertex_count=num_pipe_indices[None])
        
        canvas.scene(scene)
        
        # Sidebar UI
        # By setting x,y,w,h we define the area. 
        # GGUI doesn't support "docking" natively or styling backgrounds easily.
        window.GUI.begin("Control Panel", 0.0, 0.0, sidebar_width_ratio, 1.0)
        
        window.GUI.text("=== Tank Dimensions ===")
        tank_width = window.GUI.slider_float("Width (mm)", tank_width, 500.0, 2000.0)
        tank_height = window.GUI.slider_float("Height (mm)", tank_height, 200.0, 1000.0)
        tank_depth = window.GUI.slider_float("Depth (mm)", tank_depth, 500.0, 2000.0)
        
        if window.GUI.button("Apply New Dimensions"):
            res_x, res_y, res_z = int(tank_width/SCALE), int(tank_height/SCALE), int(tank_depth/SCALE)
            solver = FluidSolver(res_x, res_y, res_z)
            update_box_geometry(res_x, res_y, res_z)
            
            # Reset Camera
            camera_target = np.array([res_x/2, res_y/2, res_z/2], dtype=np.float32)
            camera_pos = camera_target + np.array([res_x, res_y/2, res_z], dtype=np.float32)
            
            # Clamp Z
            inlet_z_mm = min(inlet_z_mm, tank_depth)
            outlet_z_mm = min(outlet_z_mm, tank_depth)
            
            update_pipe_geometry(inlet_y_mm/SCALE, outlet_y_mm/SCALE, inlet_radius_mm/SCALE, outlet_radius_mm/SCALE, inlet_z_mm/SCALE, outlet_z_mm/SCALE, res_y, res_z)
            solver.update_params(inlet_y_mm/SCALE, outlet_y_mm/SCALE, inlet_radius_mm/SCALE, outlet_radius_mm/SCALE, inlet_z_mm/SCALE, outlet_z_mm/SCALE, inlet_flow, outlet_flow)

        window.GUI.text("")
        window.GUI.text("=== Inlet (Bottom) ===")
        new_in_y = window.GUI.slider_float("Inlet Y", inlet_y_mm, 20.0, tank_height - 50.0)
        new_in_z = window.GUI.slider_float("Inlet Z", inlet_z_mm, 20.0, tank_depth - 20.0)
        new_in_r = window.GUI.slider_float("Inlet Radius", inlet_radius_mm, 20.0, 150.0)
        inlet_flow = window.GUI.slider_float("Flow (L/min)", inlet_flow, 0.0, 2000.0)
        
        window.GUI.text("=== Outlet (Top) ===")
        is_sync = window.GUI.checkbox("Sync with Inlet", is_sync)
        
        new_out_y = window.GUI.slider_float("Outlet Y", outlet_y_mm, 20.0, tank_height - 50.0)
        new_out_z = window.GUI.slider_float("Outlet Z", outlet_z_mm, 20.0, tank_depth - 20.0)
        new_out_r = window.GUI.slider_float("Outlet Radius", outlet_radius_mm, 20.0, 150.0)
        
        if is_sync:
            outlet_flow = inlet_flow
            window.GUI.text(f"Flow: {outlet_flow:.1f} L/min (Synced)")
        else:
            outlet_flow = window.GUI.slider_float("Outlet Flow", outlet_flow, 0.0, 2000.0)
        
        # Check changes
        if (new_in_y != inlet_y_mm or new_in_z != inlet_z_mm or new_in_r != inlet_radius_mm or
            new_out_y != outlet_y_mm or new_out_z != outlet_z_mm or new_out_r != outlet_radius_mm):
            
            inlet_y_mm, inlet_z_mm, inlet_radius_mm = new_in_y, new_in_z, new_in_r
            outlet_y_mm, outlet_z_mm, outlet_radius_mm = new_out_y, new_out_z, new_out_r
            
            update_pipe_geometry(inlet_y_mm/SCALE, outlet_y_mm/SCALE, inlet_radius_mm/SCALE, outlet_radius_mm/SCALE, inlet_z_mm/SCALE, outlet_z_mm/SCALE, res_y, res_z)
            
        # Update params every frame
        solver.update_params(
             inlet_y_mm/SCALE, outlet_y_mm/SCALE, 
             inlet_radius_mm/SCALE, outlet_radius_mm/SCALE, 
             inlet_z_mm/SCALE, outlet_z_mm/SCALE, 
             inlet_flow, outlet_flow
        )
            
        window.GUI.text("")
        window.GUI.text(f"Particles: {solver.num_particles}") # Debug: Show count
        if window.GUI.button("Reset Particles"):
            solver.init_particles()
            
        window.GUI.end()
        window.show()

if __name__ == "__main__":
    main()
