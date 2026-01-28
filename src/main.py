
import numpy as np
import dearpygui.dearpygui as dpg
import os
import datetime
import pyvista as pv
from src.config import *
from src.solver import FluidSolver
from src.state import AppState
from src.ui import create_ui

# Resolution Scale
SCALE = 10.0 

def calculate_flow_stats(solver, res_x, res_y, res_z, 
                         inlet_y, inlet_z, inlet_r,
                         outlet_y, outlet_z, outlet_r):
    """流量統計を計算"""
    positions = solver.particle_pos
    velocities = solver.particle_vel
    
    # X座標チェック (壁際)
    x_mask = positions[:, 0] < 5.0
    
    # YZ距離の二乗計算
    # Inlet
    dy_in = positions[:, 1] - inlet_y
    dz_in = positions[:, 2] - inlet_z
    dist_sq_in = dy_in*dy_in + dz_in*dz_in
    inlet_mask = np.logical_and(x_mask, dist_sq_in < (inlet_r * inlet_r))
    
    # Outlet
    dy_out = positions[:, 1] - outlet_y
    dz_out = positions[:, 2] - outlet_z
    dist_sq_out = dy_out*dy_out + dz_out*dz_out
    outlet_mask = np.logical_and(x_mask, dist_sq_out < (outlet_r * outlet_r))
    
    inlet_count = np.sum(inlet_mask)
    outlet_count = np.sum(outlet_mask)
    
    # 平均速度
    speeds = np.linalg.norm(velocities, axis=1)
    avg_speed = np.mean(speeds) * SCALE  # mm/s に変換
    
    return inlet_count, outlet_count, avg_speed

def main():
    # 初期化 (State)
    state = AppState()
    # 設定読み込み
    state.load_from_file()

    # PyVista Plotter Setup
    # off_screen=True にして DPG コンテナに埋め込むのが理想だが、
    # 今回は別ウィンドウ(Qt/Win32)として表示させる既存方式を踏襲しつつ、
    # 閉じたときの制御を行う
    plotter = pv.Plotter()
    plotter.background_color = state.background_color_mode
    
    # Callbacks
    def take_screenshot():
        """スクリーンショットを保存"""
        os.makedirs(state.screenshot_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{state.screenshot_dir}/screenshot_{timestamp}.png"
        try:
            plotter.screenshot(filename)
            dpg.set_value("save_path_text", f"Saved: {filename}")
            print(f"Screenshot saved: {filename}")
        except Exception as e:
            print(f"Screenshot error: {e}")
            dpg.set_value("save_path_text", f"Error: {e}")

    def toggle_recording():
        """録画開始/停止"""
        state.is_recording = not state.is_recording
        if state.is_recording:
            dpg.set_item_label("record_button", "Stop Record")
            state.frame_count = 0
            os.makedirs(state.screenshot_dir, exist_ok=True)
            dpg.set_value("save_path_text", "Recording...")
        else:
            dpg.set_item_label("record_button", "Start Record")
            dpg.set_value("save_path_text", f"Done: {state.frame_count} frames")

    callbacks = {
        'take_screenshot': take_screenshot,
        'toggle_recording': toggle_recording
    }

    # UI Setup
    create_ui(state, callbacks)

    # Simulation Setup
    res_x = int(state.tank_width / SCALE)
    res_y = int(state.tank_height / SCALE)
    res_z = int(state.tank_depth / SCALE)
    
    solver = FluidSolver(res_x, res_y, res_z, state.current_num_particles)
    
    # Geometry Helpers inside main to access plotter/meshes
    box_mesh = None
    inlet_mesh = None
    outlet_mesh = None
    
    def update_box_geometry(rx, ry, rz):
        nonlocal box_mesh
        corners = np.array([
            [0, 0, 0], [rx, 0, 0], [rx, 0, rz], [0, 0, rz],
            [0, ry, 0], [rx, ry, 0], [rx, ry, rz], [0, ry, rz]
        ], dtype=np.float32)
        indices = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ], dtype=np.int32)
        lines_flat = np.hstack([[2, i0, i1] for i0, i1 in indices])
        
        # 新規作成または更新
        if box_mesh is None:
            box_mesh = pv.PolyData(corners)
            box_mesh.lines = lines_flat
        else:
            box_mesh.points = corners
            # linesはトポロジー変わらないのでそのまま
            
    def create_pipe_polydata(y, z, radius, start_x=0.0, length=20.0):
        """指定されたパラメータでパイプ形状のPolyDataを生成して返す"""
        verts = []
        lines_list = []
        num_pipe_segs = 16
        start_pos = [start_x, y, z]
        end_pos = [start_x + length, y, z] # length正なら+X方向, 負なら-X方向
        
        for i in range(num_pipe_segs):
            theta = (i / num_pipe_segs) * 2 * np.pi
            dy = np.cos(theta) * radius
            dz = np.sin(theta) * radius
            p1 = [start_pos[0], start_pos[1] + dy, start_pos[2] + dz]
            p2 = [end_pos[0], end_pos[1] + dy, end_pos[2] + dz]
            verts.append(p1)
            verts.append(p2)
            
            curr = 2*i
            next_seg = 2*((i+1)%num_pipe_segs)
            lines_list.append([2, curr, next_seg])
            lines_list.append([2, curr+1, next_seg+1])
            lines_list.append([2, curr, curr+1])
            
        points = np.array(verts, dtype=np.float32)
        lines_flat = np.hstack(lines_list)
        mesh = pv.PolyData(points)
        mesh.lines = lines_flat
        return mesh

    def update_inlet_geometry(y, z, radius):
        nonlocal inlet_mesh
        # Inlet: X=0 から -20 (外側) へ
        new_mesh = create_pipe_polydata(y, z, radius, start_x=0.0, length=-20.0)
        if inlet_mesh is None:
            inlet_mesh = new_mesh
        else:
            inlet_mesh.points = new_mesh.points
            inlet_mesh.lines = new_mesh.lines

    def update_outlet_geometry(y, z, radius):
        nonlocal outlet_mesh
        # Outlet: X=res_x (Grid coordinates) から +20.0 (外側) へ
        # Note: res_x is in grid units.
        # Pipe length 20.0 to match inlet visual style (symmetric).
        new_mesh = create_pipe_polydata(y, z, radius, start_x=res_x, length=20.0)
        if outlet_mesh is None:
            outlet_mesh = new_mesh
        else:
            outlet_mesh.points = new_mesh.points
            outlet_mesh.lines = new_mesh.lines

    # Initial Geometry Update
    update_box_geometry(res_x, res_y, res_z)
    update_inlet_geometry(state.inlet_y_mm/SCALE, state.inlet_z_mm/SCALE, state.inlet_radius_mm/SCALE)
    update_outlet_geometry(state.outlet_y_mm/SCALE, state.outlet_z_mm/SCALE, state.outlet_radius_mm/SCALE)

    # Add initial meshes to plotter
    # Particles
    particles_mesh = pv.PolyData(solver.particle_pos)
    particles_mesh.point_data["rgb"] = solver.particle_color
    
    # ここ重要: style='points', render_points_as_spheres=False
    particles_actor = plotter.add_mesh(
        particles_mesh, 
        scalars="rgb", 
        rgb=True, 
        point_size=state.particle_size, 
        style='points', 
        render_points_as_spheres=False, 
        name="particles"
    )

    # Walls
    if box_mesh:
        plotter.add_mesh(box_mesh, color="white", style="wireframe", line_width=2, name="walls")
    
    # Inlet/Outlet markers
    if inlet_mesh:
        plotter.add_mesh(inlet_mesh, color="cyan", style="wireframe", line_width=2, name="pipe_inlet")
    if outlet_mesh:
        plotter.add_mesh(outlet_mesh, color="red", style="wireframe", line_width=2, name="pipe_outlet") # Color difference for clarity

    # Obstacles placeholder (Vis name)
    vis_obstacles_name = "obstacles_vis"
    
    # Camera setup
    plotter.camera_position = [(-res_x*1.5, res_y/2, res_z/2), (res_x/2, res_y/2, res_z/2), (0, 1, 0)]
    
    # Try to set window position (Backend dependent, might not work on all systems)
    try:
        if hasattr(plotter, 'ren_win'):
            # VTK method
            plotter.ren_win.SetPosition(550, 50)
            plotter.ren_win.SetSize(800, 600)
    except:
        pass

    plotter.show(auto_close=False, interactive_update=True)

    # Main Loop
    flow_update_counter = 0
    try:
        while dpg.is_dearpygui_running():
            # PyVista Render Window check
            if hasattr(plotter, 'ren_win') and plotter.ren_win is None:
                break
            # DPG Render
            dpg.render_dearpygui_frame()
            
            # Debug Log (Removed)
            if not hasattr(main, "last_debug_time"): main.last_debug_time = 0
            
            if state.needs_particle_reset:
                 solver.init_particles()
                 state.needs_particle_reset = False
                 state.sim_elapsed_time = 0.0
                 state.frame_count = 0
                 
                 # Mesh Refresh (Optimized)
                 particles_mesh.points[:] = solver.particle_pos
                 particles_mesh.point_data["rgb"][:] = solver.particle_color
                 try:
                     # Reset trails if needed
                     solver.trail_positions[:] = 0
                     # Trail mesh access/update logic would go here if specialized
                 except:
                     pass

            # State Update Checks
            if state.needs_dimension_update:
                res_x = int(state.tank_width / SCALE)
                res_y = int(state.tank_height / SCALE)
                res_z = int(state.tank_depth / SCALE)
                solver.update_dimensions(res_x, res_y, res_z)
                
                update_box_geometry(res_x, res_y, res_z)
                update_inlet_geometry(state.inlet_y_mm/SCALE, state.inlet_z_mm/SCALE, state.inlet_radius_mm/SCALE)
                update_outlet_geometry(state.outlet_y_mm/SCALE, state.outlet_z_mm/SCALE, state.outlet_radius_mm/SCALE)
                
                # Plotter update
                if box_mesh:
                     # add_mesh は同名なら上書き更新される
                    plotter.add_mesh(box_mesh, color="white", style="wireframe", line_width=2, name="walls")
                # Remove/Add pipes explicitly to ensure update
                if inlet_mesh:
                    plotter.remove_actor("pipe_inlet")
                    plotter.add_mesh(inlet_mesh, color="cyan", style="wireframe", line_width=2, name="pipe_inlet")
                if outlet_mesh:
                    plotter.remove_actor("pipe_outlet")
                    plotter.add_mesh(outlet_mesh, color="red", style="wireframe", line_width=2, name="pipe_outlet")
                
                # Reset camera roughly
                # plotter.camera_position = ... (お好みで)
                
                state.needs_dimension_update = False
                state.needs_particle_reset = True
                
            if state.needs_particle_count_update:
                solver.update_particle_count(state.target_num_particles)
                state.current_num_particles = state.target_num_particles
                state.needs_particle_count_update = False
                state.needs_particle_reset = True
                dpg.set_value("particle_count_text", f"現在: {state.current_num_particles:,}")
                
                solver.init_particles()
                state.needs_particle_reset = False
                state.sim_elapsed_time = 0.0
                state.frame_count = 0
                
                # Mesh Refresh
                particles_mesh = pv.PolyData(solver.particle_pos)
                particles_mesh.point_data["rgb"] = solver.particle_color
                particles_actor = plotter.add_mesh(
                    particles_mesh, 
                    scalars="rgb", 
                    rgb=True, 
                    point_size=state.particle_size, 
                    style='points', 
                    render_points_as_spheres=False, 
                    name="particles"
                )

            # Solver Parameters Update
            # Sync Logic
            if state.is_sync:
                state.outlet_y_mm = state.inlet_y_mm
                state.outlet_z_mm = state.inlet_z_mm
                state.outlet_radius_mm = state.inlet_radius_mm
                state.outlet_flow = state.inlet_flow
            
            # Use update_params to recalculate velocities
            solver.update_params(
                 state.inlet_y_mm / SCALE,
                 state.outlet_y_mm / SCALE,
                 state.inlet_radius_mm / SCALE,
                 state.outlet_radius_mm / SCALE,
                 state.inlet_z_mm / SCALE,
                 state.outlet_z_mm / SCALE,
                 state.inlet_flow,
                 state.outlet_flow
            )
            
            # Pipe Geometry Update Check
            # Inlet
            cur_inlet = (state.inlet_y_mm, state.inlet_z_mm, state.inlet_radius_mm)
            if not hasattr(main, "last_inlet") or main.last_inlet != cur_inlet:
                update_inlet_geometry(state.inlet_y_mm/SCALE, state.inlet_z_mm/SCALE, state.inlet_radius_mm/SCALE)
                if inlet_mesh:
                     # In-place update approach (safest against flickering, assuming points matching)
                     # Since create_pipe_polydata returns same topology (16 segs), in-place is safe.
                     # But user complained about it disappearing.
                     # Let's try remove/add again but ONLY when changed.
                     # "Disappearing" might have been due to previous logic bugs.
                     # With separate actors, remove/add is safer.
                     plotter.remove_actor("pipe_inlet")
                     plotter.add_mesh(inlet_mesh, color="cyan", style="wireframe", line_width=2, name="pipe_inlet")
                main.last_inlet = cur_inlet
            
            # Outlet
            cur_outlet = (state.outlet_y_mm, state.outlet_z_mm, state.outlet_radius_mm)
            if not hasattr(main, "last_outlet") or main.last_outlet != cur_outlet:
                update_outlet_geometry(state.outlet_y_mm/SCALE, state.outlet_z_mm/SCALE, state.outlet_radius_mm/SCALE)
                if outlet_mesh:
                     plotter.remove_actor("pipe_outlet")
                     plotter.add_mesh(outlet_mesh, color="red", style="wireframe", line_width=2, name="pipe_outlet")
                main.last_outlet = cur_outlet
            
            # Particle Size update (Visual)
            if particles_actor:
                prop = particles_actor.GetProperty()
                prop.SetPointSize(state.particle_size)
            
            # Background Color
            if state.background_color_mode == "Black":
                plotter.background_color = "black"
            elif state.background_color_mode == "White":
                plotter.background_color = "white"
            elif state.background_color_mode == "Dark Gray":
                plotter.background_color = [0.2, 0.2, 0.2]
            elif state.background_color_mode == "Light Gray":
                plotter.background_color = [0.7, 0.7, 0.7]
            elif state.background_color_mode == "Paraview Blue":
                plotter.background_color = [0.32, 0.34, 0.43]

            # Step Simulation
            if not state.is_paused:
                steps_per_frame = max(1, int(state.sim_speed))
                for _ in range(steps_per_frame):
                    solver.step()
                state.sim_elapsed_time += (1.0 / 60.0) * state.sim_speed
            
            # Update Particles Mesh (In-place)
            particles_mesh.points[:] = solver.particle_pos
            particles_mesh.point_data["rgb"][:] = solver.particle_color
            
            # Cross Section
            if particles_actor:
                if state.show_cross_section:
                    center = [res_x/2 * SCALE, res_y/2 * SCALE, res_z/2 * SCALE] # unused?
                    
                    offset_pct = (state.cross_section_pos - 50.0) / 100.0
                    origin = [res_x/2, res_y/2, res_z/2]
                    normal = [1, 0, 0]
                    
                    if state.cross_section_axis == 'X':
                        origin[0] += res_x * offset_pct
                        normal = [1, 0, 0]
                    elif state.cross_section_axis == 'Y':
                        origin[1] += res_y * offset_pct
                        normal = [0, 1, 0]
                    elif state.cross_section_axis == 'Z':
                        origin[2] += res_z * offset_pct
                        normal = [0, 0, 1]
                        
                    plane = pv.Plane(center=origin, direction=normal)
                    particles_actor.mapper.RemoveAllClippingPlanes()
                    particles_actor.mapper.AddClippingPlane(plane)
                else:
                    particles_actor.mapper.RemoveAllClippingPlanes()

            # Render
            plotter.render()
            
            # Elapsed Time UI
            try:
                minutes = int(state.sim_elapsed_time // 60)
                seconds = state.sim_elapsed_time % 60
                dpg.set_value("elapsed_time_text", f"経過時間: {minutes}分 {seconds:.1f}秒")
            except:
                pass
            
            # Flow Meter UI
            flow_update_counter += 1
            if state.show_flow_meter and flow_update_counter >= 10:
                flow_update_counter = 0
                inlet_c, outlet_c, avg_s = calculate_flow_stats(
                    solver, res_x, res_y, res_z,
                    state.inlet_y_mm/SCALE, state.inlet_z_mm/SCALE, state.inlet_radius_mm/SCALE,
                    state.outlet_y_mm/SCALE, state.outlet_z_mm/SCALE, state.outlet_radius_mm/SCALE
                )
                try:
                    dpg.set_value("inlet_flow_text", f"流入口: {inlet_c} 粒子")
                    dpg.set_value("outlet_flow_text", f"流出口: {outlet_c} 粒子")
                    dpg.set_value("avg_speed_text", f"平均速度: {avg_s:.1f} mm/s")
                except:
                    pass
            
            # Recording
            if state.is_recording:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{state.screenshot_dir}/frame_{timestamp}_{state.frame_count:05d}.png"
                try:
                    plotter.screenshot(filename)
                    state.frame_count += 1
                    dpg.set_value("frame_count_text", f"Frames: {state.frame_count}")
                except:
                    pass
            
            # Obstacles
            if state.show_obstacles and len(state.obstacles) > 0:
                mb = pv.MultiBlock()
                for obs in state.obstacles:
                    ox, oy, oz = obs['x']/SCALE, obs['y']/SCALE, obs['z']/SCALE
                    osize = obs['size']/SCALE
                    if obs['type'] == 'sphere':
                        mesh = pv.Sphere(radius=osize, center=(ox, oy, oz))
                    else:
                        mesh = pv.Box(bounds=(ox-osize, ox+osize, oy-osize, oy+osize, oz-osize, oz+osize))
                    mb.append(mesh)
                plotter.add_mesh(mb, color="orange", opacity=0.5, name=vis_obstacles_name, reset_camera=False)
            elif not state.show_obstacles:
                plotter.remove_actor(vis_obstacles_name)

    except Exception as e:
        print(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            plotter.close()
        except:
            pass
        dpg.destroy_context()

if __name__ == "__main__":
    main()
