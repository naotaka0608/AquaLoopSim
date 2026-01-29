"""
Fluid Solver using Numba for JIT compilation (PyInstaller compatible)
"""
import numpy as np
from numba import njit, prange
from src.config import *


@njit(cache=True)
def apply_colormap(t, mode):
    """カラーマップを適用 (t: 0.0~1.0)"""
    r, g, b = 0.0, 0.0, 0.0
    
    if mode == 0:  # Blue-Red (デフォルト)
        r, g, b = t, t * 0.3, 1.0 - t
    elif mode == 1:  # Rainbow
        h = t * 0.8 * 6.0
        i = int(np.floor(h))
        f = h - i
        if i == 0:
            r, g, b = 1.0, f, 0.0
        elif i == 1:
            r, g, b = 1.0 - f, 1.0, 0.0
        elif i == 2:
            r, g, b = 0.0, 1.0, f
        elif i == 3:
            r, g, b = 0.0, 1.0 - f, 1.0
        elif i == 4:
            r, g, b = f, 0.0, 1.0
        else:
            r, g, b = 1.0, 0.0, 1.0 - f
    elif mode == 2:  # Cool-Warm
        if t < 0.5:
            s = t * 2.0
            r, g, b = s, s, 1.0
        else:
            s = (t - 0.5) * 2.0
            r, g, b = 1.0, 1.0 - s, 1.0 - s
    elif mode == 3:  # Viridis風
        r = 0.267 + t * 0.6
        g = 0.004 + t * 0.87
        b = 0.329 + t * 0.3 - t * t * 0.5
    
    return r, g, b


@njit(cache=True)
def trilinear_interp(velocity, base_x, base_y, base_z, frac_x, frac_y, frac_z, res):
    """三線形補間"""
    # 境界チェック
    x0, y0, z0 = base_x, base_y, base_z
    x1 = min(base_x + 1, res[0] - 1)
    y1 = min(base_y + 1, res[1] - 1)
    z1 = min(base_z + 1, res[2] - 1)
    
    result = np.zeros(3)
    for c in range(3):
        c000 = velocity[x0, y0, z0, c]
        c100 = velocity[x1, y0, z0, c]
        c010 = velocity[x0, y1, z0, c]
        c110 = velocity[x1, y1, z0, c]
        c001 = velocity[x0, y0, z1, c]
        c101 = velocity[x1, y0, z1, c]
        c011 = velocity[x0, y1, z1, c]
        c111 = velocity[x1, y1, z1, c]
        
        c00 = c000 * (1 - frac_x) + c100 * frac_x
        c10 = c010 * (1 - frac_x) + c110 * frac_x
        c01 = c001 * (1 - frac_x) + c101 * frac_x
        c11 = c011 * (1 - frac_x) + c111 * frac_x
        
        c0 = c00 * (1 - frac_y) + c10 * frac_y
        c1 = c01 * (1 - frac_y) + c11 * frac_y
        
        result[c] = c0 * (1 - frac_z) + c1 * frac_z
    
    return result


@njit(cache=True)
def sample_velocity_single(velocity, pos, res, inlet_face, inlet_p1, inlet_p2, inlet_radius, inlet_velocity,
                           outlet_face, outlet_p1, outlet_p2, outlet_radius, outlet_velocity):
    """単一位置での速度サンプリング (Face対応版)"""
    vel = np.zeros(3)
    
    # 内部領域
    if pos[0] >= 0 and pos[0] < res[0] - 1 and \
       pos[1] >= 0 and pos[1] < res[1] - 1 and \
       pos[2] >= 0 and pos[2] < res[2] - 1:
        
        base_x = int(np.floor(pos[0]))
        base_y = int(np.floor(pos[1]))
        base_z = int(np.floor(pos[2]))
        frac_x = pos[0] - base_x
        frac_y = pos[1] - base_y
        frac_z = pos[2] - base_z
        
        vel = trilinear_interp(velocity, base_x, base_y, base_z, frac_x, frac_y, frac_z, res)
        
    else:
        # 外部領域（パイプ内）
        # Inlet Pipe Check
        in_pipe = False
        if inlet_face == 0 and pos[0] < 0: # X- (Left)
            if (pos[1]-inlet_p1)**2 + (pos[2]-inlet_p2)**2 < (inlet_radius*1.5)**2:
                vel[0] = -inlet_velocity # Suction (Out)
                in_pipe = True
        elif inlet_face == 1 and pos[0] >= res[0]-1: # X+ (Right)
            if (pos[1]-inlet_p1)**2 + (pos[2]-inlet_p2)**2 < (inlet_radius*1.5)**2:
                vel[0] = inlet_velocity # Suction (Out)
                in_pipe = True
        elif inlet_face == 2 and pos[1] < 0: # Y- (Bottom)
            if (pos[0]-inlet_p1)**2 + (pos[2]-inlet_p2)**2 < (inlet_radius*1.5)**2:
                vel[1] = -inlet_velocity # Suction (Out)
                in_pipe = True
        elif inlet_face == 3 and pos[1] >= res[1]-1: # Y+ (Top)
            if (pos[0]-inlet_p1)**2 + (pos[2]-inlet_p2)**2 < (inlet_radius*1.5)**2:
                vel[1] = inlet_velocity # Suction (Out)
                in_pipe = True

        if not in_pipe:
            # Outlet Pipe Check (Discharge)
            if outlet_face == 0 and pos[0] < 0:
                if (pos[1]-outlet_p1)**2 + (pos[2]-outlet_p2)**2 < (outlet_radius*1.5)**2:
                    vel[0] = outlet_velocity # Discharge (In)
            elif outlet_face == 1 and pos[0] >= res[0]-1:
                if (pos[1]-outlet_p1)**2 + (pos[2]-outlet_p2)**2 < (outlet_radius*1.5)**2:
                    vel[0] = -outlet_velocity # Discharge (In)
            elif outlet_face == 2 and pos[1] < 0:
                if (pos[0]-outlet_p1)**2 + (pos[2]-outlet_p2)**2 < (outlet_radius*1.5)**2:
                    vel[1] = outlet_velocity # Discharge (In)
            elif outlet_face == 3 and pos[1] >= res[1]-1:
                if (pos[0]-outlet_p1)**2 + (pos[2]-outlet_p2)**2 < (outlet_radius*1.5)**2:
                    vel[1] = -outlet_velocity # Discharge (In)

    return vel

@njit(cache=True, parallel=True)
def init_particles_kernel(particle_pos, particle_vel, particle_color, particle_life, 
                          particle_absorbed, trail_positions, trail_index, res, num_particles, trail_length):
    """粒子初期化カーネル"""
    for i in prange(num_particles):
        particle_pos[i, 0] = np.random.random() * res[0]
        particle_pos[i, 1] = np.random.random() * res[1]
        particle_pos[i, 2] = np.random.random() * res[2]
        particle_life[i] = np.random.random()
        particle_absorbed[i] = 0
        trail_index[i] = 0
        for j in range(trail_length):
            trail_positions[i, j, 0] = particle_pos[i, 0]
            trail_positions[i, j, 1] = particle_pos[i, 1]
            trail_positions[i, j, 2] = particle_pos[i, 2]
        particle_vel[i, 0] = 0.0
        particle_vel[i, 1] = 0.0
        particle_vel[i, 2] = 0.0
        particle_color[i, 0] = 0.0
        particle_color[i, 1] = 1.0
        particle_color[i, 2] = 1.0

@njit(cache=True, parallel=True)
def advect_kernel(velocity, new_velocity, dt, res, inlet_face, inlet_p1, inlet_p2, inlet_radius, inlet_velocity,
                  outlet_face, outlet_p1, outlet_p2, outlet_radius, outlet_velocity):
    """速度場の移流"""
    for i in prange(res[0]):
        for j in range(res[1]):
            for k in range(res[2]):
                pos = np.array([i + 0.5, j + 0.5, k + 0.5])
                p_back = pos - velocity[i, j, k] * dt
                val = sample_velocity_single(velocity, p_back, res, inlet_face, inlet_p1, inlet_p2, inlet_radius, inlet_velocity,
                                            outlet_face, outlet_p1, outlet_p2, outlet_radius, outlet_velocity)
                new_velocity[i, j, k, 0] = val[0]
                new_velocity[i, j, k, 1] = val[1]
                new_velocity[i, j, k, 2] = val[2]

@njit(cache=True, parallel=True)
def apply_inlet_boundary_kernel(velocity, res, inlet_face, inlet_p1, inlet_p2, inlet_radius, inlet_velocity,
                                outlet_face, outlet_p1, outlet_p2, outlet_radius, outlet_velocity):
    """境界条件適用 (Inlet/Outlet)"""
    # 全体をスキャンするのは非効率だが、実装は簡単。
    # 最適化：各面付近のみループする
    
    # Inlet Face logic (Make it SUCTION/OUT)
    # 0: X- (i=0..5)
    if inlet_face == 0:
        for i in prange(min(5, res[0])):
            for j in range(res[1]):
                for k in range(res[2]):
                    if (j-inlet_p1)**2 + (k-inlet_p2)**2 < inlet_radius**2:
                        velocity[i, j, k, 0] = -inlet_velocity # Force OUT (Left)
                        velocity[i, j, k, 1] = 0.0
                        velocity[i, j, k, 2] = 0.0
    # 1: X+ (i=res[0]-5..res[0])
    elif inlet_face == 1:
        start_i = max(0, res[0]-5)
        for i in prange(start_i, res[0]):
            for j in range(res[1]):
                for k in range(res[2]):
                    if (j-inlet_p1)**2 + (k-inlet_p2)**2 < inlet_radius**2:
                        velocity[i, j, k, 0] = inlet_velocity # Force OUT (Right)
                        velocity[i, j, k, 1] = 0.0
                        velocity[i, j, k, 2] = 0.0
    # 2: Y- (j=0..5)
    elif inlet_face == 2:
        for j in prange(min(5, res[1])):
            for i in range(res[0]):
                for k in range(res[2]):
                    if (i-inlet_p1)**2 + (k-inlet_p2)**2 < inlet_radius**2:
                        velocity[i, j, k, 0] = 0.0
                        velocity[i, j, k, 1] = -inlet_velocity # Force OUT (Down)
                        velocity[i, j, k, 2] = 0.0
    # 3: Y+ (j=res[1]-5..res[1])
    elif inlet_face == 3:
        start_j = max(0, res[1]-5)
        for j in prange(start_j, res[1]):
            for i in range(res[0]):
                for k in range(res[2]):
                    if (i-inlet_p1)**2 + (k-inlet_p2)**2 < inlet_radius**2:
                        velocity[i, j, k, 0] = 0.0
                        velocity[i, j, k, 1] = inlet_velocity # Force OUT (Up)
                        velocity[i, j, k, 2] = 0.0

    # Outlet Face logic (Make it DISCHARGE/IN)
    # 0: X-
    if outlet_face == 0:
        for i in prange(min(5, res[0])):
            for j in range(res[1]):
                for k in range(res[2]):
                    if (j-outlet_p1)**2 + (k-outlet_p2)**2 < outlet_radius**2:
                        velocity[i, j, k, 0] = outlet_velocity # Force IN (Right)
                        velocity[i, j, k, 1:] = 0.0
    # 1: X+
    elif outlet_face == 1:
        start_i = max(0, res[0]-5)
        for i in prange(start_i, res[0]):
            for j in range(res[1]):
                for k in range(res[2]):
                    if (j-outlet_p1)**2 + (k-outlet_p2)**2 < outlet_radius**2:
                        velocity[i, j, k, 0] = -outlet_velocity # Force IN (Left)
                        velocity[i, j, k, 1:] = 0.0
    # 2: Y-
    elif outlet_face == 2:
        for j in prange(min(5, res[1])):
            for i in range(res[0]):
                for k in range(res[2]):
                    if (i-outlet_p1)**2 + (k-outlet_p2)**2 < outlet_radius**2:
                        velocity[i, j, k, 1] = outlet_velocity # Force IN (Up)
                        velocity[i, j, k, 0] = 0.0
                        velocity[i, j, k, 2] = 0.0
    # 3: Y+
    elif outlet_face == 3:
        start_j = max(0, res[1]-5)
        for j in prange(start_j, res[1]):
            for i in range(res[0]):
                for k in range(res[2]):
                    if (i-outlet_p1)**2 + (k-outlet_p2)**2 < outlet_radius**2:
                        velocity[i, j, k, 1] = -outlet_velocity # Force IN (Down)
                        velocity[i, j, k, 0] = 0.0
                        velocity[i, j, k, 2] = 0.0

@njit(cache=True, parallel=True)
def apply_walls_kernel(velocity, res, inlet_face, inlet_p1, inlet_p2, inlet_radius, outlet_face, outlet_p1, outlet_p2, outlet_radius):
    """壁境界条件 (穴あき対応)"""
    for i in prange(res[0]):
        for j in range(res[1]):
            for k in range(res[2]):
                
                # Check walls
                is_wall = False
                
                # X- Wall (i=0)
                if i < 1:
                    is_hole = False
                    if inlet_face == 0 and (j-inlet_p1)**2 + (k-inlet_p2)**2 < inlet_radius**2: is_hole = True
                    if outlet_face == 0 and (j-outlet_p1)**2 + (k-outlet_p2)**2 < outlet_radius**2: is_hole = True
                    if not is_hole: velocity[i,j,k,0] = 0.0

                # X+ Wall (i=res[0]-1)
                if i >= res[0]-1:
                    is_hole = False
                    if inlet_face == 1 and (j-inlet_p1)**2 + (k-inlet_p2)**2 < inlet_radius**2: is_hole = True
                    if outlet_face == 1 and (j-outlet_p1)**2 + (k-outlet_p2)**2 < outlet_radius**2: is_hole = True
                    if not is_hole: velocity[i,j,k,0] = 0.0
                
                # Y- Wall (j=0)
                if j < 1:
                    is_hole = False
                    if inlet_face == 2 and (i-inlet_p1)**2 + (k-inlet_p2)**2 < inlet_radius**2: is_hole = True
                    if outlet_face == 2 and (i-outlet_p1)**2 + (k-outlet_p2)**2 < outlet_radius**2: is_hole = True
                    if not is_hole: velocity[i,j,k,1] = 0.0

                # Y+ Wall (j=res[1]-1)
                if j >= res[1]-1:
                    is_hole = False
                    if inlet_face == 3 and (i-inlet_p1)**2 + (k-inlet_p2)**2 < inlet_radius**2: is_hole = True
                    if outlet_face == 3 and (i-outlet_p1)**2 + (k-outlet_p2)**2 < outlet_radius**2: is_hole = True
                    if not is_hole: velocity[i,j,k,1] = 0.0

                # Z- Wall (k=0)
                if k < 1: velocity[i,j,k,2] = 0.0

                # Z+ Wall (k=res[2]-1)
                # Z+ Wall (k=res[2]-1)
                if k >= res[2]-1: velocity[i,j,k,2] = 0.0

@njit(cache=True, parallel=True)
def divergence_calc_kernel(velocity, divergence, res):
    """発散の計算"""
    for i in prange(1, res[0]-1):
        for j in range(1, res[1]-1):
            for k in range(1, res[2]-1):
                div = (velocity[i+1, j, k, 0] - velocity[i-1, j, k, 0] +
                       velocity[i, j+1, k, 1] - velocity[i, j-1, k, 1] +
                       velocity[i, j, k+1, 2] - velocity[i, j, k-1, 2]) * 0.5
                divergence[i, j, k] = div

@njit(cache=True, parallel=True)
def pressure_jacobi_kernel(pressure, new_pressure, divergence, res):
    """圧力のヤコビ法反復"""
    for i in prange(1, res[0]-1):
        for j in range(1, res[1]-1):
            for k in range(1, res[2]-1):
                p_new = (pressure[i-1, j, k] + pressure[i+1, j, k] +
                         pressure[i, j-1, k] + pressure[i, j+1, k] +
                         pressure[i, j, k-1] + pressure[i, j, k+1] -
                         divergence[i, j, k]) / 6.0
                new_pressure[i, j, k] = p_new

@njit(cache=True, parallel=True)
def project_kernel(velocity, pressure, res):
    """プロジェクション（圧力勾配の減算）"""
    for i in prange(1, res[0]-1):
        for j in range(1, res[1]-1):
            for k in range(1, res[2]-1):
                velocity[i, j, k, 0] -= (pressure[i+1, j, k] - pressure[i-1, j, k]) * 0.5
                velocity[i, j, k, 1] -= (pressure[i, j+1, k] - pressure[i, j-1, k]) * 0.5
                velocity[i, j, k, 2] -= (pressure[i, j, k+1] - pressure[i, j, k-1]) * 0.5

@njit(cache=True, parallel=True)
def advect_particles_kernel(particle_pos, particle_vel, particle_color, particle_life, particle_absorbed,
                            trail_positions, trail_index, velocity, res, dt,
                            inlet_face, inlet_p1, inlet_p2, inlet_radius, inlet_velocity,
                            outlet_face, outlet_p1, outlet_p2, outlet_radius, outlet_velocity,
                            num_particles, trail_length, colormap_mode,
                            obstacle_data, num_obstacles):
    """粒子移流（メインカーネル）"""
    for i in prange(num_particles):
        pos = np.array([particle_pos[i, 0], particle_pos[i, 1], particle_pos[i, 2]])
        vel = sample_velocity_single(velocity, pos, res, inlet_face, inlet_p1, inlet_p2, inlet_radius, inlet_velocity,
                                     outlet_face, outlet_p1, outlet_p2, outlet_radius, outlet_velocity)
        
        # Explicit Suction Force (VFX Hack for Inlet)
        # Pull particles towards Inlet if they are near
        suction_radius = inlet_radius * 8.0 # Large influence
        
        # Determine inlet position
        inlet_pos = np.zeros(3)
        if inlet_face == 0: inlet_pos[:] = [0.0, inlet_p1, inlet_p2]
        elif inlet_face == 1: inlet_pos[:] = [res[0], inlet_p1, inlet_p2]
        elif inlet_face == 2: inlet_pos[:] = [inlet_p1, 0.0, inlet_p2]
        elif inlet_face == 3: inlet_pos[:] = [inlet_p1, res[1], inlet_p2]
        
        # Vector to inlet
        diff = inlet_pos - pos
        dist_sq = diff[0]**2 + diff[1]**2 + diff[2]**2
        
        if dist_sq < suction_radius**2:
            dist = np.sqrt(dist_sq)
            if dist > 0.1:
                # Force direction normalized
                dir_norm = diff / dist
                # Strength falls off with distance
                strength = inlet_velocity * (1.0 - dist/suction_radius) * 1.5 
                
                # Add suction component
                vel += dir_norm * strength

        particle_vel[i, 0] = vel[0]
        particle_vel[i, 1] = vel[1]
        particle_vel[i, 2] = vel[2]
        
        p_next = pos + vel * dt
        
        # Recycle particles
        should_recycle = False
        
        # Check if out of bounds (simplified recycle check)
        # Note: Ideally check distance from Inlet for recycling
        # For simplicity, if particle is too far or "absorbed", recycle
        
        # Absorption at INLET (Suction)
        absorbed = False
        dist_sq_out = 999999.9
        
        if inlet_face == 0: # X-
            if p_next[0] < 0:
                if (p_next[1]-inlet_p1)**2 + (p_next[2]-inlet_p2)**2 < (inlet_radius*1.5)**2: absorbed=True
        elif inlet_face == 1: # X+
            if p_next[0] > res[0]:
                if (p_next[1]-inlet_p1)**2 + (p_next[2]-inlet_p2)**2 < (inlet_radius*1.5)**2: absorbed=True
        elif inlet_face == 2: # Y-
            if p_next[1] < 0:
                if (p_next[0]-inlet_p1)**2 + (p_next[2]-inlet_p2)**2 < (inlet_radius*1.5)**2: absorbed=True
        elif inlet_face == 3: # Y+
            if p_next[1] > res[1]:
                if (p_next[0]-inlet_p1)**2 + (p_next[2]-inlet_p2)**2 < (inlet_radius*1.5)**2: absorbed=True

        if absorbed:
            particle_absorbed[i] = 1
            # Wait for recycle logic below or just respawn immediately?
            # Let's respawn immediately to keep flow constant
            should_recycle = True

        if should_recycle:
            # Respawn at OUTLET (Discharge)
            # Position depends on outlet_face
            # Randomize within circle
            r = outlet_radius * 0.8 * np.sqrt(np.random.random())
            theta = np.random.random() * 2.0 * np.pi
            
            if outlet_face == 0: # X-
                p_next[0] = -1.0 # Near edge
                p_next[1] = outlet_p1 + r * np.cos(theta)
                p_next[2] = outlet_p2 + r * np.sin(theta)
            elif outlet_face == 1: # X+
                p_next[0] = res[0] + 1.0
                p_next[1] = outlet_p1 + r * np.cos(theta)
                p_next[2] = outlet_p2 + r * np.sin(theta)
            elif outlet_face == 2: # Y-
                p_next[0] = outlet_p1 + r * np.cos(theta)
                p_next[1] = -1.0
                p_next[2] = outlet_p2 + r * np.sin(theta)
            elif outlet_face == 3: # Y+
                p_next[0] = outlet_p1 + r * np.cos(theta)
                p_next[1] = res[1] + 1.0
                p_next[2] = outlet_p2 + r * np.sin(theta)
                
            particle_vel[i] = 0.0 # Reset velocity
            particle_absorbed[i] += 1 # Increment generation count
            particle_life[i] = np.random.random()

        # ... (Wall bouncing logic needs similar generalized update, skipping for brevity, assume simple bounds check works or handled by apply_walls_kernel velocity clamping for now, though particles might leak.
        # Ideally, we implement a proper general boundary check for particles too.
        
        # Simple clamp for now to prevent explosion outside boundaries
        if not should_recycle:
            margin = 1.0
            # X limits
            if p_next[0] < 0 and inlet_face != 0 and outlet_face != 0: p_next[0] = margin; particle_vel[i,0] *= -0.5
            if p_next[0] > res[0] and inlet_face != 1 and outlet_face != 1: p_next[0] = res[0]-margin; particle_vel[i,0] *= -0.5
            # Y limits
            if p_next[1] < 0 and inlet_face != 2 and outlet_face != 2: p_next[1] = margin; particle_vel[i,1] *= -0.5
            if p_next[1] > res[1] and inlet_face != 3 and outlet_face != 3: p_next[1] = res[1]-margin; particle_vel[i,1] *= -0.5
            # Z limits
            if p_next[2] < 0: p_next[2] = margin; particle_vel[i,2] *= -0.5
            if p_next[2] > res[2]: p_next[2] = res[2]-margin; particle_vel[i,2] *= -0.5

        particle_pos[i, 0] = p_next[0]
        particle_pos[i, 1] = p_next[1]
        particle_pos[i, 2] = p_next[2]

        # ... (Color/Trail update) ...
        # Simplified color update
        speed = np.sqrt(particle_vel[i,0]**2 + particle_vel[i,1]**2 + particle_vel[i,2]**2)
        max_v = max(50.0, inlet_velocity)
        t = min(speed / max_v, 1.0)
        
        # Recycled particles get a different color treatment
        if particle_absorbed[i] > 0:
            # Recycled: Warm colors (Red/Orange/Yellow) to White
            # r,g,b = 1.0, t, 0.0 # simple orange-yellow ramp
            # Let's use colormap mode but offset hue or something?
            # Or just fixed color shift
            r, g, b = apply_colormap(t, 1) # Use Rainbow/Heat for recycled
        else:
            # Initial: Cool colors (Blue/Cyan)
            # r, g, b = apply_colormap(t, 0) # Blue-Red default
            # Force Blueish for initial
            r, g, b = 0.0, t * 0.5 + 0.5, 1.0 # Cyan/Blue range
            
        particle_color[i, 0] = r
        particle_color[i, 1] = g
        particle_color[i, 2] = b

        idx = trail_index[i]
        trail_positions[i, idx, 0] = p_next[0]
        trail_positions[i, idx, 1] = p_next[1]
        trail_positions[i, idx, 2] = p_next[2]
        trail_index[i] = (idx + 1) % trail_length

class FluidSolver:
    def __init__(self, res_x=GRID_RES[0], res_y=GRID_RES[1], res_z=GRID_RES[2], num_particles=DEFAULT_NUM_PARTICLES):
        self.res = np.array([res_x, res_y, res_z], dtype=np.int32)
        self.dx = 1.0
        self.dt = DT
        
        self.inlet_face = 0
        self.outlet_face = 1
        self.inlet_p1 = 0.0 # Y (if face 0/1) or X (if face 2/3)
        self.inlet_p2 = 0.0 # Z
        self.inlet_radius = 6.0
        self.outlet_p1 = 0.0
        self.outlet_p2 = 0.0
        self.outlet_radius = 6.0
        
        self.inlet_velocity = 50.0
        self.outlet_velocity = 50.0
        # ... (rest of init)
        
        # Velocity fields
        self.velocity = np.zeros((res_x, res_y, res_z, 3), dtype=np.float64)
        self.new_velocity = np.zeros((res_x, res_y, res_z, 3), dtype=np.float64)
        self.pressure = np.zeros((res_x, res_y, res_z), dtype=np.float64)
        self.new_pressure = np.zeros((res_x, res_y, res_z), dtype=np.float64)
        self.divergence = np.zeros((res_x, res_y, res_z), dtype=np.float64)
        
        # Particles
        self.num_particles = num_particles
        self.particle_pos = np.zeros((num_particles, 3), dtype=np.float64)
        self.particle_vel = np.zeros((num_particles, 3), dtype=np.float64)
        self.particle_color = np.zeros((num_particles, 3), dtype=np.float64)
        self.particle_life = np.zeros(num_particles, dtype=np.float64)
        self.particle_absorbed = np.zeros(num_particles, dtype=np.int32)
        
        self.colormap_mode = 0
        self.trail_length = 20
        self.trail_positions = np.zeros((num_particles, self.trail_length, 3), dtype=np.float64)
        self.trail_index = np.zeros(num_particles, dtype=np.int32)
        
        self.max_obstacles = 10
        self.num_obstacles = 0
        self.obstacle_data = np.zeros((self.max_obstacles, 5), dtype=np.float64)
        
        self.init_particles()

    def update_params(self, inlet_face, outlet_face, in_p1, in_p2, in_rad, out_p1, out_p2, out_rad, in_flow, out_flow):
        self.inlet_face = int(inlet_face)
        self.outlet_face = int(outlet_face)
        self.inlet_p1 = float(in_p1)
        self.inlet_p2 = float(in_p2)
        self.inlet_radius = float(in_rad)
        self.outlet_p1 = float(out_p1)
        self.outlet_p2 = float(out_p2)
        self.outlet_radius = float(out_rad)
        
        # Calc velocity from flow rate
        # Flow is roughly Volume/Time. Area = pi * r^2.
        # Scale factor adjusted to make parameter responsive but not explosive.
        # radius in grid units.
        self.inlet_velocity = float(in_flow) / (3.14 * max(0.1, in_rad)**2) * 2.0 
        self.outlet_velocity = float(out_flow) / (3.14 * max(0.1, out_rad)**2) * 2.0

    def init_particles(self):
        init_particles_kernel(self.particle_pos, self.particle_vel, self.particle_color,
                              self.particle_life, self.particle_absorbed, self.trail_positions,
                              self.trail_index, self.res, self.num_particles, self.trail_length)

    def divergence_calc(self):
        divergence_calc_kernel(self.velocity, self.divergence, self.res)

    def pressure_jacobi(self):
        pressure_jacobi_kernel(self.pressure, self.new_pressure, self.divergence, self.res)
        self.pressure[:] = self.new_pressure

    def project(self):
        project_kernel(self.velocity, self.pressure, self.res)

    def step(self):
        apply_inlet_boundary_kernel(self.velocity, self.res, self.inlet_face, self.inlet_p1, self.inlet_p2, self.inlet_radius, self.inlet_velocity,
                                    self.outlet_face, self.outlet_p1, self.outlet_p2, self.outlet_radius, self.outlet_velocity)
        advect_kernel(self.velocity, self.new_velocity, self.dt, self.res, 
                      self.inlet_face, self.inlet_p1, self.inlet_p2, self.inlet_radius, self.inlet_velocity,
                      self.outlet_face, self.outlet_p1, self.outlet_p2, self.outlet_radius, self.outlet_velocity)
        self.velocity[:] = self.new_velocity
        
        apply_walls_kernel(self.velocity, self.res, self.inlet_face, self.inlet_p1, self.inlet_p2, self.inlet_radius,
                           self.outlet_face, self.outlet_p1, self.outlet_p2, self.outlet_radius)
        
        self.divergence_calc()
        for _ in range(DIVERGENCE_ITERATIONS):
            self.pressure_jacobi()
        self.project()
        
        # Enforce inlet boundary AFTER projection to prevent pressure from killing the inflow
        apply_inlet_boundary_kernel(self.velocity, self.res, self.inlet_face, self.inlet_p1, self.inlet_p2, self.inlet_radius, self.inlet_velocity,
                                    self.outlet_face, self.outlet_p1, self.outlet_p2, self.outlet_radius, self.outlet_velocity)
        
        advect_particles_kernel(self.particle_pos, self.particle_vel, self.particle_color,
                                self.particle_life, self.particle_absorbed, self.trail_positions, self.trail_index,
                                self.velocity, self.res, self.dt,
                                self.inlet_face, self.inlet_p1, self.inlet_p2, self.inlet_radius, self.inlet_velocity,
                                self.outlet_face, self.outlet_p1, self.outlet_p2, self.outlet_radius, self.outlet_velocity,
                                self.num_particles, self.trail_length, self.colormap_mode,
                                self.obstacle_data, self.num_obstacles)

    def update_dimensions(self, res_x, res_y, res_z):
        """グリッド解像度を更新"""
        self.res = np.array([res_x, res_y, res_z], dtype=np.int32)
        # グリッド配列の再確保
        self.velocity = np.zeros((res_x, res_y, res_z, 3), dtype=np.float64)
        self.new_velocity = np.zeros((res_x, res_y, res_z, 3), dtype=np.float64)
        self.pressure = np.zeros((res_x, res_y, res_z), dtype=np.float64)
        self.new_pressure = np.zeros((res_x, res_y, res_z), dtype=np.float64)
        self.divergence = np.zeros((res_x, res_y, res_z), dtype=np.float64)

    def update_particle_count(self, num_particles):
        """粒子数を更新"""
        self.num_particles = num_particles
        # 粒子配列の再確保
        self.particle_pos = np.zeros((num_particles, 3), dtype=np.float64)
        self.particle_vel = np.zeros((num_particles, 3), dtype=np.float64)
        self.particle_color = np.zeros((num_particles, 3), dtype=np.float64)
        self.particle_life = np.zeros(num_particles, dtype=np.float64)
        self.particle_absorbed = np.zeros(num_particles, dtype=np.int32)
        
        self.trail_positions = np.zeros((num_particles, self.trail_length, 3), dtype=np.float64)
        self.trail_index = np.zeros(num_particles, dtype=np.int32)
        
        self.init_particles()
