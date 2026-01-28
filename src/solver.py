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
def sample_velocity_single(velocity, pos, res, inlet_y, inlet_z, inlet_radius, inlet_velocity,
                           outlet_y, outlet_z, outlet_radius, outlet_velocity):
    """単一位置での速度サンプリング"""
    vel = np.zeros(3)
    
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
        
    elif pos[0] < 0:
        # Pipe logic
        dz_in = abs(pos[2] - inlet_z)
        dz_out = abs(pos[2] - outlet_z)
        
        if abs(pos[1] - inlet_y) < (inlet_radius * 1.5) and dz_in < (inlet_radius * 1.5):
            vel[0] = inlet_velocity
        elif abs(pos[1] - outlet_y) < (outlet_radius * 1.5) and dz_out < (outlet_radius * 1.5):
            vel[0] = -outlet_velocity
    
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
def advect_kernel(velocity, new_velocity, dt, res, inlet_y, inlet_z, inlet_radius, inlet_velocity,
                  outlet_y, outlet_z, outlet_radius, outlet_velocity):
    """速度場の移流"""
    for i in prange(res[0]):
        for j in range(res[1]):
            for k in range(res[2]):
                pos = np.array([i + 0.5, j + 0.5, k + 0.5])
                p_back = pos - velocity[i, j, k] * dt
                val = sample_velocity_single(velocity, p_back, res, inlet_y, inlet_z, inlet_radius, inlet_velocity,
                                            outlet_y, outlet_z, outlet_radius, outlet_velocity)
                new_velocity[i, j, k, 0] = val[0]
                new_velocity[i, j, k, 1] = val[1]
                new_velocity[i, j, k, 2] = val[2]


@njit(cache=True, parallel=True)
def apply_inlet_boundary_kernel(velocity, res, inlet_y, inlet_z, inlet_radius, inlet_velocity,
                                outlet_y, outlet_z, outlet_radius, outlet_velocity):
    """境界条件適用"""
    for i in prange(min(5, res[0])):
        for j in range(res[1]):
            for k in range(res[2]):
                dist_z_in = (k - inlet_z) ** 2
                dist_z_out = (k - outlet_z) ** 2
                
                if (j - inlet_y) ** 2 + dist_z_in < inlet_radius ** 2:
                    velocity[i, j, k, 0] = inlet_velocity
                    velocity[i, j, k, 1] = 0.0
                    velocity[i, j, k, 2] = 0.0
                elif (j - outlet_y) ** 2 + dist_z_out < outlet_radius ** 2:
                    velocity[i, j, k, 0] = -outlet_velocity
                    velocity[i, j, k, 1] = 0.0
                    velocity[i, j, k, 2] = 0.0
                else:
                    velocity[i, j, k, 0] = 0.0
                    velocity[i, j, k, 1] = 0.0
                    velocity[i, j, k, 2] = 0.0


@njit(cache=True, parallel=True)
def apply_walls_kernel(velocity, res, inlet_y, inlet_z, inlet_radius, outlet_y, outlet_z, outlet_radius):
    """壁境界条件"""
    for i in prange(res[0]):
        for j in range(res[1]):
            for k in range(res[2]):
                if i < 1:
                    is_inlet = ((j - inlet_y) ** 2 + (k - inlet_z) ** 2 < inlet_radius ** 2)
                    is_outlet = ((j - outlet_y) ** 2 + (k - outlet_z) ** 2 < outlet_radius ** 2)
                    if not is_inlet and not is_outlet:
                        velocity[i, j, k, 0] = 0.0
                        velocity[i, j, k, 1] = 0.0
                        velocity[i, j, k, 2] = 0.0
                
                if i >= res[0] - 1:
                    velocity[i, j, k, 0] = 0.0
                if j < 1:
                    velocity[i, j, k, 1] = 0.0
                if j >= res[1] - 1:
                    velocity[i, j, k, 1] = 0.0
                if k < 1:
                    velocity[i, j, k, 2] = 0.0
                if k >= res[2] - 1:
                    velocity[i, j, k, 2] = 0.0


@njit(cache=True, parallel=True)
def divergence_calc_kernel(velocity, divergence, res):
    """発散計算"""
    for i in prange(res[0]):
        for j in range(res[1]):
            for k in range(res[2]):
                l = velocity[i - 1, j, k, 0] if i > 0 else 0.0
                r = velocity[i + 1, j, k, 0] if i < res[0] - 1 else 0.0
                d = velocity[i, j - 1, k, 1] if j > 0 else 0.0
                u = velocity[i, j + 1, k, 1] if j < res[1] - 1 else 0.0
                b = velocity[i, j, k - 1, 2] if k > 0 else 0.0
                f = velocity[i, j, k + 1, 2] if k < res[2] - 1 else 0.0
                divergence[i, j, k] = 0.5 * (r - l + u - d + f - b)


@njit(cache=True, parallel=True)
def pressure_jacobi_kernel(pressure, new_pressure, divergence, res):
    """圧力ヤコビ反復"""
    for i in prange(res[0]):
        for j in range(res[1]):
            for k in range(res[2]):
                pl = pressure[i - 1, j, k] if i > 0 else 0.0
                pr = pressure[i + 1, j, k] if i < res[0] - 1 else 0.0
                pd = pressure[i, j - 1, k] if j > 0 else 0.0
                pu = pressure[i, j + 1, k] if j < res[1] - 1 else 0.0
                pb = pressure[i, j, k - 1] if k > 0 else 0.0
                pf = pressure[i, j, k + 1] if k < res[2] - 1 else 0.0
                new_pressure[i, j, k] = (pl + pr + pd + pu + pb + pf - divergence[i, j, k]) / 6.0


@njit(cache=True, parallel=True)
def project_kernel(velocity, pressure, res):
    """圧力投影"""
    for i in prange(res[0]):
        for j in range(res[1]):
            for k in range(res[2]):
                pl = pressure[i - 1, j, k] if i > 0 else 0.0
                pr = pressure[i + 1, j, k] if i < res[0] - 1 else 0.0
                pd = pressure[i, j - 1, k] if j > 0 else 0.0
                pu = pressure[i, j + 1, k] if j < res[1] - 1 else 0.0
                pb = pressure[i, j, k - 1] if k > 0 else 0.0
                pf = pressure[i, j, k + 1] if k < res[2] - 1 else 0.0
                
                velocity[i, j, k, 0] -= 0.5 * (pr - pl)
                velocity[i, j, k, 1] -= 0.5 * (pu - pd)
                velocity[i, j, k, 2] -= 0.5 * (pf - pb)


@njit(cache=True, parallel=True)
def advect_particles_kernel(particle_pos, particle_vel, particle_color, particle_absorbed,
                            trail_positions, trail_index, velocity, res, dt,
                            inlet_y, inlet_z, inlet_radius, inlet_velocity,
                            outlet_y, outlet_z, outlet_radius, outlet_velocity,
                            num_particles, trail_length, colormap_mode,
                            obstacle_data, num_obstacles):
    """粒子移流（メインカーネル）"""
    for i in prange(num_particles):
        pos = np.array([particle_pos[i, 0], particle_pos[i, 1], particle_pos[i, 2]])
        vel = sample_velocity_single(velocity, pos, res, inlet_y, inlet_z, inlet_radius, inlet_velocity,
                                     outlet_y, outlet_z, outlet_radius, outlet_velocity)
        
        # Circulation forcing
        if pos[0] >= 0:
            # Outlet suction
            outlet_pos = np.array([0.0, outlet_y, outlet_z])
            to_outlet = outlet_pos - pos
            dist_to_outlet = np.sqrt(to_outlet[0]**2 + to_outlet[1]**2 + to_outlet[2]**2)
            
            suction_range = res[0] * 0.5
            if dist_to_outlet < suction_range and dist_to_outlet > 0.1:
                suction_strength = outlet_velocity * 0.05 * (1.0 - dist_to_outlet / suction_range)
                vel = vel + (to_outlet / dist_to_outlet) * suction_strength
            
            # Inlet jet boost
            inlet_pos = np.array([0.0, inlet_y, inlet_z])
            to_inlet = pos - inlet_pos
            dist_to_inlet = np.sqrt(to_inlet[0]**2 + to_inlet[1]**2 + to_inlet[2]**2)
            
            jet_range = res[0] * 0.3
            if dist_to_inlet < jet_range and dist_to_inlet > 0.1 and pos[0] < res[0] * 0.3:
                jet_boost = inlet_velocity * 0.03 * (1.0 - dist_to_inlet / jet_range)
                vel[0] += jet_boost
            
            # Anti-stagnation jitter
            speed = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
            if speed < 3.0:
                vel[0] += (np.random.random() - 0.5) * 5.0
                vel[1] += (np.random.random() - 0.5) * 5.0
                vel[2] += (np.random.random() - 0.5) * 5.0
        
        particle_vel[i, 0] = vel[0]
        particle_vel[i, 1] = vel[1]
        particle_vel[i, 2] = vel[2]
        
        p_next = pos + vel * dt
        
        # Recycle particles
        if p_next[0] < -20.0:
            p_next[0] = -20.0
            p_next[1] = inlet_y + (np.random.random() - 0.5) * (inlet_radius * 0.8)
            p_next[2] = inlet_z + (np.random.random() - 0.5) * (inlet_radius * 0.8)
        
        margin = 2.0
        kick = 20.0
        
        if p_next[0] >= 0:
            dist_sq_in = (p_next[1] - inlet_y)**2 + (p_next[2] - inlet_z)**2
            dist_sq_out = (p_next[1] - outlet_y)**2 + (p_next[2] - outlet_z)**2
            in_hole_inlet = dist_sq_in < inlet_radius**2
            in_hole_outlet = dist_sq_out < outlet_radius**2
            
            # Left wall
            if p_next[0] < margin:
                if not (in_hole_inlet or in_hole_outlet):
                    p_next[0] = margin
                    vel[0] *= -0.8
                    vel[1] += (np.random.random() - 0.5) * kick
                    vel[2] += (np.random.random() - 0.5) * kick
            
            # Right wall
            if p_next[0] > res[0] - margin:
                p_next[0] = res[0] - margin
                vel[0] *= -0.8
                vel[1] += (np.random.random() - 0.5) * kick
                vel[2] += (np.random.random() - 0.5) * kick
            
            # Floor
            if p_next[1] < margin:
                p_next[1] = margin
                vel[1] *= -0.8
                vel[0] += (np.random.random() - 0.5) * kick
                vel[2] += (np.random.random() - 0.5) * kick
            
            # Ceiling
            if p_next[1] > res[1] - margin:
                p_next[1] = res[1] - margin
                vel[1] *= -0.8
                vel[0] += (np.random.random() - 0.5) * kick
                vel[2] += (np.random.random() - 0.5) * kick
            
            # Back wall
            if p_next[2] < margin:
                p_next[2] = margin
                vel[2] *= -0.8
                vel[0] += (np.random.random() - 0.5) * kick
                vel[1] += (np.random.random() - 0.5) * kick
            
            # Front wall
            if p_next[2] > res[2] - margin:
                p_next[2] = res[2] - margin
                vel[2] *= -0.8
                vel[0] += (np.random.random() - 0.5) * kick
                vel[1] += (np.random.random() - 0.5) * kick
            
            # Obstacle collision
            for obs_idx in range(num_obstacles):
                ox = obstacle_data[obs_idx, 0]
                oy = obstacle_data[obs_idx, 1]
                oz = obstacle_data[obs_idx, 2]
                osize = obstacle_data[obs_idx, 3]
                otype = obstacle_data[obs_idx, 4]
                
                if otype < 0.5:  # Sphere
                    to_obs = p_next - np.array([ox, oy, oz])
                    dist = np.sqrt(to_obs[0]**2 + to_obs[1]**2 + to_obs[2]**2)
                    
                    if dist < osize and dist > 0.01:
                        normal = to_obs / dist
                        p_next = np.array([ox, oy, oz]) + normal * (osize + 0.5)
                        vel_normal = vel[0]*normal[0] + vel[1]*normal[1] + vel[2]*normal[2]
                        vel = vel - 2.0 * vel_normal * normal * 0.5
                else:  # Box
                    half = osize * 0.5
                    if (abs(p_next[0] - ox) < half and 
                        abs(p_next[1] - oy) < half and 
                        abs(p_next[2] - oz) < half):
                        dx = abs(p_next[0] - ox)
                        dy = abs(p_next[1] - oy)
                        dz = abs(p_next[2] - oz)
                        
                        if dx >= dy and dx >= dz:
                            if p_next[0] > ox:
                                p_next[0] = ox + half + 0.5
                            else:
                                p_next[0] = ox - half - 0.5
                            vel[0] *= -0.5
                        elif dy >= dx and dy >= dz:
                            if p_next[1] > oy:
                                p_next[1] = oy + half + 0.5
                            else:
                                p_next[1] = oy - half - 0.5
                            vel[1] *= -0.5
                        else:
                            if p_next[2] > oz:
                                p_next[2] = oz + half + 0.5
                            else:
                                p_next[2] = oz - half - 0.5
                            vel[2] *= -0.5
        else:
            # Pipe region
            dist_sq_out = (p_next[1] - outlet_y)**2 + (p_next[2] - outlet_z)**2
            dist_sq_in = (p_next[1] - inlet_y)**2 + (p_next[2] - inlet_z)**2
            
            if dist_sq_out < dist_sq_in:
                max_r = outlet_radius
                target_y = outlet_y
                target_z = outlet_z
                current_dist_sq = dist_sq_out
            else:
                max_r = inlet_radius
                target_y = inlet_y
                target_z = inlet_z
                current_dist_sq = dist_sq_in
            
            if current_dist_sq > max_r**2:
                current_r = np.sqrt(current_dist_sq)
                scale = (max_r - 0.1) / (current_r + 1e-5)
                p_next[1] = target_y + (p_next[1] - target_y) * scale
                p_next[2] = target_z + (p_next[2] - target_z) * scale
        
        particle_vel[i, 0] = vel[0]
        particle_vel[i, 1] = vel[1]
        particle_vel[i, 2] = vel[2]
        
        # Safety check
        is_invalid = False
        if np.isnan(p_next[0]) or np.isnan(p_next[1]) or np.isnan(p_next[2]):
            is_invalid = True
        if p_next[0] < -100 or p_next[0] > res[0] + 100:
            is_invalid = True
        if p_next[1] < -100 or p_next[1] > res[1] + 100:
            is_invalid = True
        if p_next[2] < -100 or p_next[2] > res[2] + 100:
            is_invalid = True
        
        if is_invalid:
            p_next[0] = np.random.random() * (res[0] - 4.0) + 2.0
            p_next[1] = np.random.random() * (res[1] - 4.0) + 2.0
            p_next[2] = np.random.random() * (res[2] - 4.0) + 2.0
            particle_vel[i, 0] = 0.0
            particle_vel[i, 1] = 0.0
            particle_vel[i, 2] = 0.0
        
        particle_pos[i, 0] = p_next[0]
        particle_pos[i, 1] = p_next[1]
        particle_pos[i, 2] = p_next[2]
        
        # Color update
        if particle_absorbed[i] == 0:
            if p_next[0] < 0:
                if abs(p_next[1] - outlet_y) < (outlet_radius * 1.5) and abs(p_next[2] - outlet_z) < (outlet_radius * 1.5):
                    particle_absorbed[i] = 1
                    particle_color[i, 0] = 1.0
                    particle_color[i, 1] = 0.0
                    particle_color[i, 2] = 1.0
                elif abs(p_next[1] - inlet_y) < (inlet_radius * 1.5) and abs(p_next[2] - inlet_z) < (inlet_radius * 1.5):
                    particle_color[i, 0] = 0.0
                    particle_color[i, 1] = 1.0
                    particle_color[i, 2] = 0.5
            else:
                speed = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
                max_v = max(50.0, inlet_velocity)
                t = min(speed / max_v, 1.0)
                r, g, b = apply_colormap(t, colormap_mode)
                particle_color[i, 0] = r
                particle_color[i, 1] = g
                particle_color[i, 2] = b
        
        # Trail update
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
        
        # Parameters
        self.inlet_y = 10.0
        self.inlet_z = res_z / 2.0
        self.inlet_radius = 6.0
        self.outlet_y = res_y - 10.0
        self.outlet_z = res_z / 2.0
        self.outlet_radius = 6.0
        self.inlet_velocity = 50.0
        self.outlet_velocity = 50.0
        self.flow_rate_base = 10000.0
        
        # Velocity fields (4D: x, y, z, component)
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
        
        # Colormap
        self.colormap_mode = 0
        
        # Trails
        self.trail_length = 20
        self.trail_positions = np.zeros((num_particles, self.trail_length, 3), dtype=np.float64)
        self.trail_index = np.zeros(num_particles, dtype=np.int32)
        
        # Obstacles
        self.max_obstacles = 10
        self.num_obstacles = 0
        self.obstacle_data = np.zeros((self.max_obstacles, 5), dtype=np.float64)
        
        self.init_particles()
    
    def update_params(self, in_y, out_y, in_rad, out_rad, in_z, out_z, in_flow_lpm, out_flow_lpm):
        self.inlet_y = float(in_y)
        self.outlet_y = float(out_y)
        self.inlet_radius = float(in_rad)
        self.outlet_radius = float(out_rad)
        self.inlet_z = float(in_z)
        self.outlet_z = float(out_z)
        
        import math
        r_in = max(1.0, float(in_rad))
        r_out = max(1.0, float(out_rad))
        
        area_in = math.pi * r_in**2
        area_out = math.pi * r_out**2
        
        q_in_mm3s = float(in_flow_lpm) * 1000000.0 / 60.0
        q_out_mm3s = float(out_flow_lpm) * 1000000.0 / 60.0
        
        v_in_mm = q_in_mm3s / area_in
        v_out_mm = q_out_mm3s / area_out
        
        v_in_grid = v_in_mm / 10.0
        v_out_grid = v_out_mm / 10.0
        
        self.inlet_velocity = min(500.0, v_in_grid)
        self.outlet_velocity = min(500.0, v_out_grid)
    
    def init_particles(self):
        init_particles_kernel(self.particle_pos, self.particle_vel, self.particle_color,
                             self.particle_life, self.particle_absorbed, self.trail_positions,
                             self.trail_index, self.res, self.num_particles, self.trail_length)
    
    def advect(self):
        advect_kernel(self.velocity, self.new_velocity, self.dt, self.res,
                     self.inlet_y, self.inlet_z, self.inlet_radius, self.inlet_velocity,
                     self.outlet_y, self.outlet_z, self.outlet_radius, self.outlet_velocity)
        self.velocity[:] = self.new_velocity
    
    def apply_inlet_boundary(self):
        apply_inlet_boundary_kernel(self.velocity, self.res,
                                   self.inlet_y, self.inlet_z, self.inlet_radius, self.inlet_velocity,
                                   self.outlet_y, self.outlet_z, self.outlet_radius, self.outlet_velocity)
    
    def apply_walls(self):
        apply_walls_kernel(self.velocity, self.res,
                          self.inlet_y, self.inlet_z, self.inlet_radius,
                          self.outlet_y, self.outlet_z, self.outlet_radius)
    
    def divergence_calc(self):
        divergence_calc_kernel(self.velocity, self.divergence, self.res)
    
    def pressure_jacobi(self):
        pressure_jacobi_kernel(self.pressure, self.new_pressure, self.divergence, self.res)
        self.pressure[:] = self.new_pressure
    
    def project(self):
        project_kernel(self.velocity, self.pressure, self.res)
    
    def advect_particles(self):
        advect_particles_kernel(self.particle_pos, self.particle_vel, self.particle_color,
                               self.particle_absorbed, self.trail_positions, self.trail_index,
                               self.velocity, self.res, self.dt,
                               self.inlet_y, self.inlet_z, self.inlet_radius, self.inlet_velocity,
                               self.outlet_y, self.outlet_z, self.outlet_radius, self.outlet_velocity,
                               self.num_particles, self.trail_length, self.colormap_mode,
                               self.obstacle_data, self.num_obstacles)
    
    def step(self):
        self.apply_inlet_boundary()
        self.advect()
        self.apply_walls()
        
        self.divergence_calc()
        for _ in range(DIVERGENCE_ITERATIONS):
            self.pressure_jacobi()
        
        self.project()
        self.apply_inlet_boundary()
        self.advect_particles()

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
