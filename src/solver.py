import taichi as ti
import numpy as np
from src.config import *

@ti.data_oriented
class FluidSolver:
    def __init__(self, res_x=GRID_RES[0], res_y=GRID_RES[1], res_z=GRID_RES[2], num_particles=DEFAULT_NUM_PARTICLES):
        self.res = (res_x, res_y, res_z)
        self.dx = 1.0
        self.dt = DT
        
        # Configurable Parameters (Dynamic Fields)
        self.inlet_y = ti.field(dtype=float, shape=())
        self.inlet_z = ti.field(dtype=float, shape=())
        self.inlet_radius = ti.field(dtype=float, shape=())
        
        self.outlet_y = ti.field(dtype=float, shape=())
        self.outlet_z = ti.field(dtype=float, shape=())
        self.outlet_radius = ti.field(dtype=float, shape=())
        
        self.inlet_velocity = ti.field(dtype=float, shape=())
        self.outlet_velocity = ti.field(dtype=float, shape=())
        
        # Initial Values
        self.inlet_y[None] = 10.0
        self.inlet_z[None] = self.res[2] / 2.0
        self.inlet_radius[None] = 6.0
        
        self.outlet_y[None] = self.res[1] - 10.0
        self.outlet_z[None] = self.res[2] / 2.0
        self.outlet_radius[None] = 6.0
        
        self.inlet_velocity[None] = 50.0
        self.outlet_velocity[None] = 50.0
        
        # Physics Params (flow_rate_base remains a Python float)
        self.flow_rate_base = 10000.0 # Arbitrary unit scaling for V*A
        
        # Velocity fields (3D)
        self.velocity = ti.Vector.field(3, dtype=float, shape=self.res)
        self.new_velocity = ti.Vector.field(3, dtype=float, shape=self.res)
        self.pressure = ti.field(dtype=float, shape=self.res)
        self.new_pressure = ti.field(dtype=float, shape=self.res)
        self.divergence = ti.field(dtype=float, shape=self.res)
        
        # Visualization particles
        self.num_particles = num_particles
        self.particle_pos = ti.Vector.field(3, dtype=float, shape=self.num_particles)
        self.particle_vel = ti.Vector.field(3, dtype=float, shape=self.num_particles)
        self.particle_color = ti.Vector.field(3, dtype=float, shape=self.num_particles)
        self.particle_life = ti.field(dtype=float, shape=self.num_particles)
        self.particle_absorbed = ti.field(dtype=int, shape=self.num_particles)  # 吸い込まれたかどうかのフラグ

        self.init_particles()
        
    def update_params(self, in_y, out_y, in_rad, out_rad, in_z, out_z, in_flow_lpm, out_flow_lpm):
        self.inlet_y[None] = float(in_y)
        self.outlet_y[None] = float(out_y)
        self.inlet_radius[None] = float(in_rad)
        self.outlet_radius[None] = float(out_rad)
        self.inlet_z[None] = float(in_z)
        self.outlet_z[None] = float(out_z)
        
        # Calculate Velocity from Flow Rate (L/min)
        # 1 L = 1,000,000 mm^3
        # 1 min = 60 s
        # Q (mm^3/s) = flow_lpm * 1e6 / 60
        # Area (mm^2) = pi * r^2
        # V (mm/s) = Q / Area
        
        # NOTE: Solver works in "Grid Units" (1 unit = 10mm)
        # So we need V_grid = V_mm / 10.0
        
        r_in = max(1.0, float(in_rad))
        r_out = max(1.0, float(out_rad))
        
        import math
        area_in = math.pi * r_in**2
        area_out = math.pi * r_out**2
        
        q_in_mm3s = float(in_flow_lpm) * 1000000.0 / 60.0
        q_out_mm3s = float(out_flow_lpm) * 1000000.0 / 60.0
        
        v_in_mm = q_in_mm3s / area_in
        v_out_mm = q_out_mm3s / area_out
        
        # Convert to Grid Units
        v_in_grid = v_in_mm / 10.0
        v_out_grid = v_out_mm / 10.0
        
        # Clamp velocity maxima to avoid explosion (e.g. 200 grid units/s is FAST)
        self.inlet_velocity[None] = min(500.0, v_in_grid)
        self.outlet_velocity[None] = min(500.0, v_out_grid)

    @ti.kernel
    def init_particles(self):
        for i in range(self.num_particles):
            # Random distribution
            self.particle_pos[i] = ti.Vector([
                ti.random() * self.res[0],
                ti.random() * self.res[1],
                ti.random() * self.res[2]
            ])
            self.particle_life[i] = ti.random()
            self.particle_absorbed[i] = 0  # 初期状態は吸い込まれていない

    @ti.kernel
    def advect(self):
        # Semi-Lagrangian advection
        for I in ti.grouped(self.velocity):
            pos = I + 0.5
            p_back = pos - self.velocity[I] * self.dt
            val = self.sample_velocity(p_back)
            self.new_velocity[I] = val

        for I in ti.grouped(self.velocity):
            self.velocity[I] = self.new_velocity[I]

    @ti.func
    def sample_velocity(self, pos):
        # Handle "Pipe" regions (X < 0)
        vel = ti.Vector([0.0, 0.0, 0.0])
        
        # Check if inside simulation grid
        if pos[0] >= 0 and pos[0] < self.res[0]-1 and \
           pos[1] >= 0 and pos[1] < self.res[1]-1 and \
           pos[2] >= 0 and pos[2] < self.res[2]-1:
            
            # Trilinear interpolation
            base = ti.floor(pos).cast(int)
            frac = pos - base
            
            v000 = self.velocity[base]
            v100 = self.velocity[base + ti.Vector([1, 0, 0])]
            v010 = self.velocity[base + ti.Vector([0, 1, 0])]
            v110 = self.velocity[base + ti.Vector([1, 1, 0])]
            v001 = self.velocity[base + ti.Vector([0, 0, 1])]
            v101 = self.velocity[base + ti.Vector([1, 0, 1])]
            v011 = self.velocity[base + ti.Vector([0, 1, 1])]
            v111 = self.velocity[base + ti.Vector([1, 1, 1])]

            vel = self.trilinear_interp(v000, v100, v010, v110, v001, v101, v011, v111, frac)
            
        elif pos[0] < 0:
            # PIPE LOGIC
            # dist Z
            dz_in = abs(pos[2] - self.inlet_z[None])
            dz_out = abs(pos[2] - self.outlet_z[None])
            
            # If roughly in Y range of pipes AND Z range
            # Check Inlet Pipe
            if abs(pos[1] - self.inlet_y[None]) < (self.inlet_radius[None] * 1.5) and dz_in < (self.inlet_radius[None] * 1.5):
                vel = ti.Vector([self.inlet_velocity[None], 0.0, 0.0]) 
            # Check Outlet Pipe
            elif abs(pos[1] - self.outlet_y[None]) < (self.outlet_radius[None] * 1.5) and dz_out < (self.outlet_radius[None] * 1.5):
                vel = ti.Vector([-self.outlet_velocity[None], 0.0, 0.0]) 
                
        return vel

    @ti.func
    def trilinear_interp(self, c000, c100, c010, c110, c001, c101, c011, c111, frac):
        c00 = c000 * (1 - frac[0]) + c100 * frac[0]
        c10 = c010 * (1 - frac[0]) + c110 * frac[0]
        c01 = c001 * (1 - frac[0]) + c101 * frac[0]
        c11 = c011 * (1 - frac[0]) + c111 * frac[0]
        
        c0 = c00 * (1 - frac[1]) + c10 * frac[1]
        c1 = c01 * (1 - frac[1]) + c11 * frac[1]
        
        return c0 * (1 - frac[2]) + c1 * frac[2]

    @ti.kernel
    def apply_inlet_boundary(self):
        # Apply boundary conditions
        # We explicitly set values near the wall. 
        # Crucial: Reset neighbors to 0 if they are not in the pipe, 
        # to prevent "ghost" sources when moving the pipe.
        
        for i, j, k in self.velocity:
            if i < 5: # Boundary Zone
                # Check Z range for both ports
                dist_z_in = (k - self.inlet_z[None])**2
                dist_z_out = (k - self.outlet_z[None])**2
                
                # Inlet Port Logic
                if (j - self.inlet_y[None])**2 + dist_z_in < self.inlet_radius[None]**2:
                     self.velocity[i, j, k] = ti.Vector([self.inlet_velocity[None], 0.0, 0.0]) 
                
                # Outlet Port Logic
                elif (j - self.outlet_y[None])**2 + dist_z_out < self.outlet_radius[None]**2:
                     self.velocity[i, j, k] = ti.Vector([-self.outlet_velocity[None], 0.0, 0.0]) 
                
                else:
                     # CLEAR OLD VELOCITY to prevent ghost jets
                     self.velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def apply_walls(self):
        for i, j, k in self.velocity:
            # Left Wall (X=0)
            if i < 1: 
                # Check if we are in a hole
                is_inlet = ((j - self.inlet_y[None])**2 + (k - self.inlet_z[None])**2 < self.inlet_radius[None]**2)
                is_outlet = ((j - self.outlet_y[None])**2 + (k - self.outlet_z[None])**2 < self.outlet_radius[None]**2)
                
                if not is_inlet and not is_outlet:
                    self.velocity[i, j, k] = ti.Vector([0.0, 0.0, 0.0])
            
            if i >= self.res[0] - 1: self.velocity[i, j, k][0] = 0
            
            if j < 1: self.velocity[i, j, k][1] = 0
            if j >= self.res[1] - 1: self.velocity[i, j, k][1] = 0
            
            if k < 1: self.velocity[i, j, k][2] = 0
            if k >= self.res[2] - 1: self.velocity[i, j, k][2] = 0

    @ti.kernel
    def divergence_calc(self):
        for I in ti.grouped(self.velocity):
            l = self.velocity[I + ti.Vector([-1, 0, 0])][0] if I[0] > 0 else 0
            r = self.velocity[I + ti.Vector([1, 0, 0])][0] if I[0] < self.res[0]-1 else 0
            d = self.velocity[I + ti.Vector([0, -1, 0])][1] if I[1] > 0 else 0
            u = self.velocity[I + ti.Vector([0, 1, 0])][1] if I[1] < self.res[1]-1 else 0
            b = self.velocity[I + ti.Vector([0, 0, -1])][2] if I[2] > 0 else 0
            f = self.velocity[I + ti.Vector([0, 0, 1])][2] if I[2] < self.res[2]-1 else 0
            
            self.divergence[I] = 0.5 * (r - l + u - d + f - b)

    @ti.kernel
    def pressure_jacobi(self):
        for I in ti.grouped(self.pressure):
            pl = self.pressure[I + ti.Vector([-1, 0, 0])] if I[0] > 0 else 0
            pr = self.pressure[I + ti.Vector([1, 0, 0])] if I[0] < self.res[0]-1 else 0
            pd = self.pressure[I + ti.Vector([0, -1, 0])] if I[1] > 0 else 0
            pu = self.pressure[I + ti.Vector([0, 1, 0])] if I[1] < self.res[1]-1 else 0
            pb = self.pressure[I + ti.Vector([0, 0, -1])] if I[2] > 0 else 0
            pf = self.pressure[I + ti.Vector([0, 0, 1])] if I[2] < self.res[2]-1 else 0
            
            self.new_pressure[I] = (pl + pr + pd + pu + pb + pf - self.divergence[I]) / 6.0

        for I in ti.grouped(self.pressure):
            self.pressure[I] = self.new_pressure[I]

    @ti.kernel
    def project(self):
        for I in ti.grouped(self.velocity):
            pl = self.pressure[I + ti.Vector([-1, 0, 0])] if I[0] > 0 else 0
            pr = self.pressure[I + ti.Vector([1, 0, 0])] if I[0] < self.res[0]-1 else 0
            pd = self.pressure[I + ti.Vector([0, -1, 0])] if I[1] > 0 else 0
            pu = self.pressure[I + ti.Vector([0, 1, 0])] if I[1] < self.res[1]-1 else 0
            pb = self.pressure[I + ti.Vector([0, 0, -1])] if I[2] > 0 else 0
            pf = self.pressure[I + ti.Vector([0, 0, 1])] if I[2] < self.res[2]-1 else 0
            
            grad_p = 0.5 * ti.Vector([pr - pl, pu - pd, pf - pb])
            self.velocity[I] -= grad_p

    @ti.kernel
    def advect_particles(self):
        for i in range(self.num_particles):
            pos = self.particle_pos[i]
            vel = self.sample_velocity(pos)
            
            # ============================================
            # CIRCULATION FORCING
            # ============================================
            # To create proper circulation:
            # 1. Particles NEAR the OUTLET get pulled INTO it (suction effect)
            # 2. Particles NEAR the INLET get pushed OUT of it (jet effect)
            # 3. Anti-stagnation jitter for slow particles
            
            if pos[0] >= 0: # Only inside the tank
                # --- Outlet Suction Zone ---
                # Create a "cone of influence" extending from the outlet into the tank
                outlet_pos = ti.Vector([0.0, self.outlet_y[None], self.outlet_z[None]])
                to_outlet = outlet_pos - pos
                dist_to_outlet = to_outlet.norm()
                
                # Suction strength decreases with distance
                suction_range = self.res[0] * 0.5 # Effective range (half the tank)
                if dist_to_outlet < suction_range and dist_to_outlet > 0.1:
                    # Calculate suction strength (stronger when closer)
                    suction_strength = self.outlet_velocity[None] * 0.05 * (1.0 - dist_to_outlet / suction_range)
                    suction_dir = to_outlet / dist_to_outlet
                    vel = vel + suction_dir * suction_strength
                
                # --- Inlet Jet Boost Zone ---
                # Particles near the inlet get an extra push away from it
                inlet_pos = ti.Vector([0.0, self.inlet_y[None], self.inlet_z[None]])
                to_inlet = pos - inlet_pos
                dist_to_inlet = to_inlet.norm()
                
                jet_range = self.res[0] * 0.3 # Effective range for jet boost
                if dist_to_inlet < jet_range and dist_to_inlet > 0.1 and pos[0] < self.res[0] * 0.3:
                    # Extra push in +X direction
                    jet_boost = self.inlet_velocity[None] * 0.03 * (1.0 - dist_to_inlet / jet_range)
                    vel[0] += jet_boost
                
                # --- Anti-stagnation Jitter ---
                speed = vel.norm()
                min_speed = 3.0
                if speed < min_speed:
                    vel[0] += (ti.random() - 0.5) * 5.0
                    vel[1] += (ti.random() - 0.5) * 5.0
                    vel[2] += (ti.random() - 0.5) * 5.0
            
            self.particle_vel[i] = vel
            
            p_next = pos + vel * self.dt
            
            # ---------------------------------------------------------
            # STRICT PLUMBING LOGIC
            # ---------------------------------------------------------
            
            # 1. Recycle ANY particle that goes deep enough (X < -20)
            # This acts as the "Pump" mechanism. 
            # We don't care exactly where it is, if it's that far back, it goes back to the inlet.
            if p_next[0] < -20.0:
                 p_next = ti.Vector([
                     -20.0,
                     self.inlet_y[None] + (ti.random() - 0.5) * (self.inlet_radius[None] * 0.8),
                     self.inlet_z[None] + (ti.random() - 0.5) * (self.inlet_radius[None] * 0.8)
                 ])

            # 2. Wall & Pipe Constraints (Bouncy Walls)
            # To prevent stagnation (particles sticking to walls), we invert velocity component
            # and push them slightly away from the wall.
            # Increased margin to 2.0 (2 grid cells) to ensure they return to active flow.
            margin = 2.0
            
            # Random Kick Scale for Tangential motion (break stagnation)
            kick = 20.0 # Velocity kick
            
            if p_next[0] >= 0:
                # --- Inside the Main Tank ---
                
                # Check Holes
                dist_sq_in = (p_next[1] - self.inlet_y[None])**2 + (p_next[2] - self.inlet_z[None])**2
                dist_sq_out = (p_next[1] - self.outlet_y[None])**2 + (p_next[2] - self.outlet_z[None])**2
                
                in_hole_inlet = (dist_sq_in < self.inlet_radius[None]**2)
                in_hole_outlet = (dist_sq_out < self.outlet_radius[None]**2)
                
                # Left Wall (X=0)
                if p_next[0] < margin:
                    if not (in_hole_inlet or in_hole_outlet):
                        p_next[0] = margin 
                        vel[0] *= -0.8 # Bounce
                        # Tangential Kick
                        vel[1] += (ti.random() - 0.5) * kick
                        vel[2] += (ti.random() - 0.5) * kick
                    # Else: Pass through to pipe

                # Right Wall (X = ResX)
                elif p_next[0] > self.res[0] - margin: 
                     p_next[0] = self.res[0] - margin
                     vel[0] *= -0.8 # Bounce
                     vel[1] += (ti.random() - 0.5) * kick
                     vel[2] += (ti.random() - 0.5) * kick
                
                # Floor (Y=0)
                if p_next[1] < margin: 
                    p_next[1] = margin
                    vel[1] *= -0.8 # Bounce
                    vel[0] += (ti.random() - 0.5) * kick
                    vel[2] += (ti.random() - 0.5) * kick
                    
                # Ceiling (Y=ResY)
                if p_next[1] > self.res[1] - margin: 
                    p_next[1] = self.res[1] - margin
                    vel[1] *= -0.8 # Bounce
                    vel[0] += (ti.random() - 0.5) * kick
                    vel[2] += (ti.random() - 0.5) * kick
                    
                # Back Wall (Z=0)
                if p_next[2] < margin: 
                    p_next[2] = margin
                    vel[2] *= -0.8 # Bounce
                    vel[0] += (ti.random() - 0.5) * kick
                    vel[1] += (ti.random() - 0.5) * kick
                    
                # Front Wall (Z=ResZ)
                if p_next[2] > self.res[2] - margin: 
                    p_next[2] = self.res[2] - margin
                    vel[2] *= -0.8 # Bounce
                    vel[0] += (ti.random() - 0.5) * kick
                    vel[1] += (ti.random() - 0.5) * kick

            else:
                # --- Inside the Pipe Region (X < 0) ---
                # Keep existing constraint logic but maybe add bounce?
                # For pipes, we want them to flow along, not bounce around too much. 
                # Just keeping them strictly inside is fine, 
                # but if they hit the "rim", we should ensure they slide.
                
                dist_sq_out = (p_next[1] - self.outlet_y[None])**2 + (p_next[2] - self.outlet_z[None])**2
                dist_sq_in = (p_next[1] - self.inlet_y[None])**2 + (p_next[2] - self.inlet_z[None])**2
                
                current_dist_sq = 0.0
                max_r = 1.0
                target_y = 0.0
                target_z = 0.0
                
                if dist_sq_out < dist_sq_in:
                    # Outlet Pipe
                    current_dist_sq = dist_sq_out
                    max_r = self.outlet_radius[None]
                    target_y = self.outlet_y[None]
                    target_z = self.outlet_z[None]
                else:
                    # Inlet Pipe
                    current_dist_sq = dist_sq_in
                    max_r = self.inlet_radius[None]
                    target_y = self.inlet_y[None]
                    target_z = self.inlet_z[None]

                if current_dist_sq > max_r**2:
                     # Project back
                     current_r = ti.sqrt(current_dist_sq)
                     scale = (max_r - 0.1) / (current_r + 1e-5)
                     p_next[1] = target_y + (p_next[1] - target_y) * scale
                     p_next[2] = target_z + (p_next[2] - target_z) * scale
                     # Kill normal velocity? Or just let it slide?
                     # Simple projection is usually enough for pipes.
            
            # Update Velocity for next step (Bouncing)
            self.particle_vel[i] = vel

            # 3. Comprehensive Safety Net
            # Check for NaN, infinite, or out-of-bounds positions
            # If ANY of these are true, respawn the particle inside the tank
            is_invalid = False
            
            # NaN check (NaN != NaN is True)
            if p_next[0] != p_next[0] or p_next[1] != p_next[1] or p_next[2] != p_next[2]:
                is_invalid = True
            
            # Extreme bounds check
            if p_next[0] < -100 or p_next[0] > self.res[0] + 100:
                is_invalid = True
            if p_next[1] < -100 or p_next[1] > self.res[1] + 100:
                is_invalid = True
            if p_next[2] < -100 or p_next[2] > self.res[2] + 100:
                is_invalid = True
            
            if is_invalid:
                # Respawn randomly inside the main tank (NOT in pipe region)
                p_next = ti.Vector([
                    ti.random() * (self.res[0] - 4.0) + 2.0,
                    ti.random() * (self.res[1] - 4.0) + 2.0,
                    ti.random() * (self.res[2] - 4.0) + 2.0
                ])
                vel = ti.Vector([0.0, 0.0, 0.0])
                self.particle_vel[i] = vel
            
            self.particle_pos[i] = p_next
            
            # Color update
            # 一度吸い込まれた粒子は色を変えない
            
            if self.particle_absorbed[i] == 0:
                # まだ吸い込まれていない粒子のみ色を更新
                
                # Check if entering outlet (being absorbed)
                if p_next[0] < 0:
                    if abs(p_next[1] - self.outlet_y[None]) < (self.outlet_radius[None] * 1.5) and abs(p_next[2] - self.outlet_z[None]) < (self.outlet_radius[None] * 1.5):
                        # 吸い込まれた！フラグを立てて色を変える
                        self.particle_absorbed[i] = 1
                        self.particle_color[i] = ti.Vector([1.0, 0.0, 1.0])  # Magenta
                    elif abs(p_next[1] - self.inlet_y[None]) < (self.inlet_radius[None] * 1.5) and abs(p_next[2] - self.inlet_z[None]) < (self.inlet_radius[None] * 1.5):
                        # Inlet pipe (green)
                        self.particle_color[i] = ti.Vector([0.0, 1.0, 0.5])
                else:
                    # Inside Tank: Speed-based color
                    speed = vel.norm()
                    max_v = ti.max(50.0, self.inlet_velocity[None])
                    t = ti.min(speed / max_v, 1.0)
                    
                    # Blue (Slow) -> Red (Fast)
                    self.particle_color[i] = ti.Vector([
                        t,           # R
                        t * 0.5,     # G
                        1.0 - t      # B
                    ])
            # else: 吸い込まれた粒子は色をそのまま維持（マゼンタ）


    def step(self):
        self.apply_inlet_boundary()
        self.advect()
        self.apply_walls()
        
        self.divergence_calc()
        for _ in range(DIVERGENCE_ITERATIONS):
            self.pressure_jacobi()
            
        self.project()
        
        # KEY FIX: Re-apply inlet forcing AFTER projection
        # The projection step often "corrects" the source/sink divergence by zeroing it out.
        # We need to force it back to maintain the flow jet.
        self.apply_inlet_boundary()
        
        self.advect_particles()
