import numpy as np
import sys
import math

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from .config import GRID_RES, DEFAULT_NUM_PARTICLES, DT, DIVERGENCE_ITERATIONS

# CUDA Kernel for Particle Advection
# This is a direct port of the Numba advect_particles_kernel and sample_velocity_single logic
PARTICLE_KERNEL_SOURCE = r'''
extern "C" {

__device__ float random_float(unsigned int seed) {
    seed = (seed ^ 0x27bb2ee6u) * 0x76543210u;
    seed ^= (seed << 13);
    seed ^= (seed >> 17);
    seed ^= (seed << 5);
    return (float)(seed % 10000) / 10000.0f;
}



__device__ float3 trilinear_interp(const float* velocity, int res_x, int res_y, int res_z, 
                                  float x, float y, float z) {
    int x0 = (int)floor(x);
    int y0 = (int)floor(y);
    int z0 = (int)floor(z);
    
    // Clamp
    if (x0 < 0) x0 = 0; if (x0 >= res_x - 1) x0 = res_x - 2;
    if (y0 < 0) y0 = 0; if (y0 >= res_y - 1) y0 = res_y - 2;
    if (z0 < 0) z0 = 0; if (z0 >= res_z - 1) z0 = res_z - 2;
    
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    int z1 = z0 + 1;
    
    float fx = x - x0;
    float fy = y - y0;
    float fz = z - z0;
    
    int s_z = 3;
    int s_y = 3 * res_z;
    int s_x = 3 * res_z * res_y;
    
    float3 result;
    
    for (int c = 0; c < 3; c++) {
        float c000 = velocity[x0*s_x + y0*s_y + z0*s_z + c];
        float c100 = velocity[x1*s_x + y0*s_y + z0*s_z + c];
        float c010 = velocity[x0*s_x + y1*s_y + z0*s_z + c];
        float c110 = velocity[x1*s_x + y1*s_y + z0*s_z + c];
        float c001 = velocity[x0*s_x + y0*s_y + z1*s_z + c];
        float c101 = velocity[x1*s_x + y0*s_y + z1*s_z + c];
        float c011 = velocity[x0*s_x + y1*s_y + z1*s_z + c];
        float c111 = velocity[x1*s_x + y1*s_y + z1*s_z + c];
        
        float c00 = c000 * (1.0f - fx) + c100 * fx;
        float c10 = c010 * (1.0f - fx) + c110 * fx;
        float c01 = c001 * (1.0f - fx) + c101 * fx;
        float c11 = c011 * (1.0f - fx) + c111 * fx;
        
        float c0 = c00 * (1.0f - fy) + c10 * fy;
        float c1 = c01 * (1.0f - fy) + c11 * fy;
        
        float val = c0 * (1.0f - fz) + c1 * fz;
        
        if (c == 0) result.x = val;
        else if (c == 1) result.y = val;
        else result.z = val;
    }
    return result;
}

__device__ float3 sample_velocity_single(const float* velocity, float px, float py, float pz,
                                        int res_x, int res_y, int res_z,
                                        int inlet_face, int outlet_face,
                                        float inlet_y, float inlet_z, float inlet_radius, float inlet_velocity,
                                        float outlet_y, float outlet_z, float outlet_radius, float outlet_velocity) {
    
    float3 vel = make_float3(0.0f, 0.0f, 0.0f);
    
    if (px >= 0 && px < res_x - 1 && py >= 0 && py < res_y - 1 && pz >= 0 && pz < res_z - 1) {
        vel = trilinear_interp(velocity, res_x, res_y, res_z, px, py, pz);
    } else {
        bool in_inlet = false;
        
        // Check Inlet
        if (inlet_face == 0 && px < 0) { // Left
             if (fabs(py - inlet_y) < (inlet_radius * 1.5f) && fabs(pz - inlet_z) < (inlet_radius * 1.5f)) {
                 in_inlet = true; vel.x = inlet_velocity;
             }
        } else if (inlet_face == 1 && px >= res_x - 1) { // Right
             if (fabs(py - inlet_y) < (inlet_radius * 1.5f) && fabs(pz - inlet_z) < (inlet_radius * 1.5f)) {
                 in_inlet = true; vel.x = -inlet_velocity;
             }
        } else if (inlet_face == 2 && py < 0) { // Bottom
             if (fabs(px - inlet_y) < (inlet_radius * 1.5f) && fabs(pz - inlet_z) < (inlet_radius * 1.5f)) { 
                 in_inlet = true; vel.y = inlet_velocity;
             }
        } else if (inlet_face == 3 && py >= res_y - 1) { // Top
             if (fabs(px - inlet_y) < (inlet_radius * 1.5f) && fabs(pz - inlet_z) < (inlet_radius * 1.5f)) {
                 in_inlet = true; vel.y = -inlet_velocity;
             }
        }
        
        // Check Outlet
        if (!in_inlet) {
            if (outlet_face == 0 && px < 0) {
                 if (fabs(py - outlet_y) < (outlet_radius * 1.5f) && fabs(pz - outlet_z) < (outlet_radius * 1.5f)) vel.x = -outlet_velocity;
            } else if (outlet_face == 1 && px >= res_x - 1) {
                 if (fabs(py - outlet_y) < (outlet_radius * 1.5f) && fabs(pz - outlet_z) < (outlet_radius * 1.5f)) vel.x = outlet_velocity;
            } else if (outlet_face == 2 && py < 0) {
                 if (fabs(px - outlet_y) < (outlet_radius * 1.5f) && fabs(pz - outlet_z) < (outlet_radius * 1.5f)) vel.y = -outlet_velocity;
            } else if (outlet_face == 3 && py >= res_y - 1) {
                 if (fabs(px - outlet_y) < (outlet_radius * 1.5f) && fabs(pz - outlet_z) < (outlet_radius * 1.5f)) vel.y = outlet_velocity;
            }
        }
    }
    return vel;
}

__device__ float3 get_face_pos(int face, float c1, float c2, int res_x, int res_y) {
    if (face == 0) return make_float3(0.0f, c1, c2); // Left
    if (face == 1) return make_float3((float)res_x, c1, c2); // Right
    if (face == 2) return make_float3(c1, 0.0f, c2); // Bottom
    return make_float3(c1, (float)res_y, c2); // Top
}

__device__ void apply_colormap_gpu(float t, int mode, float* r, float* g, float* b) {
    if (mode == 0) { // Blue-Red (Default)
        *r = t;
        *g = t * 0.3f;
        *b = 1.0f - t;
    } else if (mode == 1) { // Rainbow
        float h = t * 0.8f * 6.0f;
        int i = (int)floor(h);
        float f = h - i;
        if (i == 0) { *r=1.0f; *g=f; *b=0.0f; }
        else if (i == 1) { *r=1.0f-f; *g=1.0f; *b=0.0f; }
        else if (i == 2) { *r=0.0f; *g=1.0f; *b=f; }
        else if (i == 3) { *r=0.0f; *g=1.0f-f; *b=1.0f; }
        else if (i == 4) { *r=f; *g=0.0f; *b=1.0f; }
        else { *r=1.0f; *g=0.0f; *b=1.0f-f; }
    } else if (mode == 2) { // Cool-Warm
        if (t < 0.5f) {
            float s = t * 2.0f;
            *r = s; *g = s; *b = 1.0f;
        } else {
            float s = (t - 0.5f) * 2.0f;
            *r = 1.0f; *g = 1.0f - s; *b = 1.0f - s;
        }
    } else if (mode == 3) { // Viridis-like
        *r = 0.267f + t * 0.6f;
        *g = 0.004f + t * 0.87f;
        *b = 0.329f + t * 0.3f - t * t * 0.5f;
    }
}

__global__ void advect_particles(
    float* pos_arr, float* vel_arr, float* color_arr, int* absorbed_arr, 
    float* trail_pos_arr, int* trail_idx_arr,
    const float* velocity, int res_x, int res_y, int res_z, float dt,
    int inlet_face, int outlet_face,
    float inlet_y, float inlet_z, float inlet_radius, float inlet_velocity,
    float outlet_y, float outlet_z, float outlet_radius, float outlet_velocity,
    int num_particles, int trail_length, int colormap_mode,
    const float* obstacle_data, int num_obstacles,
    unsigned int frame_seed
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_particles) return;
    
    // Unpack Position
    float3 pos = make_float3(pos_arr[i*3], pos_arr[i*3+1], pos_arr[i*3+2]);
    
    // Sample Velocity
    float3 vel = sample_velocity_single(velocity, pos.x, pos.y, pos.z, res_x, res_y, res_z, 
                                        inlet_face, outlet_face, inlet_y, inlet_z, inlet_radius, inlet_velocity,
                                        outlet_y, outlet_z, outlet_radius, outlet_velocity);
                                        
    // Forces (Inlet/Outlet) - simplified logic (OMITTED UNCHANGED LINES...)

                                        
    // Forces (Inlet/Outlet) - simplified logic
    bool in_tank = (pos.x >= 0 && pos.x < res_x && pos.y >= 0 && pos.y < res_y && pos.z >= 0 && pos.z < res_z);
    
    float3 inlet_pos = get_face_pos(inlet_face, inlet_y, inlet_z, res_x, res_y);
    float3 outlet_pos = get_face_pos(outlet_face, outlet_y, outlet_z, res_x, res_y);
    
    if (in_tank) {
        // Outlet Suction
        float3 to_outlet = make_float3(outlet_pos.x - pos.x, outlet_pos.y - pos.y, outlet_pos.z - pos.z);
        float dist_out = sqrt(to_outlet.x*to_outlet.x + to_outlet.y*to_outlet.y + to_outlet.z*to_outlet.z);
        float max_dim = (float)(res_x > res_y ? res_x : res_y);
        float suction_range = max_dim * 0.5f;
        
        if (dist_out < suction_range && dist_out > 0.1f) {
            float strength = outlet_velocity * 0.05f * (1.0f - dist_out/suction_range);
            vel.x += (to_outlet.x/dist_out) * strength;
            vel.y += (to_outlet.y/dist_out) * strength;
            vel.z += (to_outlet.z/dist_out) * strength;
        }
        
        // Inlet Jet Boost
        float3 to_inlet_vec = make_float3(pos.x - inlet_pos.x, pos.y - inlet_pos.y, pos.z - inlet_pos.z);
        float dist_in = sqrt(to_inlet_vec.x*to_inlet_vec.x + to_inlet_vec.y*to_inlet_vec.y + to_inlet_vec.z*to_inlet_vec.z);
        float jet_range = max_dim * 0.3f;
        
        if (dist_in < jet_range && dist_in > 0.1f) {
            float strength = inlet_velocity * 0.03f * (1.0f - dist_in/jet_range);
            vel.x += (to_inlet_vec.x/dist_in) * strength;
            vel.y += (to_inlet_vec.y/dist_in) * strength;
            vel.z += (to_inlet_vec.z/dist_in) * strength;
        }
        
        // Anti-stagnation jitter
        float speed = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
        if (speed < 3.0f) {
            vel.x += (random_float(frame_seed + i * 10 + 0) - 0.5f) * 5.0f;
            vel.y += (random_float(frame_seed + i * 10 + 1) - 0.5f) * 5.0f;
            vel.z += (random_float(frame_seed + i * 10 + 2) - 0.5f) * 5.0f;
        }
    }
    
    // Integration
    vel_arr[i*3] = vel.x;
    vel_arr[i*3+1] = vel.y;
    vel_arr[i*3+2] = vel.z;
    
    float3 p_next;
    p_next.x = pos.x + vel.x * dt;
    p_next.y = pos.y + vel.y * dt;
    p_next.z = pos.z + vel.z * dt;
    
    // Wall Collisions
    float margin = 2.0f;
    float kick = 20.0f;
    
    // Helper hole checks
    bool in_hole_inlet_x = false;
    bool in_hole_outlet_x = false;
    bool in_hole_inlet_y = false;
    bool in_hole_outlet_y = false;
        
    if ((pow(p_next.y - inlet_y, 2.0f) + pow(p_next.z - inlet_z, 2.0f)) < pow(inlet_radius, 2.0f)) in_hole_inlet_x = true;
    if ((pow(p_next.y - outlet_y, 2.0f) + pow(p_next.z - outlet_z, 2.0f)) < pow(outlet_radius, 2.0f)) in_hole_outlet_x = true;
    if ((pow(p_next.x - inlet_y, 2.0f) + pow(p_next.z - inlet_z, 2.0f)) < pow(inlet_radius, 2.0f)) in_hole_inlet_y = true;
    if ((pow(p_next.x - outlet_y, 2.0f) + pow(p_next.z - outlet_z, 2.0f)) < pow(outlet_radius, 2.0f)) in_hole_outlet_y = true;

    // Left Wall (X=0)
    if (p_next.x < margin) {
        bool has_hole = (inlet_face == 0 && in_hole_inlet_x) || (outlet_face == 0 && in_hole_outlet_x);
        if (!has_hole) {
            p_next.x = margin;
            vel.x *= -0.8f;
            vel.y += (random_float(frame_seed + i * 10 + 3) - 0.5f) * kick;
            vel.z += (random_float(frame_seed + i * 10 + 4) - 0.5f) * kick;
        }
    }
    // Right Wall (X=res_x)
    if (p_next.x > res_x - margin) {
        bool has_hole = (inlet_face == 1 && in_hole_inlet_x) || (outlet_face == 1 && in_hole_outlet_x);
        if (!has_hole) {
            p_next.x = res_x - margin;
            vel.x *= -0.8f;
            vel.y += (random_float(frame_seed + i * 10 + 5) - 0.5f) * kick;
            vel.z += (random_float(frame_seed + i * 10 + 6) - 0.5f) * kick;
        }
    }
    // Floor (Y=0)
    if (p_next.y < margin) {
        bool has_hole = (inlet_face == 2 && in_hole_inlet_y) || (outlet_face == 2 && in_hole_outlet_y);
        if (!has_hole) {
            p_next.y = margin;
            vel.y *= -0.8f;
            vel.x += (random_float(frame_seed + i * 10 + 7) - 0.5f) * kick;
            vel.z += (random_float(frame_seed + i * 10 + 8) - 0.5f) * kick;
        }
    }
    // Ceiling (Y=res_y)
    if (p_next.y > res_y - margin) {
        bool has_hole = (inlet_face == 3 && in_hole_inlet_y) || (outlet_face == 3 && in_hole_outlet_y);
        if (!has_hole) {
            p_next.y = res_y - margin;
            vel.y *= -0.8f;
            vel.x += (random_float(frame_seed + i * 10 + 9) - 0.5f) * kick;
            vel.z += (random_float(frame_seed + i * 10 + 0) - 0.5f) * kick;
        }
    }
    // Back Wall (Z=0)
    if (p_next.z < margin) {
         p_next.z = margin;
         vel.z *= -0.8f;
         vel.x += (random_float(frame_seed + i * 10 + 1) - 0.5f) * kick;
         vel.y += (random_float(frame_seed + i * 10 + 2) - 0.5f) * kick;
    }
    // Front Wall (Z=res_z)
    if (p_next.z > res_z - margin) {
         p_next.z = res_z - margin;
         vel.z *= -0.8f;
         vel.x += (random_float(frame_seed + i * 10 + 3) - 0.5f) * kick;
         vel.y += (random_float(frame_seed + i * 10 + 4) - 0.5f) * kick;
    }
    
    // Obstacle Collision
    for (int k = 0; k < num_obstacles; k++) {
        float ox = obstacle_data[k*5 + 0];
        float oy = obstacle_data[k*5 + 1];
        float oz = obstacle_data[k*5 + 2];
        float osize = obstacle_data[k*5 + 3];
        float otype = obstacle_data[k*5 + 4];
        
        if (otype < 0.5f) { // Sphere
            float dx = p_next.x - ox;
            float dy = p_next.y - oy;
            float dz = p_next.z - oz;
            float dist_sq = dx*dx + dy*dy + dz*dz;
            
            if (dist_sq < osize*osize && dist_sq > 0.0001f) {
                float dist = sqrt(dist_sq);
                float nx = dx / dist;
                float ny = dy / dist;
                float nz = dz / dist;
                
                // Push out
                p_next.x = ox + nx * (osize + 0.1f);
                p_next.y = oy + ny * (osize + 0.1f);
                p_next.z = oz + nz * (osize + 0.1f);
                
                // Reflect velocity
                float v_dot_n = vel.x * nx + vel.y * ny + vel.z * nz;
                vel.x -= 2.0f * v_dot_n * nx * 0.5f; // 0.5 restitution
                vel.y -= 2.0f * v_dot_n * ny * 0.5f;
                vel.z -= 2.0f * v_dot_n * nz * 0.5f;
            }
        } else { // Box
            float half = osize * 0.5f;
            if (fabs(p_next.x - ox) < half && fabs(p_next.y - oy) < half && fabs(p_next.z - oz) < half) {
                float dx = fabs(p_next.x - ox);
                float dy = fabs(p_next.y - oy);
                float dz = fabs(p_next.z - oz);
                
                if (dx >= dy && dx >= dz) {
                    if (p_next.x > ox) p_next.x = ox + half + 0.1f;
                    else p_next.x = ox - half - 0.1f;
                    vel.x *= -0.5f;
                } else if (dy >= dx && dy >= dz) {
                    if (p_next.y > oy) p_next.y = oy + half + 0.1f;
                    else p_next.y = oy - half - 0.1f;
                    vel.y *= -0.5f;
                } else {
                    if (p_next.z > oz) p_next.z = oz + half + 0.1f;
                    else p_next.z = oz - half - 0.1f;
                    vel.z *= -0.5f;
                }
            }
        }
    }
    
    // Recycling Logic
    bool should_recycle = false;
    float outlet_depth = 0.0f;
    
    if (outlet_face == 0) outlet_depth = -p_next.x;
    else if (outlet_face == 1) outlet_depth = p_next.x - res_x;
    else if (outlet_face == 2) outlet_depth = -p_next.y;
    else if (outlet_face == 3) outlet_depth = p_next.y - res_y;
    
    if (outlet_depth > 20.0f) should_recycle = true;
    
    if (should_recycle) {
        // Respawn at inlet
        unsigned int seed = frame_seed + i * 19937;
        float r = sqrt(random_float(seed)) * (inlet_radius * 0.8f);
        float theta = random_float(seed+1) * 6.28318f;
        float u = r * cos(theta);
        float v = r * sin(theta);
        
        if (inlet_face == 0) { // Left
            p_next.x = -19.0f;
            p_next.y = inlet_y + u;
            p_next.z = inlet_z + v;
        } else if (inlet_face == 1) { // Right
            p_next.x = res_x + 19.0f;
            p_next.y = inlet_y + u;
            p_next.z = inlet_z + v;
        } else if (inlet_face == 2) { // Bottom
            p_next.x = inlet_y + u;
            p_next.y = -19.0f;
            p_next.z = inlet_z + v;
        } else if (inlet_face == 3) { // Top
            p_next.x = inlet_y + u;
            p_next.y = res_y + 19.0f;
            p_next.z = inlet_z + v;
        }
        
        // Reset velocity
        vel.x = 0.0f; vel.y = 0.0f; vel.z = 0.0f;
        vel_arr[i*3] = 0.0f;
        vel_arr[i*3+1] = 0.0f;
        vel_arr[i*3+2] = 0.0f;
        
        // Mark as Recycled (Tracer) and turn Pink
        absorbed_arr[i] = 2; 
        color_arr[i*3] = 1.0f; 
        color_arr[i*3+1] = 0.0f; 
        color_arr[i*3+2] = 1.0f;
    }
    
    // Store back
    pos_arr[i*3] = p_next.x;
    pos_arr[i*3+1] = p_next.y;
    pos_arr[i*3+2] = p_next.z;
    
    // Color Update
    if (absorbed_arr[i] == 0) {
        bool in_tank_next = (p_next.x >= 0 && p_next.x < res_x &&
                             p_next.y >= 0 && p_next.y < res_y &&
                             p_next.z >= 0 && p_next.z < res_z);
                             
        bool is_in_pipe = false;
        bool is_outlet_pipe = false;
        
        if (!in_tank_next) {
             float d_in = 1.0e9f;
             float d_out = 1.0e9f;
             
             if (inlet_face < 2) d_in = (p_next.y - inlet_y)*(p_next.y - inlet_y) + (p_next.z - inlet_z)*(p_next.z - inlet_z);
             else d_in = (p_next.x - inlet_y)*(p_next.x - inlet_y) + (p_next.z - inlet_z)*(p_next.z - inlet_z);
             
             if (outlet_face < 2) d_out = (p_next.y - outlet_y)*(p_next.y - outlet_y) + (p_next.z - outlet_z)*(p_next.z - outlet_z);
             else d_out = (p_next.x - outlet_y)*(p_next.x - outlet_y) + (p_next.z - outlet_z)*(p_next.z - outlet_z);
             
             if (d_out < outlet_radius * outlet_radius * 1.0f) {
                 is_in_pipe = true;
                 is_outlet_pipe = true;
             } else if (d_in < inlet_radius * inlet_radius * 1.0f) {
                 is_in_pipe = true;
                 is_outlet_pipe = false;
             }
        }
        
        float r, g, b;
        
        if (is_in_pipe) {
            if (is_outlet_pipe) {
                // Sucked in: Turn Blue/Dark (Hide) and Mark as Absorbed(1)
                r = 0.1f; g = 0.1f; b = 0.5f; 
                absorbed_arr[i] = 1; 
            } else {
                r = 0.0f; g = 1.0f; b = 0.5f; // Cyan
            }
        } else {
            float speed = sqrt(vel.x*vel.x + vel.y*vel.y + vel.z*vel.z);
            float max_v = (inlet_velocity < 50.0f) ? 50.0f : inlet_velocity;
            float t = speed / max_v;
            if (t > 1.0f) t = 1.0f;
            if (should_recycle) t = 0.0f; 
            
            apply_colormap_gpu(t, colormap_mode, &r, &g, &b);
        }
        
        color_arr[i*3] = r;
        color_arr[i*3+1] = g;
        color_arr[i*3+2] = b;
    }
    
    // Trail Update
    int t_idx = trail_idx_arr[i];
    int t_stride = trail_length * 3;
    trail_pos_arr[i * t_stride + t_idx * 3] = p_next.x;
    trail_pos_arr[i * t_stride + t_idx * 3 + 1] = p_next.y;
    trail_pos_arr[i * t_stride + t_idx * 3 + 2] = p_next.z;
    
    trail_idx_arr[i] = (t_idx + 1) % trail_length;
}

__global__ void advect_velocity_kernel(
    float* new_velocity, const float* velocity,
    int res_x, int res_y, int res_z, float dt,
    int inlet_face, int outlet_face,
    float inlet_y, float inlet_z, float inlet_radius, float inlet_velocity,
    float outlet_y, float outlet_z, float outlet_radius, float outlet_velocity
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (idx >= res_x || idy >= res_y || idz >= res_z) return;
    
    // Calculate stride for (res_x, res_y, res_z, 3) C-order
    int s_z = 3;
    int s_y = 3 * res_z;
    int s_x = 3 * res_z * res_y;
    
    int flat_idx = idx * s_x + idy * s_y + idz * s_z;
    
    // Current velocity
    float3 current_vel = make_float3(velocity[flat_idx], velocity[flat_idx+1], velocity[flat_idx+2]);
    
    // Backtrace
    float3 pos = make_float3((float)idx + 0.5f, (float)idy + 0.5f, (float)idz + 0.5f);
    float3 p_back;
    p_back.x = pos.x - current_vel.x * dt;
    p_back.y = pos.y - current_vel.y * dt;
    p_back.z = pos.z - current_vel.z * dt;
    
    // Sample with pipe logic
    float3 new_vel = sample_velocity_single(velocity, p_back.x, p_back.y, p_back.z, res_x, res_y, res_z,
                                            inlet_face, outlet_face, inlet_y, inlet_z, inlet_radius, inlet_velocity,
                                            outlet_y, outlet_z, outlet_radius, outlet_velocity);
                                            
    new_velocity[flat_idx] = new_vel.x;
    new_velocity[flat_idx+1] = new_vel.y;
    new_velocity[flat_idx+2] = new_vel.z;
}

__global__ void apply_obstacles_velocity_kernel(
    float* velocity, int res_x, int res_y, int res_z,
    const float* obstacle_data, int num_obstacles
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int idz = blockDim.z * blockIdx.z + threadIdx.z;
    
    if (idx >= res_x || idy >= res_y || idz >= res_z) return;
    
    float px = (float)idx;
    float py = (float)idy;
    float pz = (float)idz;
    
    for (int k = 0; k < num_obstacles; k++) {
        float ox = obstacle_data[k*5 + 0];
        float oy = obstacle_data[k*5 + 1];
        float oz = obstacle_data[k*5 + 2];
        float osize = obstacle_data[k*5 + 3];
        float otype = obstacle_data[k*5 + 4];
        
        bool inside = false;
        
        if (otype < 0.5f) { // Sphere
            float dist_sq = (px-ox)*(px-ox) + (py-oy)*(py-oy) + (pz-oz)*(pz-oz);
            if (dist_sq < osize*osize) inside = true;
        } else { // Box
            float half = osize * 0.5f;
            if (fabs(px - ox) < half && fabs(py - oy) < half && fabs(pz - oz) < half) inside = true;
        }
        
        if (inside) {
            int s_z = 3;
            int s_y = 3 * res_z;
            int s_x = 3 * res_z * res_y;
            int flat = idx * s_x + idy * s_y + idz * s_z;
            velocity[flat] = 0.0f;
            velocity[flat+1] = 0.0f;
            velocity[flat+2] = 0.0f;
        }
    }
}

}

'''

class FluidSolverGPU:
    def __init__(self, res_x=GRID_RES[0], res_y=GRID_RES[1], res_z=GRID_RES[2], num_particles=DEFAULT_NUM_PARTICLES):
        if not HAS_CUPY:
             raise ImportError("CuPy is not installed.")

        self.res = np.array([res_x, res_y, res_z], dtype=np.int32)
        self.dx = 1.0
        self.dt = DT
        
        # Compile Kernels
        self.advect_particles_fn = cp.RawKernel(PARTICLE_KERNEL_SOURCE, 'advect_particles')
        self.advect_velocity_fn = cp.RawKernel(PARTICLE_KERNEL_SOURCE, 'advect_velocity_kernel')
        self.apply_obstacles_fn = cp.RawKernel(PARTICLE_KERNEL_SOURCE, 'apply_obstacles_velocity_kernel')
        
        # Fluid fields (on GPU)
        self.velocity_gpu = cp.zeros((res_x, res_y, res_z, 3), dtype=cp.float32)
        self.new_velocity_gpu = cp.zeros((res_x, res_y, res_z, 3), dtype=cp.float32)
        self.pressure_gpu = cp.zeros((res_x, res_y, res_z), dtype=cp.float32)
        self.new_pressure_gpu = cp.zeros((res_x, res_y, res_z), dtype=cp.float32)
        self.divergence_gpu = cp.zeros((res_x, res_y, res_z), dtype=cp.float32)
        
        # Particles (on GPU)
        self.num_particles = num_particles
        self.particle_pos_gpu = cp.zeros(num_particles * 3, dtype=cp.float32) # Flattened for easier CUDA access
        self.particle_vel_gpu = cp.zeros(num_particles * 3, dtype=cp.float32)
        self.particle_color_gpu = cp.zeros(num_particles * 3, dtype=cp.float32)
        self.particle_life_gpu = cp.zeros(num_particles, dtype=cp.float32)
        self.particle_absorbed_gpu = cp.zeros(num_particles, dtype=cp.int32)
        
        # Trails
        self.trail_length = 20
        self.trail_positions_gpu = cp.zeros(num_particles * self.trail_length * 3, dtype=cp.float32)
        self.trail_index_gpu = cp.zeros(num_particles, dtype=cp.int32)
        
        # Helper arrays for mesh generation (Host needs Nx3)
        # self.params removed as it was unused
        
        # Obstacles
        self.max_obstacles = 10
        self.num_obstacles = 0
        self.obstacle_data = np.zeros((self.max_obstacles, 5), dtype=np.float32) # Host side (N x 5)
        self.obstacle_data_gpu = cp.zeros(self.max_obstacles * 5, dtype=cp.float32)
        
        self.init_particles()
        
        self.update_params(0, 1, 10.0, 10.0, 5.0, 5.0, 10.0, 10.0, 500.0, 500.0)
        
        self.frame_count = 0


    @property
    def velocity(self):
        return self.velocity_gpu.get()
        
    @property
    def particle_pos(self):
        return self.particle_pos_gpu.get().reshape(-1, 3)
        
    @property
    def particle_color(self):
        return self.particle_color_gpu.get().reshape(-1, 3)

    def init_particles(self):
        # Initialize randomly (flattened)
        pos = np.random.rand(self.num_particles * 3).astype(np.float32)
        for c in range(3):
            pos[c::3] *= (self.res[c] - 2)
            pos[c::3] += 1.0
            
        self.particle_pos_gpu = cp.asarray(pos)
        self.particle_color_gpu[:] = 0.0
        self.particle_color_gpu[1::3] = 1.0
        self.particle_color_gpu[2::3] = 1.0
        
        # Reset other states explicitly (Critical for Reset button)
        self.particle_vel_gpu[:] = 0.0
        self.particle_absorbed_gpu[:] = 0
        self.particle_life_gpu[:] = cp.random.rand(self.num_particles, dtype=cp.float32)
        self.trail_index_gpu[:] = 0
        self.trail_positions_gpu[:] = 0.0  # Trails will streak from 0 for first few frames, acceptable
        
    def update_params(self, inlet_face, outlet_face, in_y, out_y, in_rad, out_rad, in_z, out_z, in_flow_lpm, out_flow_lpm):
        self.inlet_face = int(inlet_face)
        self.outlet_face = int(outlet_face)
        self.inlet_y = float(in_y)
        self.outlet_y = float(out_y)
        self.inlet_radius = float(in_rad)
        self.outlet_radius = float(out_rad)
        self.inlet_z = float(in_z)
        self.outlet_z = float(out_z)
        
        r_in = max(1.0, float(in_rad))
        r_out = max(1.0, float(out_rad))
        
        area_in = math.pi * r_in**2
        area_out = math.pi * r_out**2
        
        q_in_mm3s = float(in_flow_lpm) * 1000000.0 / 60.0
        q_out_mm3s = float(out_flow_lpm) * 1000000.0 / 60.0
        
        v_in_mm = q_in_mm3s / area_in
        v_out_mm = q_out_mm3s / area_out
        
        # Grid scaling (same as CPU)
        v_in_grid = v_in_mm / 10.0
        v_out_grid = v_out_mm / 10.0
        
        self.inlet_velocity = min(500.0, v_in_grid)
        self.outlet_velocity = min(500.0, v_out_grid)
    
    @property
    def velocity(self):
        return cp.asnumpy(self.velocity_gpu)
    
    @property
    def particle_pos(self):
        return cp.asnumpy(self.particle_pos_gpu).reshape((-1, 3))
    
    @property
    def particle_color(self):
        return cp.asnumpy(self.particle_color_gpu).reshape((-1, 3))

    @property
    def particle_vel(self):
        return cp.asnumpy(self.particle_vel_gpu).reshape((-1, 3))

    @property
    def particle_absorbed(self):
        return cp.asnumpy(self.particle_absorbed_gpu)

    def step(self):
        self.advect()
        
        # Sync obstacles and apply
        if self.num_obstacles > 0:
            self.obstacle_data_gpu.set(self.obstacle_data.ravel())
            
            block = (8, 8, 8)
            grid = (
                (self.res[0] + 7) // 8,
                (self.res[1] + 7) // 8,
                (self.res[2] + 7) // 8
            )
            self.apply_obstacles_fn(
                grid, block,
                (
                    self.velocity_gpu,
                    self.res[0], self.res[1], self.res[2],
                    self.obstacle_data_gpu, cp.int32(self.num_obstacles)
                )
            )

        self.apply_walls()
        self.apply_inlet_boundary()
        
        self.divergence_calc()
        for _ in range(DIVERGENCE_ITERATIONS):
            self.pressure_jacobi()
        self.project()
        
        self.apply_inlet_boundary()
        self.advect_particles()
        self.frame_count += 1

    def advect(self):
        # Use custom CUDA kernel for Semi-Lagrangian Advection with Pipe Support
        block = (8, 8, 8)
        grid = (
            (self.res[0] + 7) // 8,
            (self.res[1] + 7) // 8,
            (self.res[2] + 7) // 8
        )
        
        self.advect_velocity_fn(
            grid, block,
            (
                self.new_velocity_gpu, self.velocity_gpu,
                self.res[0], self.res[1], self.res[2], cp.float32(self.dt),
                cp.int32(self.inlet_face), cp.int32(self.outlet_face),
                cp.float32(self.inlet_y), cp.float32(self.inlet_z), cp.float32(self.inlet_radius), cp.float32(self.inlet_velocity),
                cp.float32(self.outlet_y), cp.float32(self.outlet_z), cp.float32(self.outlet_radius), cp.float32(self.outlet_velocity)
            )
        )
        
        # Swap
        self.velocity_gpu[:] = self.new_velocity_gpu

    def apply_inlet_boundary(self):
        # Refined approach: Use a specific RawKernel for boundary application
        if not hasattr(self, 'apply_boundary_fn'):
            source = r'''
            extern "C" __global__ void apply_boundary_kernel(
                float* velocity, int res_x, int res_y, int res_z,
                int face, float vy, float vz, float radius, float v_mag, float dt
            ) {
                int idx = blockDim.x * blockIdx.x + threadIdx.x;
                int idy = blockDim.y * blockIdx.y + threadIdx.y;
                int idz = blockDim.z * blockIdx.z + threadIdx.z;
                
                if (idx >= res_x || idy >= res_y || idz >= res_z) return;
                
                // Check if we are in the boundary layer (depth 5)
                bool active = false;
                
                if (face == 0 && idx < 5) active = true; // Left
                else if (face == 1 && idx >= res_x - 5) active = true; // Right
                else if (face == 2 && idy < 5) active = true; // Bottom
                else if (face == 3 && idy >= res_y - 5) active = true; // Top
                
                if (!active) return;
                
                // Circle check
                // Face 0/1 (X): check (y-vy)^2 + (z-vz)^2 < r^2
                // Face 2/3 (Y): check (x-vy)^2 + (z-vz)^2 < r^2  (Note: vy arg passed as 'coord1', vz as 'coord2')
                
                float c1 = (float)idy;
                float c2 = (float)idz;
                if (face >= 2) c1 = (float)idx;
                
                float dist_sq = (c1 - vy)*(c1 - vy) + (c2 - vz)*(c2 - vz);
                
                if (dist_sq < radius * radius) {
                    int flat_idx = (idx * res_y * res_z + idy * res_z + idz) * 3;
                    
                    float vx = 0.0f;
                    float vy_val = 0.0f;
                    float vz_val = 0.0f;
                    
                    if (face == 0) vx = v_mag;
                    else if (face == 1) vx = -v_mag;
                    else if (face == 2) vy_val = v_mag;
                    else if (face == 3) vy_val = -v_mag;
                    
                    velocity[flat_idx] = vx;
                    velocity[flat_idx+1] = vy_val;
                    velocity[flat_idx+2] = vz_val;
                }
            }
            '''
            self.apply_boundary_fn = cp.RawKernel(source, 'apply_boundary_kernel')

        block = (8, 8, 8)
        grid = (
            (self.res[0] + 7) // 8,
            (self.res[1] + 7) // 8,
            (self.res[2] + 7) // 8
        )
        
        # Apply Inlet
        self.apply_boundary_fn(
            grid, block,
            (
                self.velocity_gpu,
                self.res[0], self.res[1], self.res[2],
                cp.int32(self.inlet_face),
                cp.float32(self.inlet_y), cp.float32(self.inlet_z),
                cp.float32(self.inlet_radius),
                cp.float32(self.inlet_velocity),
                cp.float32(self.dt)
            )
        )
        
        # Apply Outlet
        self.apply_boundary_fn(
            grid, block,
            (
                self.velocity_gpu,
                self.res[0], self.res[1], self.res[2],
                cp.int32(self.outlet_face),
                cp.float32(self.outlet_y), cp.float32(self.outlet_z),
                cp.float32(self.outlet_radius),
                cp.float32(self.outlet_velocity),
                cp.float32(self.dt)
            )
        ) 

    def apply_walls(self):
        # Enforce 0 at boundaries (except holes)
        self.velocity_gpu[0, :, :, 0] = 0
        self.velocity_gpu[-1, :, :, 0] = 0
        self.velocity_gpu[:, 0, :, 1] = 0
        self.velocity_gpu[:, -1, :, 1] = 0
        self.velocity_gpu[:, :, 0, 2] = 0
        self.velocity_gpu[:, :, -1, 2] = 0

    def _shift(self, arr, shift, axis):
        """Helper to shift array with 0-padding (non-periodic)"""
        result = cp.zeros_like(arr)
        if axis == 0:
            if shift > 0: result[shift:] = arr[:-shift]
            elif shift < 0: result[:shift] = arr[-shift:]
        elif axis == 1:
            if shift > 0: result[:, shift:] = arr[:, :-shift]
            elif shift < 0: result[:, :shift] = arr[:, -shift:]
        elif axis == 2:
            if shift > 0: result[:, :, shift:] = arr[:, :, :-shift]
            elif shift < 0: result[:, :, :shift] = arr[:, :, -shift:]
        return result

    def divergence_calc(self):
        # Calculate divergence using central difference
        # div = 0.5 * (u[x+1] - u[x-1] + v[y+1] - v[y-1] + w[z+1] - w[z-1])
        # We use 0 for OOB (Dirichlet 0 velocity at walls)
        
        u = self.velocity_gpu[..., 0]
        v = self.velocity_gpu[..., 1]
        w = self.velocity_gpu[..., 2]
        
        u_plus = self._shift(u, -1, 0)
        u_minus = self._shift(u, 1, 0)
        v_plus = self._shift(v, -1, 1)
        v_minus = self._shift(v, 1, 1)
        w_plus = self._shift(w, -1, 2)
        w_minus = self._shift(w, 1, 2)
        
        self.divergence_gpu = 0.5 * (
            (u_plus - u_minus) +
            (v_plus - v_minus) +
            (w_plus - w_minus)
        )

    def pressure_jacobi(self):
        # Jacobi iteration for Poisson equation: laplacian(p) = div
        # p_new = (p_x+1 + p_x-1 + ... - div) / 6
        # Boundary condition: p=0 outside (Dirichlet)
        
        p = self.pressure_gpu
        
        p_left = self._shift(p, 1, 0)
        p_right = self._shift(p, -1, 0)
        p_down = self._shift(p, 1, 1)
        p_up = self._shift(p, -1, 1)
        p_back = self._shift(p, 1, 2)
        p_front = self._shift(p, -1, 2)
        
        self.new_pressure_gpu[:] = (
            p_left + p_right + p_down + p_up + p_back + p_front - self.divergence_gpu
        ) / 6.0
        
        # Swap
        self.pressure_gpu[:] = self.new_pressure_gpu

    def project(self):
        # Subtract pressure gradient from velocity
        # u -= 0.5 * (p_x+1 - p_x-1)
        
        p = self.pressure_gpu
        
        p_left = self._shift(p, 1, 0)
        p_right = self._shift(p, -1, 0)
        p_down = self._shift(p, 1, 1)
        p_up = self._shift(p, -1, 1)
        p_back = self._shift(p, 1, 2)
        p_front = self._shift(p, -1, 2)
        
        self.velocity_gpu[..., 0] -= 0.5 * (p_right - p_left)
        self.velocity_gpu[..., 1] -= 0.5 * (p_up - p_down)
        self.velocity_gpu[..., 2] -= 0.5 * (p_front - p_back)

    def advect_particles(self):
        grid_dim = (int((self.num_particles + 511) / 512), 1, 1)
        block_dim = (512, 1, 1)
        
        self.advect_particles_fn(
            grid_dim, block_dim,
            (
                self.particle_pos_gpu, self.particle_vel_gpu, self.particle_color_gpu, self.particle_absorbed_gpu,
                self.trail_positions_gpu, self.trail_index_gpu,
                self.velocity_gpu, 
                self.res[0], self.res[1], self.res[2], cp.float32(self.dt),
                self.inlet_face, self.outlet_face,
                cp.float32(self.inlet_y), cp.float32(self.inlet_z), cp.float32(self.inlet_radius), cp.float32(self.inlet_velocity),
                cp.float32(self.outlet_y), cp.float32(self.outlet_z), cp.float32(self.outlet_radius), cp.float32(self.outlet_velocity),
                self.num_particles, self.trail_length, self.colormap_mode,
                self.obstacle_data_gpu, self.num_obstacles,
                cp.uint32(self.frame_count)
            )
        )
