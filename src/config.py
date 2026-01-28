# Physical dimensions (mm)
TANK_WIDTH = 1000
TANK_DEPTH = 1000
TANK_HEIGHT = 500

# Simulation Resolution
# We scale down to maintain performance. 1 unit = 10mm
SCALE = 10.0
RES_X = int(TANK_WIDTH / SCALE)  # 100
RES_Y = int(TANK_HEIGHT / SCALE)  # 50
RES_Z = int(TANK_DEPTH / SCALE)  # 100

GRID_RES = (RES_X, RES_Y, RES_Z)

# Time step
DT = 0.03

# Stabilization
DIVERGENCE_ITERATIONS = 50

# Visualization
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
BACKGROUND_COLOR = (0.1, 0.1, 0.1)
PARTICLE_COLOR_LOW = (0.0, 0.5, 1.0)
PARTICLE_COLOR_HIGH = (1.0, 0.2, 0.2)

# Particle Settings
DEFAULT_NUM_PARTICLES = 256000
MIN_NUM_PARTICLES = 10000
MAX_NUM_PARTICLES = 1000000
