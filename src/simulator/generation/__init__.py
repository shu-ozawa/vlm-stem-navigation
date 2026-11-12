"""The specimens: the image a microscope looks at in one episode and where its targets are (dopant.py,
grain_boundary.py), and the offline simulations that make their images (simulate_dopant.py, simulate_gb.py).

Every specimen class has
  pixel_size_A             [A / px]
  window                   (darkest intensity, brightest atomic column): the reference display
  build(rng)               draw a new episode (placement, orientation); sets image and targets
  image                    the episode's specimen image (float32, rows x cols)
  targets                  geometry.PointTargets | geometry.LineTargets (qualifying targets only)
  markers()                ground truth for the viewer / evaluation: the targets
  episode_info()           episode parameters recorded in the environment info
The physics (abTEM) runs offline in the simulate_*.py scripts; the specimens only compose and place their results.
"""
