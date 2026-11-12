"""The simulated microscope and its tasks: given a seed and actions, the images, the microscope state and whether a
`found` succeeds. It knows nothing about models or what they are told.

  env.py         the environment of a task (reset, step);  tasks.py  the tasks (make)
  microscope.py  field of view, magnification, stage, noise;  actions.py  how an action is written
  generation/    the specimen of an episode and where its targets are, and the offline abTEM simulations
  oracle.py      the shortest path (evaluation only);  truth.py  the correct answer to each question for a view
  paths.py       where the simulated specimens are (data/)
"""
