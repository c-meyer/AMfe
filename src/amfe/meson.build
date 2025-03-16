py_mod = import('python')
py3 = py_mod.find_installation(pure: false)

python_sources = [
  '__init__.py',
  'forces.py',
  'logging.py',
  'material.py',
  'mesh.py',
  'structural_dynamics.py',
  'tools.py',
  'ui.py',
]

py3.install_sources(
  python_sources,
  subdir: 'amfe'
)

subdir('assembly')
subdir('component')
subdir('constraint')
subdir('element')
subdir('io')
subdir('linalg')
subdir('mapping')
subdir('mor')
subdir('neumann')
subdir('solver')
