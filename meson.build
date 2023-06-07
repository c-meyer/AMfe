project('amfe_fortran', 'c',
  version : '0.1',
  license: 'BSD-3',
  meson_version: '>=0.64.0',
  default_options : ['warning_level=2'],
)

add_languages('fortran')

py_mod = import('python')
py = py_mod.find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy, incdir_f2py)


assembly_source = custom_target('f90_assembly.c',
  input : ['src/amfe/fortran/assembly.f90'],  # .f so no F90 wrappers
  output : ['f90_assemblymodule.c', 'f90_assembly-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'f90_assembly', '--lower']
)

py.extension_module('f90_assembly',
  ['src/amfe/fortran/assembly.f90', assembly_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true,
  subdir: 'amfe'
)

material_source = custom_target('f90_material.c',
  input : ['src/amfe/fortran/material.f90'],  # .f so no F90 wrappers
  output : ['f90_materialmodule.c', 'f90_material-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '-m', 'f90_material', '--lower']
)

py.extension_module('f90_material',
  ['src/amfe/fortran/material.f90', material_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true,
  subdir: 'amfe'
)

element_source = custom_target('f90_element.c',
  input : ['src/amfe/fortran/element.pyf'],  # .f so no F90 wrappers
  output : ['f90_elementmodule.c', 'f90_element-f2pywrappers.f'],
  command : [py, '-m', 'numpy.f2py', '@INPUT@', '--lower']
)

py.extension_module('f90_element',
  ['src/amfe/fortran/element.f90', element_source],
  incdir_f2py / 'fortranobject.c',
  include_directories: inc_np,
  dependencies : py_dep,
  install : true,
  subdir: 'amfe'
)

subdir('src/amfe')