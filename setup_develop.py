import configparser
import sys
from setuptools import find_packages
from numpy.distutils.core import Extension, setup


no_fortran_str = '''

###############################################################################
############### Compilation of Fortran sources is disabled!  ##################
###############################################################################
'''

no_extension_str = '''

###############################################################################
############### Fortran-Extensions cannot be installed!      ##################
############### Install Numpy before installing AMfe         ##################
###############################################################################
'''

no_feti_str = '''

###############################################################################
###############    Import of PYFETI-library is disabled!     ##################
###############################################################################
'''


if __name__ == '__main__':
    configuration = configparser.ConfigParser()
    configuration.read('./meta.cfg')
    ext_modules = []

    config = {'name': configuration['metadata']['name'],
              'version': configuration['metadata']['version'],
              'packages': find_packages()
              }

    print(sys.argv)
    if 'no_feti' in sys.argv:
        sys.argv.remove('no_feti')
        print(no_feti_str)
    else:
        pyfeti_ver = configuration['pyfeti']['spec']
        pyfeti_repo = configuration['pyfeti']['dependency_links']

        if 'install_requires' in config:
            config['install_requires'] = config['install_requires'] + [pyfeti_ver]
        else:
            config['install_requires'] = [pyfeti_ver]
        if 'dependency_links' in config:
            config['dependency_links'] = config['dependency_links'] + [pyfeti_repo]
        else:
            config['dependency_links'] = [pyfeti_repo]

    if 'no_fortran' in sys.argv:
        sys.argv.remove('no_fortran')
        print(no_fortran_str)
        from setuptools import setup

        setup(**config)

    else:
        try:
            from setuptools import setup
            from numpy.distutils.core import Extension, setup
            ext_assembly = Extension(name='amfe.f90_assembly',
                                     sources=['amfe/fortran/assembly.f90'],
                                     language='f90',)
            ext_element = Extension(name='amfe.f90_element',
                                    sources=['amfe/fortran/element.pyf',
                                             'amfe/fortran/element.f90'],
                                    language='f90',)
            ext_material = Extension(name='amfe.f90_material',
                                     sources=['amfe/fortran/material.f90'],
                                     language='f90',)

            ext_modules = [ext_assembly, ext_element, ext_material]

            print(config)
            setup(ext_modules=ext_modules, **config)

        except ImportError:
            # from distutils.core import setup
            from setuptools import setup
            print(no_extension_str)
            answer = query_yes_no('Fortran files cannot be installed. \
                                  It is recommended to abort installation and \
                                  first install numpy. Then retry \
                                  installation of AMfe. \
                                  Do you want to continue installation?', 'no')
            if answer:
                setup(**config)
