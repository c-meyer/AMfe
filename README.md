AMfe - Finite Element Research Code at the Chair of Applied Mechanics
=====================================================================

(c) 2020 Lehrstuhl für Angewandte Mechanik, Technische Universität München


This Finite Element Research code is developed, maintained and used by a part of the numerics group of AM.

Overview:
---------

1.  [Installation](#installation-of-amfe)
2.  [Documentation](#documentation)
3.  [Workflow](#workflow-for-pre--and-postprocessing)
4.  [Hints](#hints)


Installation of AMfe
--------------------

### Development Version

For managing python packages, the **Python distribution Anaconda** is **highly recommended**.
It has a very easy and effective packaging system and can thus handle all Python sources needed for this project.
For installation and usage of Anaconda checkout http://docs.continuum.io/anaconda/install#anaconda-install.

The following packages should be installed before you install amfe:

   - Python version 3.7 or greater
   - `numpy`, `mkl`, `meson`, `meson-python`
   - A fortran compiler (e.g. gfortran)
   - for building the documentation `sphinx`, `numpydoc`, `sphinx_rtd_theme`
   - for testing `pytest`
   - for checking the code readability: `flake8`

We recommend to create a separate environment in anaconda for your amfe installation.
Then, you have the opportunity to create a new environment for other projects that can have different
requirements (such as python 2.7 instead 3.10). 


For installing the package in development mode, clone the repository (via git clone) and run

    cd AMfe
    conda create --name <environment-name-of-choice> python=3.10
    conda activate <environment-name-of-choice> 
    python -m pip install --no-build-isolation --editable .

in the main folder.

This command automatically builds the fortran routines and installs them locally.
Python-modules are installed in-place,
i.e., when you apply changes to the source code, they will be used the next time the module is loaded.


### Production version

If you do not develop AMfe you can install AMfe with conda dependencies via

    cd AMfe
    conda create --name <environment-name-of-choice> python=3.10
    conda activate <environment-name-of-choice> 
    python -m pip install .


Documentation
-------------

The documentation can be built from source by changing into the folder `cd docs/` and running

    make html

This requires to install sphinx (`conda install sphinx sphinx_rtd_theme numpydoc`) beforehand.
The documentation will be built in the folder `docs/` available as html in `_build`.
If the command above does not work, try to run `python setup.py build_sphinx` in the main-folder
also builds the documentation.

Workflow for Pre- and Postprocessing
------------------------------------
Preprocessing and postprocessing is not part of the code AMfe, but the open source tools gmsh, salome
and Paraview are recommended:

- [gmsh](https://gmsh.info) The open-source meshing tool can create unstructured meshes for 2D and 3D geometries. The geometry can either be built inside the tool or outside in a CAD program with the `.stp`-file imported into gmsh. In order to define volumes for materials or points/lines/surfaces for boundaries, physical groups must be assigned in gmsh.
- [salome](https://salome-platform.org) Salome-Meca is an open source preprocessor. You can export *.med files from Salome to be used in AMfe.
- [ParaView](https://www.paraview.org) With ParaView the results can be analyzed. For showing the displacements, usually it is very handy to apply the *Warp By Vector* filter to see the displaced configuration.

AMfe provides a Mesh-Exporter for the Preprocessor [GiD](http://www.gidhome.com/).
But this is not maintained anymore and will be deprecated in future.
If you are interested in using this, checkout the gid folder in this repository.

Hints
-----

#### Python and the Scientific Ecosystem

Though Python is a general purpose programming language, it provides a great ecosystem for scientific computing.
As resources to learn both, Python as a language and the scientific Python ecosystem,
the following resources are recommended to become familiar with them.
As these topics are interesting for many people on the globe, lots of resources can be found in the internet.

##### Python language:
- [A byte of Python:](http://python.swaroopch.com/) A good introductory tutorial to Python. My personal favorite.
- [Learn Python the hard way:](http://learnpythonthehardway.org/book/) good introductory tutorial to the programming language.

##### Scientific Python Stack (numpy, scipy, matplotlib):
- [Scipy Lecture Notes:](https://www.scipy-lectures.org/) Good and extensive lecture notes which are evolutionary improved online with very good reference on special topics, e.g. sparse matrices in `scipy`.
- [Youtube: Talk about the numpy data type ](https://www.youtube.com/watch?v=EEUXKG97YRw) This amazing talk **is a must-see** for using `numpy` arrays properly. It shows the concept of array manipulations, which are very effective and powerful and extensively used in `amfe`.
- [Youtube: Talk about color maps in matplotlib](https://youtu.be/xAoljeRJ3lU?list=PLYx7XA2nY5Gcpabmu61kKcToLz0FapmHu) This interesting talk is a little off-topic but cetainly worth to see. It is about choosing a good color-map for your diagrams.
- [Youtube: Talk about the HDF5 file format and the use of Python:](https://youtu.be/nddj5OA8LJo?list=PLYx7XA2nY5Gcpabmu61kKcToLz0FapmHu) Maybe of interest, if the HDF5 data structure, in which the simulation data are extracted, is of interest. This video is no must-have.

##### Version Control with git:
- [Cheat sheet with the important git commands](https://www.git-tower.com/blog/git-cheat-sheet/) Good cheatsheet with all the commands needed for git version control.
- [Youtube: git-Workshop](https://youtu.be/Qthor07loHM) This workshop is extensive and time intensive but definetely worth the time spent. It is a great workshop introducing the concepts of git in a well paced manner ([The slides are also available](https://speakerdeck.com/singingwolfboy/get-started-with-git)).
- [Youtube: git-Talk](https://youtu.be/ZDR433b0HJY) Very fast and informative technical talk on git. Though it is a little bit dated, it is definitely worth watching. 

##### gmsh:
- [HowTo for generating structured meshes in gmsh](https://openfoamwiki.net/index.php/2D_Mesh_Tutorial_using_GMSH) This tutorial is about the generation of structured meshes in gmsh, in this case for the use in the CFD-Framework OpenFOAM. Nonetheless, everything can be used for AMfe as well.

#### IDEs:

You are free to use any IDE that you like. We recommend PyCharm developed by jetbrains.

#### Profiling the code

a good profiling tool is the cProfile module. It runs with

    python -m cProfile -o stats.dat myscript.py

The stats.dat file can be analyzed using the `snakeviz`-tool which is a Python tool which is available via `conda` or `pip` and runs with a web-based interface. To start run

    snakeviz stats.dat

in your console.


#### Theory of Finite Elements
The theory for finite elements is very well developed, though the knowledge is quite fragmented.
When it comes to element technology for instance, good benchmarks and guidelines are often missed.
A good guideline is the [Documentation of the CalculiX-Software-Package](http://web.mit.edu/calculix_v2.7/CalculiX/ccx_2.7/doc/ccx/ccx.html)
which covers a lot about element technology, that is also used in AMfe.
CalculiX is also an OpenSource Finite Element software written in FORTRAN an C++.
