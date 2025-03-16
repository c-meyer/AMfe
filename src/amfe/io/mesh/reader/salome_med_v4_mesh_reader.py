
#
# Some part of the code of this module is leaned on code from the meshio
# library which is distributed under MIT license.
# The license information of this part is given below:
#
# The MIT License (MIT)
#
# Copyright (c) 2015-2020 meshio developers
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import h5py
import numpy as np

from amfe.io.mesh.base import MeshReader
from amfe.logging import log_warning


class SalomeMedV4MeshReader(MeshReader):
    """
    This Mesh reader can read MED mesh files from SALOME Meca Preprocessor.
    """

    eletypes = {
        'vertex': None,
        'SE2': 'straight_line',
        'SE3': 'quadratic_line',
        'TR3': 'Tri3',
        'TR6': 'Tri6',
        'QU4': 'Quad4',
        'QU8': 'Quad8',
        'TE4': 'Tet4',
        'T10': 'Tet10',
        'HE8': 'Hexa8',
        'H20': 'Hexa20',
        'PY5': None,
        'P13': None,
        'PE6': None,
        'P15': None,
    }

    hexa8reorder = np.array([1, 2, 6, 5, 0, 3, 7, 4], dtype=int)
    hexa20reorder = np.array([1, 2, 6, 5,
                              0, 3, 7, 4,
                              9, 18, 13, 17,
                              11, 19, 15, 16,
                              8, 10, 14, 12])
    tet10reorder = np.array([0, 2, 1, 3, 6, 5, 4, 9, 8, 7], dtype=int)
    tet4reorder = np.array([0, 2, 1, 3], dtype=int)
    reorderings = {'Hexa8': hexa8reorder,
                   'Hexa20': hexa20reorder,
                   'Tet4': tet4reorder,
                   'Tet10': tet10reorder
                   }

    shape2no_of_nodes = {
        'straight_line': 2,
        'quadratic_line': 3,
        'Tri3': 3,
        'Tri6': 6,
        'Quad4': 4,
        'Quad8': 8,
        'Tet4': 4,
        'Tet10': 10,
        'Hexa8': 8,
        'Hexa20': 20
    }

    def __init__(self, filename=None):
        super().__init__()
        self._filename = filename
        return

    def parse(self, builder):
        """

        Parameters
        ----------
        builder: amfe.io.mesh.base.MeshConverter

        Returns
        -------

        """
        with h5py.File(self._filename, 'r') as infile:
            # Get the submesh
            meshes = infile["ENS_MAA"]
            if len(meshes.keys()) != 1:
                raise OSError('Meshfile may only have one mesh')
            mesh_name = list(meshes)[0]
            mesh = meshes[mesh_name]

            if "NOE" not in mesh:
                time_step = mesh.keys()
                if len(time_step) != 1:
                    raise OSError('Mesh must contain only one time-step,'
                                  'found {}'.format(len(time_step)))
                mesh = mesh[list(time_step)[0]]

            nodes = mesh["NOE"]
            no_of_nodes = nodes["COO"].attrs["NBR"]
            node_coordinates = nodes["COO"][:].reshape(no_of_nodes, -1, order='F')
            nodeids = nodes["NUM"]
            builder.build_no_of_nodes(no_of_nodes)
            # Build nodes
            if node_coordinates.shape[1] == 2:
                builder.build_mesh_dimension(2)
                for nodeid, node in zip(nodeids, node_coordinates):
                    builder.build_node(nodeid, node[0], node[1], 0.0)
            else:
                for nodeid, node in zip(nodeids, node_coordinates):
                    builder.build_node(nodeid, node[0], node[1], node[2])

            if "FAM" in nodes:
                node_tags = nodes["FAM"][:]
                unique_tags = np.unique(node_tags)
                nodetag2nodeid = {tag: nodeids[np.argwhere(node_tags == tag).flatten()] for tag in unique_tags}

            fas = mesh["FAS"] if "FAS" in mesh else infile["FAS"][mesh_name]
            if "NOEUD" in fas:
                node_groups = self._read_families(fas["NOEUD"])

            elements = mesh["MAI"]
            no_of_elements = 0
            for medshape in elements.keys():
                shape = self.eletypes[medshape]
                if shape is None:
                    log_warning(__name__, 'Shape {} cannot be converted. This shape is ignored.'.format(shape))
                    continue
                no_of_elements += elements[medshape]["NOD"].attrs["NBR"]

            builder.build_no_of_elements(no_of_elements)
            element_tags = {}

            for medshape in elements.keys():
                shape = self.eletypes[medshape]
                if shape is None:
                    continue
                if shape in self.reorderings.keys():
                    needs_reordering = True
                    current_reordering = self.reorderings[shape]
                else:
                    needs_reordering = False
                    current_reordering = None
                no_of_elements_of_current_shape = elements[medshape]["NOD"].attrs["NBR"]
                connectivities_of_current_shape = elements[medshape]["NOD"][:].reshape(no_of_elements_of_current_shape, -1, order="F")
                for elementid, connectivity, tagid in zip(elements[medshape]["NUM"], connectivities_of_current_shape,
                                            elements[medshape]["FAM"]):
                    if needs_reordering:
                        connectivity = connectivity[current_reordering]
                    builder.build_element(elementid, shape, connectivity)

                    if tagid in element_tags.keys():
                        element_tags[tagid].append(elementid)
                    else:
                        element_tags.update({tagid: [elementid]})

            if "ELEME" in fas:
                element_groups = self._read_families(fas["ELEME"])

            builder.build_tag('salome_family', element_tags, dtype=int)

            groups_set_elements = self._build_groups_dict(element_groups, element_tags)
            groups_to_nodeids = self._build_groups_dict(node_groups, nodetag2nodeid)

            for groupname, groupelements in groups_set_elements.items():
                builder.build_group(groupname, [], np.unique(groupelements))

            for groupname, groupelements in groups_to_nodeids.items():
                builder.build_group(groupname, np.array(groupelements), [])

    @staticmethod
    def _build_groups_dict(element_groups, element_tags):
        groups_set = {}
        for family, groups in element_groups.items():
            for group in groups:
                try:
                    if group in groups_set:
                        groups_set[group].extend(list(element_tags[family]))
                    else:
                        groups_set.update({group: list(element_tags[family].copy())})
                except KeyError:
                    log_warning(__name__, 'There seem no elements assigned to '
                                   'MED-File Family {}. Check if group {} '
                                   'is imported accordingly.'.format(
                        family, group))
        return groups_set

    def _read_families(self, fas_data):
        families = {}
        for _, node_set in fas_data.items():
            set_id = node_set.attrs["NUM"]  # unique set id
            n_subsets = node_set["GRO"].attrs["NBR"]  # number of subsets
            nom_dataset = node_set["GRO"]["NOM"][:]  # (n_subsets, 80) of int8
            name = [None] * n_subsets
            for i in range(n_subsets):
                name[i] = "".join(
                    [chr(x) for x in nom_dataset[i]]).strip().rstrip("\x00")
            families[set_id] = name
        return families

