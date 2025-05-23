#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Gmsh ascii mesh reader for I/O module.
"""

import numpy as np

from amfe.io.mesh.base import MeshReader
from collections.abc import Iterable

__all__ = [
    'GmshAsciiMeshReader'
]


class GmshAsciiMeshReader(MeshReader):
    """
    Reader for gmsh ascii files.
    """

    eletypes = {
        1: 'straight_line',
        2: 'Tri3',
        3: 'Quad4',
        4: 'Tet4',
        5: 'Hexa8',
        6: 'Prism6',
        7: None,  # Pyramid
        8: 'quadratic_line',
        9: 'Tri6',
        10: None,  # 9 node quad
        11: 'Tet10',
        12: None,  # 27 Node Hex
        13: None,  # 2nd order prism
        14: None,  # 2nd order pyramid
        15: 'point',
        16: 'Quad8',
        17: 'Hexa20',
        18: None,  # 15node 2nd order prism
        19: None,  # 13 node pyramid
        20: None,  # 9 node triangle
        21: 'Tri10',
        22: None,  # 12 node triangle
        23: None,  # 15 node triangle
        24: None,
        25: None,
        26: None,
        27: None,
        28: None,
        29: None,
        30: None,
        31: None,
        92: None,
        93: None
    }

    eletypes_3d = ['Tet4', 'Hexa8', 'Prism6', 'Tet10', 'Hexa20']

    tag_format_start = "$MeshFormat"
    tag_format_end = "$EndMeshFormat"
    tag_nodes_start = "$Nodes"
    tag_nodes_end = "$EndNodes"
    tag_elements_start = "$Elements"
    tag_elements_end = "$EndElements"
    tag_physical_names_start = "$PhysicalNames"
    tag_physical_names_end = "$EndPhysicalNames"

    def __init__(self, filename=None):
        super().__init__()
        self._filename = filename
        self._dimension = 2
        return

    def parse(self, builder):
        """
        Parse the Mesh with builder

        Parameters
        ----------
        builder : MeshConverter
            Mesh converter object that builds the mesh

        Returns
        -------
        None
        """
        with open(self._filename, 'r') as infile:
            # Read all lines into data_geometry
            data_geometry = infile.read().splitlines()

        n_nodes = None
        n_elements = None
        i_nodes_start = None
        i_nodes_end = None
        i_elements_start = None
        i_elements_end = None
        i_format_start = None
        i_format_end = None
        i_physical_names_start = None
        i_physical_names_end = None

        # Store indices of lines where different sections start and end
        for index, s in enumerate(data_geometry):
            if s == self.tag_format_start:  # Start Formatliste
                i_format_start = index + 1
            elif s == self.tag_format_end:  # Ende Formatliste
                i_format_end = index
            elif s == self.tag_nodes_start:  # Start Knotenliste
                i_nodes_start = index + 2
                n_nodes = int(data_geometry[i_nodes_start - 1])
            elif s == self.tag_nodes_end:  # Ende Knotenliste
                i_nodes_end = index
            elif s == self.tag_elements_start:  # Start Elementliste
                i_elements_start = index + 2
                n_elements = int(data_geometry[i_elements_start - 1])
            elif s == self.tag_elements_end:  # Ende Elementliste
                i_elements_end = index
            elif s == self.tag_physical_names_start:  # Start Physical Names
                i_physical_names_start = index + 2
            elif s == self.tag_physical_names_end:
                i_physical_names_end = index

        # build number of nodes and elements
        if n_nodes is not None and n_elements is not None:
            builder.build_no_of_nodes(n_nodes)
            builder.build_no_of_elements(n_elements)
        else:
            raise ValueError('Could not read number of nodes and number of elements in File {}'.format(self._filename))

        # Check if indices could be read:
        if None in [i_nodes_start, i_nodes_end, i_elements_start, i_elements_end, i_format_start, i_format_end]:
            raise ValueError('Could not read start and end tags of format, nodes and elements '
                             'in file {}'.format(self._filename))

        # Check inconsistent dimensions
        if (i_nodes_end - i_nodes_start) != n_nodes \
                or (i_elements_end - i_elements_start) != n_elements:
            raise ValueError('Error while processing the file {}',
                             'Dimensions are not consistent.'.format(self._filename))

        # extract data from file to lists
        list_imported_mesh_format = data_geometry[i_format_start: i_format_end]
        list_imported_nodes = data_geometry[i_nodes_start: i_nodes_end]
        list_imported_elements = data_geometry[i_elements_start: i_elements_end]

        # Dict for physical names
        groupnames = dict()
        if i_physical_names_start is not None and i_physical_names_end is not None:
            list_imported_physical_names = data_geometry[i_physical_names_start: i_physical_names_end]
            # Make a dict for physical names:
            for group in list_imported_physical_names:
                groupinfo = group.split()
                idx = int(groupinfo[1])
                # split double quotes
                name = groupinfo[2][1:-1]
                groupnames.update({idx: name})

        # conversion of the read strings in mesh format to integer and floats
        for j in range(len(list_imported_mesh_format)):
            list_imported_mesh_format[j] = [float(x) for x in
                                            list_imported_mesh_format[j].split()]

        # Build nodes
        for node in list_imported_nodes:
            nodeinfo = node.split()
            nodeid = int(nodeinfo[0])
            x = float(nodeinfo[1])
            y = float(nodeinfo[2])
            z = float(nodeinfo[3])
            builder.build_node(nodeid, x, y, z)

        # Build elements
        groupentities = dict()

        tag_physical_group = {'name': 'physical_group',
                              'dtype': int,
                              'default': 0,
                              'value2elements': {}}

        tag_elementary_model_entity = {'name': 'elementary_model_entity',
                                       'dtype': int,
                                       'default': 0,
                                       'value2elements': {}}

        tag_no_of_mesh_partitions = {'name': 'no_of_mesh_partitions',
                                     'dtype': int,
                                     'default': 0,
                                     'value2elements': {},
                                     }

        tag_partition_id = {'name': 'partition_id',
                            'dtype': int,
                            'default': 0,
                            'value2elements': {}
                            }

        tag_partitions_neighbors = {'name': 'partitions_neighbors',
                                   'dtype': object,
                                   'default': (),
                                   'value2elements': {}
                                   }

        tagattrs = ['physical_group', 'elementary_model_entity', 'partition_id', 'no_of_partitions',
                    'partitions_neighbors']
        tagdicts = [tag_physical_group, tag_elementary_model_entity, tag_partition_id, tag_no_of_mesh_partitions,
                    tag_partitions_neighbors]

        has_partitions = False
        for ele_string in list_imported_elements:
            element = ListElement(ele_string, self.eletypes)
            if element.type in self.eletypes_3d:
                self._dimension = 3

            builder.build_element(element.id, element.type, element.connectivity)
            # Add element to group
            if element.physical_group in groupentities:
                groupentities[element.physical_group].append(element.id)
            else:
                groupentities.update({element.physical_group: [element.id]})

            # add element to tags
            if element.no_of_partitions is not None:
                has_partitions = True

                # add partitions_neighbors to line-elements (because this is not done for 2d Meshes by Gmsh
                if element.type == self.eletypes[1]:
                    for other_ele_string in list_imported_elements:
                        other_element = ListElement(other_ele_string, self.eletypes)
                        if other_element.id is not element.id and other_element.type == element.type and other_element.partition_id != element.partition_id:
                            for node in element.connectivity:
                                if node in other_element.connectivity and other_element.partition_id not in element.partitions_neighbors:
                                    element.no_of_partitions += 1
                                    if element.partitions_neighbors == (None,):
                                        element.partitions_neighbors = (other_element.partition_id,)
                                    else:
                                        if isinstance(element.partitions_neighbors, Iterable):
                                            element.partitions_neighbors += (other_element.partition_id,)
                                        else:
                                            element.partitions_neighbors = tuple((element.partitions_neighbors, other_element.partition_id))
                element.partitions_neighbors = tuple(element.partitions_neighbors)
            if not has_partitions:
                tagdicts = [tag_physical_group, tag_elementary_model_entity]
            else:
                tagdicts = [tag_physical_group, tag_elementary_model_entity, tag_partition_id,
                            tag_no_of_mesh_partitions,
                            tag_partitions_neighbors]

            for tagdict, tagattr in zip(tagdicts, tagattrs):
                value = getattr(element, tagattr)
                if value in tagdict['value2elements']:
                    tagdict['value2elements'][value].append(element.id)
                else:
                    tagdict['value2elements'].update({value: [element.id]})

        # Build groups
        for group in groupentities:
            if group in groupnames:
                builder.build_group(groupnames[group], [], groupentities[group])
            else:
                builder.build_group(group, [], groupentities[group])

        for tagdict in tagdicts:
            builder.build_tag(tagdict['name'], tagdict['value2elements'], tagdict['dtype'], tagdict['default'])

        builder.build_mesh_dimension(self._dimension)
        return


class ListElement:
    def __init__(self, gmsh_string, eletypes):
        """
        Class that provides information about an element that is defined by a string of a Gmsh Ascii File Version 2

        Parameters
        ----------
        gmsh_string: str
            Gmsh string in an Gmsh ASCII File Version 2
            The format is: <id>,<shape>,<no_of_tags>,<tag1>,...,<tagN>,<node1>,<node2>,...,<nodeN>
        eletypes: dict
            dict mapping the numbers of the element to the shape string that is understood by AMfe
            e.g.: {2: 'Tri3',...}
        """
        elementinfo = gmsh_string.split()
        self.id = int(elementinfo[0])
        self.type = eletypes[int(elementinfo[1])]

        self.no_of_tags = int(elementinfo[2])
        self.connectivity = elementinfo[2 + self.no_of_tags + 1:]
        self.connectivity = np.array([int(node) for node in self.connectivity])

        # Change the indices of Tet10-elements, as they are numbered differently
        # from the numbers used in AMFE and ParaView (last two indices permuted)
        if self.type == 'Tet10':
            self.connectivity[np.array([9, 8], dtype=np.intp)] = \
                self.connectivity[np.array([8, 9], dtype=np.intp)]
        # Same node numbering issue with Hexa20
        if self.type == 'Hexa20':
            hexa8_gmsh_swap = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 11, 13, 9, 16, 18, 19,
                                        17, 10, 12, 14, 15], dtype=np.intp)
            self.connectivity[:] = self.connectivity[hexa8_gmsh_swap]
        self.connectivity = self.connectivity.tolist()

        self.physical_group = int(elementinfo[3])
        self.elementary_model_entity = int(elementinfo[4])

        if self.no_of_tags > 3:
            partition_info = [abs(int(tag)) for tag in elementinfo[3:3 + self.no_of_tags][2:]]

            self.no_of_partitions = partition_info[0]
            try:
                self.partition_id = partition_info[1]
            except IndexError:
                self.partition_id = None
            try:
                self.partitions_neighbors = partition_info[2:]
            except IndexError:
                self.partitions_neighbors = []

        else:
            self.no_of_partitions = None

