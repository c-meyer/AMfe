#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Abaqus ASCII mesh reader for I/O module.

This module is under active development and in beta phase.
Use it on your own risk.

"""
import numpy as np
import pandas as pd

from amfe.logging import log_debug, log_warning, log_error
from amfe.io.mesh.base import MeshReader

__all__ = [
    'AbaqusAsciiMeshReader'
]


class AbaqusAsciiMeshReader(MeshReader):
    """
    Reader for Abaqus ascii files.
    """

    eletypes = {
        'C3D10': 'Tet10',
        'CPS4R': 'Quad4',
        'CPS8R': 'Quad8',
        'C3D4': 'Tet4',
        'C3D6': 'Prism6',
        'C3D8': 'Hexa8',
        'C3D8I': 'Hexa8',
        'C3D20': 'Hexa20',
        'B31': 'straight_line',
        'CONN3D2': 'straight_line'

    }

    amfeshape2num_nodes = {
        'straight_line': 2,
        'quadratic_line': 3,
        'Tri3': 3,
        'Tri6': 6,
        'Quad4': 4,
        'Quad8': 8,
        'Tet4': 4,
        'Tet10': 10,
        'Prism6': 6,
        'Hexa8': 8,
        'Hexa20': 20,
    }

    eletypes_3d = ['C3D4', 'C3D6', 'C3D8', 'C3D8I', 'C3D10', 'C3D20']
    eletypes_2d = ['CPS4R', 'CPS8R', 'B31']

    surface2localelementnodes = {
        'Hexa8': {
            'S1': ('Quad4', np.array([0, 1, 2, 3], dtype=int)),
            'S2': ('Quad4', np.array([4, 7, 6, 5], dtype=int)),
            'S3': ('Quad4', np.array([0, 4, 5, 1], dtype=int)),
            'S4': ('Quad4', np.array([1, 5, 6, 2], dtype=int)),
            'S5': ('Quad4', np.array([2, 6, 7, 3], dtype=int)),
            'S6': ('Quad4', np.array([3, 7, 4, 0], dtype=int)),
        },
        'Tet4': {
            'S1': ('Tri3', np.array([0, 1, 2], dtype=int)),
            'S2': ('Tri3', np.array([0, 3, 1], dtype=int)),
            'S3': ('Tri3', np.array([1, 3, 2], dtype=int)),
            'S4': ('Tri3', np.array([2, 3, 0], dtype=int))
        },
        'Prism6': {
            'S1': ('Tri3', np.array([0, 1, 2], dtype=int)),
            'S2': ('Tri3', np.array([3, 4, 5], dtype=int)),
            'S3': ('Quad4', np.array([0, 1, 4, 3], dtype=int)),
            'S4': ('Quad4', np.array([1, 2, 5, 4], dtype=int)),
            'S5': ('Quad4', np.array([2, 0, 3, 5], dtype=int)),
        },
        'Tet10': {
            # the reordering is applied here: tet10reordering = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 7], dtype=int)
            'S1': ('Tri6', np.array([0, 1, 2, 4, 5, 6], dtype=int)),
            'S2': ('Tri6', np.array([0, 3, 1, 9, 7, 4], dtype=int)),
            'S3': ('Tri6', np.array([1, 3, 2, 7, 8, 5], dtype=int)),
            'S4': ('Tri6', np.array([2, 3, 0, 8, 9, 6], dtype=int))
        },
        'Hexa20': {
            'S1': ('Quad8', np.array([0, 1, 2, 3, 8, 9, 10, 11], dtype=int)),
            'S2': ('Quad8', np.array([4, 7, 6, 5, 15, 14, 13, 11], dtype=int)),
            'S3': ('Quad8', np.array([0, 4, 5, 1, 16, 12, 17, 8], dtype=int)),
            'S4': ('Quad8', np.array([1, 5, 6, 2, 17, 13, 18, 9], dtype=int)),
            'S5': ('Quad8', np.array([2, 6, 7, 3, 18, 14, 19, 10], dtype=int)),
            'S6': ('Quad8', np.array([3, 7, 4, 0, 19, 15, 16, 11], dtype=int)),
        }
    }

    def __init__(self, filename=None, ignore_errors=False):
        super().__init__()
        self._filename = filename
        self._dimension = 2
        self._groups = dict()
        self._max_num_for_eltype = dict()
        self._ignore_errors = ignore_errors

        self._element_ids = []
        self._element_shapes = []
        self._element_connectivity = []

        self._post_elements = {}  # These elements do not have an ID in input file, but need to be created at the end.
        return

    def _get_next_elset_num(self, eltype):
        if eltype not in self._max_num_for_eltype.keys():
            self._max_num_for_eltype[eltype] = 1
        else:
            self._max_num_for_eltype[eltype] += 1
        return "_" + eltype + "_" + str(self._max_num_for_eltype[eltype])

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
        log_debug(__name__, 'Open file {}'.format(self._filename))
        with open(self._filename, 'r') as infile:
            for line in infile:
                while line is not None and infile is not None:
                    line = self._dispatch(line, infile, builder)


        log_debug(__name__, 'End of file reached.')
        # Build elements
        element_df = pd.DataFrame({'shape': self._element_shapes, 'connectivity': self._element_connectivity},
                                  index=self._element_ids)
        del self._element_ids
        del self._element_shapes
        del self._element_connectivity

        for row in element_df.itertuples():
            builder.build_element(row.Index, row.shape, row.connectivity)
        # Build surfaces
        max_elementid = np.max(element_df.index)
        for groupname, entityset in self._post_elements.items():
            group_elementids = []
            for referenced_eleid, surface_str in entityset:
                source_eleshape = element_df.loc[referenced_eleid, 'shape']
                target_eleshape, local_nodeidxs = self.surface2localelementnodes[source_eleshape][surface_str]
                nodeids = np.array(element_df.loc[referenced_eleid, 'connectivity'], dtype=int)[local_nodeidxs]
                max_elementid = max_elementid + 1
                builder.build_element(max_elementid, target_eleshape, nodeids)
                group_elementids.append(max_elementid)
            builder.build_group(groupname, [], group_elementids)
        # Build groups
        for name in self._groups:
            log_debug(__name__, 'Parse group {}'.format(name))
            builder.build_group(name, self._groups[name]['nodes'], self._groups[name]['elements'])
        # Build dimension
        log_debug(__name__, 'Parse dimension = {}'.format(self._dimension))
        builder.build_mesh_dimension(self._dimension)

    def _dispatch(self, line, infile, builder):
        log_debug(__name__, 'Dispatch line {}'.format(line))
        if line.startswith('**'):
            log_debug(__name__, 'Found comment: {}'.format(line))
            line = next(infile)
            return line

        if ',' in line:
            command = (line.split(',')[0]).lower()
        else:
            command = line.strip().lower()
        if command == '*node':
            log_debug(__name__, 'Found *Node section, dispatch to node handler')
            last_line = self._parse_nodes(infile, builder)
            return last_line
        if command == '*element':
            log_debug(__name__, 'Found *Element section, dispatch to element handler')
            last_line = self._parse_elements(line, infile, builder)
            return last_line
        if command == '*elset':
            log_debug(__name__, 'Found *Elset section, dispatch to set handler')
            last_line = self._parse_set(line, infile, builder, 'elements')
            return last_line
        if command == '*nset':
            log_debug(__name__, 'Found *Nset section, dispatch to set handler')
            last_line = self._parse_set(line, infile, builder, 'nodes')
            return last_line
        if command == '*surface':
            log_debug(__name__, 'Found *surface section, dispatch to surface handler')
            last_line = self._parse_surface(line, infile, builder)
            return last_line
        else:
            err_msg = 'Unrecognized command: {}'.format(command)
            log_warning(__name__, err_msg)
            last_line = self._parse_unrecognized(line, infile, builder)
            return last_line

    def _parse_unrecognized(self, line, infile, builder):
        for line in infile:
            if line.startswith('*'):
                log_debug(__name__, 'End of unrecognized section reached')
                return line

    def _parse_surface(self, line, infile, builder):
        is_elementset = False
        is_nodeset = False
        surface_name = None
        entityset = []
        separated_command = [e.strip() for e in line.split(',')]
        for part in separated_command[1:]:
            key_value = part.split('=')
            key = key_value[0].strip()
            try:
                value = key_value[1].strip()
                if key.lower() == 'name':
                    surface_name = value
                elif key.lower() == 'type':
                    if value.lower() == 'node':
                        is_nodeset = True
                    elif value.lower() == 'element':
                        is_elementset = True
                    else:
                        log_warning(__name__, "Could not parse surface {}".format(line))
                else:
                    log_warning(__name__, "Could not parse surface {}".format(line))

            except IndexError:
                log_warning(__name__, "Could not get value for key-value pair {}".format(key_value))

        for line in infile:
            if line.startswith('*'):
                log_debug(__name__, 'End of surface section reached')
                # Build surface:
                if surface_name is not None:
                    groupname = "_SURFACE_" + surface_name
                    if groupname not in self._groups:
                        if is_nodeset:
                            self._groups.update({groupname: {'nodes': entityset, 'elements': []}})
                        elif is_elementset:
                            # Now we need to build elements
                            self._post_elements.update({groupname: entityset})
                    else:
                        log_error(__name__, "SURFACE {} with groupname {} already exists.".format(surface_name, groupname))
                else:
                    log_error(__name__, "SURFACE could not be parsed due to missing surface name.")
                return line
            else:
                if is_elementset:
                    entityset.append((int(line.split(',')[0].strip()), str(line.split(',')[1].strip())))
                else:
                    entityset.append(int(line.split(',')[0].strip()))

    def _parse_nodes(self, infile, builder):
        for line in infile:
            if line.startswith('*'):
                log_debug(__name__, 'End of Node section reached')
                return line
            try:
                separated_node_data = line.split(',')
                nodeid = int(separated_node_data[0])
                x = float(separated_node_data[1])
                y = float(separated_node_data[2])
                try:
                    z = float(separated_node_data[3])
                except IndexError:
                    z = 0.0
            except ValueError:
                raise ValueError('Could not parse line as node: {}'.format(line))
            builder.build_node(nodeid, x, y, z)

    def _parse_elements(self, line, infile, builder):
        tet10reordering = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9, 7], dtype=int)
        separated_command = [e.strip() for e in line.split(',')]
        elementtype = None
        is_elset = False
        elset_name = None
        elementids = []

        for part in separated_command[1:]:
            key_value = part.split('=')
            key = key_value[0].strip()
            try:
                value = key_value[1].strip()
                if key.lower() == 'type':
                    elementtype = value.upper()
                elif key.lower() == 'elset':
                    is_elset = True
                    elset_name = value
            except IndexError:
                log_warning(__name__, "Could not get value for key-value pair {}".format(key_value))

        if elementtype == None or elementtype not in self.eletypes:
            err_string = 'Elementtype {} could not be parsed because it is not supported by AMfe'.format(elementtype)
            if self._ignore_errors:
                log_warning(__name__, err_string)
                for line in infile:
                    if line.startswith('*'):
                        log_debug(__name__, 'End of Element section reached')
                        return line
            else:
                raise ValueError(err_string)
        else:
            amfeshape = self.eletypes[elementtype]
            if elementtype in self.eletypes_3d:
                self._dimension = 3
            for line in infile:
                if line.startswith('*'):
                    log_debug(__name__, 'End of Element section reached')
                    # Always build a group with elementtype:
                    groupname = self._get_next_elset_num(elementtype)
                    if groupname not in self._groups:
                        self._groups.update({groupname: {'nodes': [], 'elements': elementids}})
                    if is_elset and elset_name not in self._groups:
                        self._groups.update({elset_name: {'nodes': [], 'elements': elementids}})
                    return line
                else:
                    try:
                        separated_element_data = line.split(',')
                        elementid = int(separated_element_data[0])
                        num_nodes = self.amfeshape2num_nodes[amfeshape]
                        if num_nodes > 8:
                            nodeids = [int(nodeid) for nodeid in separated_element_data[1:-1]]
                            # continue with next line:
                            line = next(infile)
                            separated_element_data = line.split(',')
                            if num_nodes > 17:
                                nodeids.extend([int(nodeid) for nodeid in separated_element_data[:-1]])
                                line = next(infile)
                                separated_element_data = line.split(',')
                                nodeids.extend([int(nodeid) for nodeid in separated_element_data])
                            else:
                                nodeids.extend([int(nodeid) for nodeid in separated_element_data])
                        else:
                            nodeids = [int(nodeid) for nodeid in separated_element_data[1:]]
                        if amfeshape == 'Tet10':
                            nodeids = np.array(nodeids)[tet10reordering].tolist()
                        if amfeshape == 'straight_line':
                            # TODO: We can only parse elements with 2 nodes at the moment. The orientation node is dropped.
                            nodeids = nodeids[:2]
                        elementids.append(elementid)

                        self._element_ids.append(elementid)
                        self._element_shapes.append(amfeshape)
                        self._element_connectivity.append(nodeids)
                    except ValueError:
                        raise ValueError('Could not parse line as element: {}'.format(line))


    def _parse_set(self, firstline, infile, builder, entitytype):
        if entitytype != 'elements' and entitytype != 'nodes':
            raise ValueError('Entitytype {} is not valid'.format(entitytype))
        separated_command = [e.strip() for e in firstline.split(',')]
        groupname = None
        groupname_prefix = ''
        for part in separated_command[1:]:
            key_value = part.split('=')
            key = key_value[0].strip()
            try:
                value = key_value[1].strip()
                if key.lower() == 'instance' or key.lower() == 'nset' or key.lower() == 'elset':
                    groupname = groupname_prefix + value
                else:
                    log_warning(__name__, "Could not parse set {}".format(firstline))
            except IndexError:
                log_warning(__name__, "Could not get value for key-value pair {}".format(key_value))
        if 'generate' in separated_command:
            generate = True
        else:
            generate = False
        if groupname not in self._groups:
            self._groups.update({groupname: {'nodes': [], 'elements': []}})
        # Collect nodeids or elementids, respectively
        all_ids = []
        for line in infile:
            if line.startswith('*'):
                log_debug(__name__, 'End of {} set section reached'.format(entitytype))
                if len(self._groups[groupname][entitytype]) > 0:
                    all_ids.extend(self._groups[groupname][entitytype])
                    all_ids = list(set(all_ids))
                self._groups[groupname][entitytype] = all_ids
                return line
            if generate:
                rangeargs = [int(n) for n in line.strip().split(',')]
                rangeargs[1] += 1
                entityids = [i for i in range(*rangeargs)]
            else:
                try:
                    entityids = [int(eid) for eid in line.strip().split(',') if eid != '']
                except ValueError:
                    raise ValueError('Entities could not be parsed as set for line{}'.format(line))
            all_ids.extend(entityids)
