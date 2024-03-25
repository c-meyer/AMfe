# Copyright (c) 2018, Lehrstuhl für Angewandte Mechanik, Technische Universität München.
#
# Distributed under BSD-3-Clause License. See LICENSE-File for more information
#
import enum

import numpy as np
import pandas as pd
import h5py
import xml.etree.ElementTree as ET
from os.path import basename, splitext

from amfe.io.mesh.base import MeshConverter
from amfe.io.tools import insert_line_breaks_in_xml, check_filename_or_filepointer
from amfe.io.constants import XDMFDICT

__all__ = [
    'Hdf5MeshConverter',
    'write_xdmf_mesh_from_hdf5'
    ]


def write_hdf5_table(fp, arr, path):
    # arr = np.array(range(6), dtype=np.float32).reshape(3, 2)
    dt = np.dtype([('etype', 'S14'), ('etype_index', np.uint32), ('index', np.uint32)])
    # arr = np.array([("quadratic_line", 1, 5),
#                    ("Tri6", 2, 12)], dtype=dt)
    dset = fp.create_dataset(path, (len(arr),), dtype=dt, data=arr)
    dset.attrs.create("CLASS", "TABLE")
    dset.attrs.create("VERSION", "0.2")
    dset.attrs.create("TITLE", "Element IDs")
    dset.attrs.create("FIELD_0_NAME", "etype")
    dset.attrs.create("FIELD_0_FILL", b'')
    dset.attrs.create("FIELD_1_NAME", "etype_index")
    dset.attrs.create("FIELD_1_FILL", 0, dtype=np.uint32)
    dset.attrs.create("FIELD_2_NAME", "index")
    dset.attrs.create("FIELD_2_FILL", 0, dtype=np.uint32)
    dset.attrs.create("NROWS", len(arr))


class Hdf5MeshConverter(MeshConverter):
    """
    MeshConverter to HDF5 files
    """
    class Preallocation(enum.Enum):
        PREALLOCATED = 1
        NOTPREALLOCATED = 2
        UNKNOWN = 0

    def __init__(self, filename):
        super().__init__()
        self._filename = filename
        self._nodes = np.empty((0, 3), dtype=float)
        self._nodes_current_row = 0
        self._node_preallocation = self.Preallocation.UNKNOWN
        self._nodeids2row = dict()
        self._tags = dict()
        self._connectivity = list()
        self._ele_indices = list()
        self._eleshapes = list()
        self._nodes_df = None
        self._el_df = None
        self._no_of_nodes = 0
        self._version = 1.0

    def build_no_of_nodes(self, no):
        if self._node_preallocation == self.Preallocation.UNKNOWN:
            self._nodes = np.zeros((no, 4))
            self._node_preallocation = self.Preallocation.PREALLOCATED

    def build_no_of_elements(self, no):
        pass

    def build_node(self, idx, x, y, z):
        if self._node_preallocation == self.Preallocation.PREALLOCATED:
            self._nodes[self._nodes_current_row, :] = [float(idx), float(x), float(y), float(z)]
        else:
            if self._node_preallocation == self.Preallocation.UNKNOWN:
                self._nodes = np.empty((0, 4), dtype=float)
                self._node_preallocation = self.Preallocation.NOTPREALLOCATED
            self._nodes = np.append(self._nodes, np.array([idx, x, y, z], ndmin=2, dtype=float), axis=0)
        self._nodeids2row.update({id: self._nodes_current_row})
        self._nodes_current_row += 1

    def build_element(self, eid, etype, nodes):
        self._connectivity.append(np.array(nodes, dtype=int))
        self._ele_indices.append(eid)
        self._eleshapes.append(etype)

    def build_group(self, name, nodeids, elementids):
        """

        Parameters
        ----------
        name: str
            Name identifying the node group.
        nodeids: list
            List with node ids.
        elementids: list
            List with element ids.

        Returns
        -------
        None
        """

        pass

    def build_material(self, material):
        pass

    def build_partition(self, partition):
        pass

    def build_mesh_dimension(self, dim):
        pass

    def build_tag(self, tag_name, values2elements, dtype=None, default=None):
        # append tag information
        if dtype is None:
            dtype = object
        self._tags.update({tag_name: {'values2elements': values2elements,
                                      'dtype': dtype,
                                      'default': default
                                      }
                           }
                          )
        return None

    def return_mesh(self):
        self._prepare_return()
        self._write_hdf5(self._filename)

    def _prepare_return(self):
        # Make a pd Dataframe for the nodes
        # If dimension = 2 cut the z coordinate
        x = self._nodes[:, 1]
        y = self._nodes[:, 2]
        z = self._nodes[:, 3]
        self._no_of_nodes = self._nodes.shape[0]

        self._nodes_df = pd.DataFrame({'row': np.arange(self._no_of_nodes), 'x': x, 'y': y, 'z': z}, index=np.array(self._nodes[:, 0], dtype=int))

        # make a pd Dataframe for the elements
        data = {'shape': self._eleshapes,
                'connectivity': self._connectivity}
        self._el_df = pd.DataFrame(data, index=self._ele_indices)
        # introduce row values for each shape. The row values will map to the row entries in a separate array for each
        # elementtype
        etypes_in_el_df = self._el_df['shape'].unique()
        self._el_df['row'] = None
        for etype in etypes_in_el_df:
            self._el_df.loc[self._el_df['shape'] == etype, 'row'] = np.arange(sum(self._el_df['shape'] == etype))

        # Function change connectivity ids to row ids in nodes array:
        def change_connectivity(arr):
            return np.array([self._nodes_df.loc[node, 'row'] for node in arr], ndmin=2, dtype=int)
        self._el_df['connectivity'] = self._el_df['connectivity'].apply(change_connectivity)

        self._tag_names = list()

        name2scalars = dict()
        for tag_name, tag_dict in self._tags.items():
            self._el_df[tag_name] = None
            if tag_dict is not None:
                currentscalar = -1

                for tag_value, elem_list in tag_dict['values2elements'].items():
                    # Check if tag_value is string
                    # if it is a string map the string to a scalar because xdmf does not support strings
                    if isinstance(tag_value, str):
                        name2scalars.update({tag_value: currentscalar})
                        tag_value = currentscalar
                        currentscalar -= 1
                    self._el_df.loc[elem_list, tag_name] = tag_value
            self._tag_names.extend([tag_name])
            self._tags[tag_name].update({'name2scalars': name2scalars})

    @check_filename_or_filepointer(h5py._hl.files.File, h5py.File, 1, writeable=True)
    def _write_hdf5(self, hdf_fp):
        # write hdf5 file and xdmf file
        # create mesh group
        group_mesh = hdf_fp.create_group("/mesh")
        group_mesh.attrs.create("TITLE", "Mesh")
        group_mesh.attrs.create("MESH_VERSION", data=str(self._version))

        nodes_values = self._nodes_df[['x', 'y', 'z']].values
        nodes_ds = group_mesh.create_dataset("nodes", data=nodes_values, shape=nodes_values.shape)
        nodes_ds.attrs.create("TITLE", "Node Coordinates")

        nodeids_ds = group_mesh.create_dataset("nodeids", data=self._nodes_df.index.values)
        nodeids_ds.attrs.create("TITLE", "Node IDs")

        group_elements = group_mesh.create_group("topology")
        group_elements.attrs.create("TITLE", "Topology")

        group_tags = group_mesh.create_group("tags")
        group_tags.attrs.create("TITLE", "Tags")

        tag_group_pointers = dict()
        for tag in self._tag_names:
            tag_group = group_tags.create_group(tag)
            tag_group_pointers.update({tag: tag_group})

        dt = np.dtype([('etype', 'S14'), ('etype_index', np.uint32), ('index', np.uint32)])
        rows = []

        # write topology for each element
        for etype in self._el_df['shape'].unique():
            el_df_by_shape = self._el_df[self._el_df['shape'] == etype]
            for counter, (index, item) in enumerate(el_df_by_shape['shape'].items()):
                rows.append((item, counter, index))

            connectivity_of_current_etype = el_df_by_shape['connectivity'].values
            no_of_elements_of_current_etype = len(connectivity_of_current_etype)
            no_of_nodes_per_element = connectivity_of_current_etype[0].shape[1]
            connectivity_array = np.concatenate(connectivity_of_current_etype, axis=0)
            if etype == 'Tet10':
                reordering = np.array([0, 1, 2, 3, 4, 5, 6, 9, 7, 8],
                                      dtype=int)
                connectivity_array = connectivity_array[:, reordering]
            if no_of_elements_of_current_etype > 0:
                group_elements.create_dataset(etype, data=connectivity_array.astype(int),
                                              shape=(no_of_elements_of_current_etype, no_of_nodes_per_element))

            for tag in self._tag_names:
                # write values for each element type
                if self._tags[tag]['default'] is None:
                    default = np.nan
                else:
                    default = self._tags[tag]['default']
                tag_values = el_df_by_shape[tag].fillna(default).values
                tag_group_pointers[tag].create_dataset(etype, data=tag_values.astype(self._tags[tag]['dtype']),
                                                       shape=tag_values.shape)

        row_arr = np.array(rows, dtype=dt)
        write_hdf5_table(hdf_fp, row_arr, "/mesh/elementids")


def write_xdmf_mesh_from_hdf5(xdmffilename, hdffilename, meshroot):
    # get Infos from hdf5
    def get_infos_from_hdf(fp):
        rno_of_nodes = fp[meshroot + '/nodes'].shape[0]
        topologynode = fp[meshroot + '/topology']
        relementsshape_by_etype = dict()
        for etype in topologynode.keys():
            relementsshape_by_etype.update({etype: topologynode[etype].shape})
        tagsnode = fp[meshroot + '/tags']
        rtags = [rtag for rtag in tagsnode.keys()]
        return rno_of_nodes, relementsshape_by_etype, rtags

    if isinstance(hdffilename, str):
        with h5py.File(hdffilename, 'r') as hdffp:
            no_of_nodes, elementsshape_by_etype, tags = get_infos_from_hdf(hdffp)
    elif isinstance(hdffilename, h5py._hl.files.File):
        no_of_nodes, elementsshape_by_etype, tags = get_infos_from_hdf(hdffilename)
    else:
        raise ValueError('hdffilename must be either a valid filename or a h5py.File object')

    relative_hdf5_path = splitext(basename(hdffilename))[0]
    with open(xdmffilename, 'wb') as xdmf_fp:
        root = ET.Element('Xdmf', {'Version': '3.0'})
        domain = ET.SubElement(root, 'Domain')  # , {'Type': 'Uniform'})
        collection = ET.SubElement(domain, 'Grid', {'GridType': 'Collection', 'CollectionType': 'Spatial'})
        # no_of_nodes = ppmeshobj['nodes'].count()
        # el_df = ppmeshobj['elements']
        for etype in elementsshape_by_etype.keys():
            no_of_elements_of_current_etype = elementsshape_by_etype[etype][0]
            no_of_nodes_per_element = elementsshape_by_etype[etype][1]
            grid = ET.SubElement(collection, 'Grid', {'Name': 'mesh', 'GridType': 'Uniform'})
            topology = ET.SubElement(grid, 'Topology',
                                     {'NumberOfElements': str(no_of_elements_of_current_etype),
                                      'TopologyType': XDMFDICT[etype]['xdmf_name']})
            elementdata = ET.SubElement(topology, 'DataItem',
                                        {'Dimensions': '{} {}'.format(no_of_elements_of_current_etype,
                                                                      no_of_nodes_per_element),
                                         'Format': 'HDF', 'NumberType': 'Int'})
            elementdata.text = relative_hdf5_path + '.hdf5:/mesh/topology/' + etype
            geometry = ET.SubElement(grid, 'Geometry', {'GeometryType': 'XYZ'})
            nodedata = ET.SubElement(geometry, 'DataItem', {'Dimensions': '{} {}'.format(
                no_of_nodes, 3), 'Format': 'HDF', 'NumberType': 'Float'})
            nodedata.text = relative_hdf5_path + '.hdf5:/mesh/nodes'
            for tag in tags:
                attribute = ET.SubElement(grid, 'Attribute', {'Name': tag, 'Center': 'Cell',
                                                              'AttributeType': 'Scalar'})
                attribute_data = ET.SubElement(attribute, 'DataItem',
                                               {'Dimensions': '{} 1'.format(no_of_elements_of_current_etype),
                                                'Format': 'HDF'})
                attribute_data.text = relative_hdf5_path + '.hdf5:/mesh/tags/{}/{}'.format(tag, etype)
        insert_line_breaks_in_xml(root)
        tree = ET.ElementTree(root)
        tree.write(xdmf_fp, 'UTF-8', True)
