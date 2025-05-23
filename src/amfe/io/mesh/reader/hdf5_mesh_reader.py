#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
Hdf5 mesh reader for I/O module.
"""

import numpy as np
import h5py
import pandas as pd

from amfe.io.mesh.base import MeshReader
from amfe.io.tools import check_filename_or_filepointer


__all__ = [
    'Hdf5MeshReader'
]


class Hdf5MeshReader(MeshReader):
    """
    Reader Hdf5 Mesh Files written by AMfe

    Attributes
    ----------
    _filename : str or h5py.File
        Filename or h5py.File object where the mesh is read from
    _meshrootpath : str
        Path where the mesh is located within the HDF structure e.g. "/mesh"

    """

    def __init__(self, filename, meshrootpath):
        """
        Constructor of Hdf5MeshReader

        Parameters
        ----------
        filename : str or h5py.File
            Filename or h5py.File object where the mesh is read from
        meshrootpath : str
            Path where the mesh is located within the HDF structure e.g. "/mesh"
            Must start with "/".
        """
        super().__init__()

        # ----------------- Set properties -----------------------------

        # 1. filename:
        # Type check filename can be a str or a File object
        if isinstance(filename, (h5py._hl.files.File, str)):
            self._filename = filename

        # 2. meshrootpath
        # Type check meshrootpath
        if isinstance(meshrootpath, str):
            # Mesh rootpath must start with a slash and end without a slash:
            if meshrootpath.startswith('/'):
                if meshrootpath.endswith('/'):
                    meshrootpath = meshrootpath[:-1]
                self._meshrootpath = meshrootpath
            else:
                raise ValueError('meshrootpath must start with "/", given {}'.format(meshrootpath))
        else:
            raise ValueError('meshrootpath must be a string')

        # End setting properties, return
        return

    def parse(self, builder):
        """
        Parse Hdf5 file to object specified by the builder (MeshConverter object).

        Parameters
        ----------
        builder : MeshConverter
            MeshConverter object that is used for conversion

        Returns
        -------
        None
        """
        return self._parse(self._filename, builder)

    @check_filename_or_filepointer(h5py._hl.files.File, h5py.File, 1, writeable=False)
    def _parse(self, infile, builder):
        """
        Private function for parsing the mesh with a MeshConverter

        Parameters
        ----------
        infile : h5py.File
            File object where the mesh is read from
        builder : amfe.io.MeshConverter
            MeshConverter object that is used to parse the mesh
        """

        # Read nodes array from file
        nodes = infile[self._meshrootpath + '/nodes'][...]
        # nodes array has shape no_of_nodes x 3, get number of nodes
        no_of_nodes = nodes.shape[0]
        # nodeids are stored in nodeids array
        nodeids = infile[self._meshrootpath + '/nodeids'][...]

        # As the number of nodes are known build_no_of_nodes if preallocation is available in builder
        builder.build_no_of_nodes(no_of_nodes)
        # for each node build it
        for nodeid, node_coordinates in zip(nodeids, nodes):
            builder.build_node(nodeid, node_coordinates[0], node_coordinates[1], node_coordinates[2])

        # get the elementids table
        # the table has 3 columns:
        #   index: describes the index of the element
        #   etype: describes the shape of the element
        #   etype_index: describes the local index location of the element in the array belonging to its shape
        #   (for each shape is a separate array
        elementids_table = infile[self._meshrootpath + '/elementids'][...]
        no_of_elements = elementids_table.shape[0]
        # As the no_of_elements is known build them
        builder.build_no_of_elements(no_of_elements)
        # The arrays of the iconnectivites are stored within the topology group.
        # The name of the arrays (property _v_name) are the same as the internal shape names
        # get all elementshapes that
        elementshapes = infile[self._meshrootpath + '/topology'].keys()
        # create empty iconncectivity dict
        iconnectivity_dict = dict()
        # for each element shape get the array where the iconnectivities are stored and put them into the dictionary
        # key = elementshape, value = iconnectivity array
        for etype in elementshapes:
            arr = infile[self._meshrootpath + '/topology/{}'.format(etype)][...]
            iconnectivity_dict.update({etype: arr})

        # iterate over all elements (by iterating over all elementids) and build each element
        for row in elementids_table:
            # Get element id, shape end local index in its array from the table
            eid = row[2]
            etype = row[0].decode('UTF-8')
            etype_index = row[1]
            # Get the iconnectivity array of current etype from the dict
            iconnectivity = iconnectivity_dict[etype][etype_index, :]
            # map the iconnectivity to the real connectivity (nodeids instead of iloc of nodes array
            connectivity = [nodeids[iloc] for iloc in iconnectivity]
            # call buildd_element
            builder.build_element(eid, etype, connectivity)

        tag_group = infile[self._meshrootpath + '/tags']
        for tag in tag_group.keys():
            tagname = tag
            current_tag_dict = dict()
            for etype_node in tag_group[tag].keys():
                etype_name = etype_node
                etype_arr = tag_group[tag][etype_node][...]
                etype_arr = etype_arr[~np.isnan(etype_arr)]
                etype_unique_tag_values = np.unique(etype_arr)
                for tag_value in etype_unique_tag_values:
                    ilocelementids = np.where(etype_arr == tag_value)[0]
                    tag_elementids = []
                    for ilocelementid in ilocelementids:
                        # tag_elementids.append([x['index'] for x in elementids.where("(etype == {}) & (etype_index == {})".format(etype_name.encode('UTF-8'), ilocelementid))])

                        df = pd.DataFrame(elementids_table)
                        tag_elementids.extend(list(df.loc[(df["etype"] == etype_name.encode("UTF-8")) & (df["etype_index"] == ilocelementid), "index"].values))

                    if tag_value in current_tag_dict:
                        current_tag_dict[tag_value] = current_tag_dict[tag_value].append(tag_elementids)
                    else:
                        current_tag_dict.update({tag_value: tag_elementids})
            for tag_value in current_tag_dict:
                current_tag_dict[tag_value] = np.unique(np.array([current_tag_dict[tag_value]])).tolist()
            dtype = None
            for shape in tag_group[tag].keys():
                if np.issubdtype(tag_group[tag][shape].dtype, np.integer):
                    if dtype is None:
                        dtype = int
                    elif dtype == int:
                        pass
                    else:
                        dtype = object
                elif np.issubdtype(tag_group[tag][shape].dtype, np.floating):
                    if dtype is None:
                        dtype = float
                    elif dtype == float:
                        pass
                    else:
                        dtype = object
            builder.build_tag(tagname, current_tag_dict, dtype)
        return
