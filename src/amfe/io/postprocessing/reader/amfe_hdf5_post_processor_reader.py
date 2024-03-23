#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

import numpy as np
import h5py

from amfe.io import check_filename_or_filepointer, MeshEntityType, PostProcessDataType
from amfe.io.mesh import Hdf5MeshReader
from amfe.io.postprocessing import AmfePostprocessMeshConverter

from ..base import PostProcessorReader


__all__ = ['AmfeHdf5PostProcessorReader']


class AmfeHdf5PostProcessorReader(PostProcessorReader):
    def __init__(self, hdf5filename, meshrootpath='/mesh', resultsrootpath='/results', write_only=None):
        """
        Reader for Amfe HDF5 Postprocessor Files

        Parameters
        ----------
        hdf5filename : h5py.File or str
            path to hdf5 file that shall be read
        meshrootpath : str
            hdf5-path to the mesh information (should contain nodes array and topology folder with arrays
        resultsrootpath : str
            hdf5-path to results information
        write_only : tuple or list
            tuple or list with strings describing the names of the fields that shall be parsed.
            If None (default) all available fields will be parsed
        """
        super().__init__()
        self._filename = hdf5filename
        self._meshrootpath = meshrootpath
        self._resultsrootpath = resultsrootpath
        self._write_only = write_only

    def parse(self, builder):
        """
        Build the postprocessing data with given builder

        Parameters
        ----------
        builder : amfe.io.PostProcessorWriter
            builder with which the output is built

        Returns
        -------
        None
        """
        return self._parse(self._filename, builder)

    @check_filename_or_filepointer(h5py._hl.files.File, h5py.File, 1, writeable=False)
    def _parse(self, hdf5fp, builder):
        """

        Parameters
        ----------
        hdf5fp : h5py.File
            h5py.File object that is parsed
        builder : amfe.io.postprocessor.PostProcessorWriter
            builder that is used for building the desired object

        Returns
        -------
        None
        """
        # -- Mesh Conversion --
        meshreader = Hdf5MeshReader(hdf5fp, self._meshrootpath)
        converter = AmfePostprocessMeshConverter()
        meshreader.parse(converter)
        mesh = converter.return_mesh()
        # -- End Mesh Conversion --

        # Retrieve the group in HDF5-file containing the results data
        resultsgroup = hdf5fp[self._resultsrootpath]

        # If write_only is None: Build a list with all fields that are contained in the file
        if self._write_only is None:
            self._write_only = list(resultsgroup.keys())
            if 'timesteps' in self._write_only:
                self._write_only.remove('timesteps')

        # Retrieve the timesteps belonging to the results data
        timesteps = hdf5fp[self._resultsrootpath + '/timesteps'][...]
        for fieldname in self._write_only:
            # Get the fieldnode
            fieldnode = hdf5fp[self._resultsrootpath + '/' + fieldname]


            # Element field information are stored in a folder, node information directly in an array.
            # Thus check the fieldnode instance if it is a Group (-> element info) or an array (-> probably node info)
            is_dataset = isinstance(fieldnode, h5py.Dataset)
            if is_dataset:
                mesh_entity_type = MeshEntityType.NODE
                field_type = PostProcessDataType[fieldnode.attrs.get('data_type')]
            else:
                mesh_entity_type = MeshEntityType.ELEMENT
                # Field type still unknown because it is stored in the arrays inside the Group
                field_type = None

            # -- Build data and index information --
            if mesh_entity_type == MeshEntityType.NODE:
                # get indices and data
                data = fieldnode[...]
                index = mesh['nodes'].index.values

            elif mesh_entity_type == MeshEntityType.ELEMENT:
                # Put all etype_data together:
                no_of_timesteps = len(timesteps)
                el_df = mesh['elements']
                index = el_df.index.values
                no_of_elements = len(el_df.index)
                data = np.empty((no_of_elements, no_of_timesteps))
                data[:] = np.nan
                # For each element type is a separate array
                for etype_node in fieldnode.keys():
                    if isinstance(fieldnode[etype_node], h5py.Dataset):
                        # Now we are inside the group and can get field_type information.
                        # It is assumed that it is consistent and equal in all arrays
                        field_type = PostProcessDataType[fieldnode[etype_node].attrs.get('data_type')]
                        etype = etype_node
                        etype_array = fieldnode[etype_node][...]
                        etype_indices = el_df[el_df['shape'] == etype].index.values
                        iloc = el_df.loc[etype_indices, 'iconnectivity'].values
                        data[iloc, :] = etype_array
                isnotnan = ~np.isnan(data[:, 0])
                index = index[isnotnan]
                data = data[isnotnan]
            else:
                raise NotImplementedError('The given MeshEntityType {} can not be parsed'.format(mesh_entity_type.name))
            # -- End building data and index information --
            builder.write_field(fieldname, field_type, timesteps, data, index, mesh_entity_type)
