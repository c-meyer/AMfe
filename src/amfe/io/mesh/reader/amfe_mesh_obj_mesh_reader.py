#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#

"""
AMfe mesh object reader for I/O module.
"""

from amfe.io.mesh.base import MeshReader

import pandas as pd
import numpy as np

__all__ = [
    'AmfeMeshObjMeshReader'
    ]


class AmfeMeshObjMeshReader(MeshReader):
    """
    Reader for AMfe mesh objects.
    """

    def __init__(self, meshobj=None):
        super().__init__()
        self._meshobj = meshobj
        return

    def parse(self, builder, verbose=False):
        """

        Parameters
        ----------
        builder : MeshConverter
            MeshConverter object that builds the mesh
        verbose : bool
            If True, verbose mode is activated

        Returns
        -------
        None
        """
        # build dimension
        builder.build_mesh_dimension(self._meshobj.dimension)
        builder.build_no_of_nodes(self._meshobj.no_of_nodes)
        builder.build_no_of_elements(self._meshobj.no_of_elements + self._meshobj.no_of_boundary_elements)
        # build nodes
        if self._meshobj.dimension == 2:
            for index, row in self._meshobj.nodes_df.iterrows():
                builder.build_node(index, row['x'], row['y'], 0.0)
        else:
            for index, row in self._meshobj.nodes_df.iterrows():
                builder.build_node(index, row['x'], row['y'], row['z'])

        # build elements
        for elementid, element in self._meshobj.el_df.iterrows():
            etype = element['shape']
            connectivity = list(element['connectivity'])
            builder.build_element(elementid, etype, connectivity)
        # build groups
        for group in self._meshobj.groups:
            builder.build_group(group,
                                self._meshobj.groups[group]['nodes'],
                                self._meshobj.groups[group]['elements'])
        # build tags
        no_tags = ['shape', 'connectivity', 'is_boundary']
        for column in self._meshobj.el_df.columns:
            if column in no_tags:
                continue
            dfcol = self._meshobj.el_df[column]
            uniquevalues = dfcol.unique()
            tagsdict = dict()
            default = None
            dtype = object
            for d in [int, np.intp, np.int32, np.int64, pd.Int64Dtype(), pd.Int32Dtype()]:
                if d == dfcol.dtype:
                    dtype = int
            if dfcol.dtype in [float, np.float32, np.float64]:
                dtype = float
            for uniquevalue in uniquevalues:
                if pd.isna(uniquevalue):
                    default = None
                    continue
                tagsdict.update({uniquevalue: dfcol[dfcol == uniquevalue].index.tolist()})
            builder.build_tag(column, tagsdict, dtype, default)
        return
