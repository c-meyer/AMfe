#
# Copyright (c) 2018 TECHNICAL UNIVERSITY OF MUNICH, DEPARTMENT OF MECHANICAL ENGINEERING, CHAIR OF APPLIED MECHANICS,
# BOLTZMANNSTRASSE 15, 85748 GARCHING/MUNICH, GERMANY, RIXEN@TUM.DE.
#
# Distributed under 3-Clause BSD license. See LICENSE file for more information.
#
import pandas as pd
from copy import deepcopy

from amfe.component.constants import ELEPROTOTYPEHELPERLIST
from .structural_neumann import *


class NeumannManager:
    """
    Manager-class to create, assign and manipulate several different Neumann conditions
    """
    BOUNDARYELEMENTSHAPES = [element[0] for element in ELEPROTOTYPEHELPERLIST if element[2] is not None]
    
    def __init__(self):
        # Dataframe for reconstructing applied conditions
        self._neumann_df = pd.DataFrame(columns=['name', 'tag', 'property_names', 'neumann_obj'])
        
        # Dataframe containing element_objects and their position in connectivity array and their
        # foreign key to the _neumann_df they belong to
        self._neumann_obj_df = pd.DataFrame(columns=['neumann_obj', 'fk_mesh', 'fk_neumann_df', 'fk_mapping'])
        self._neumann_obj_df['fk_neumann_df'] = self._neumann_obj_df['fk_neumann_df'].astype(np.intp)
        self._neumann_obj_df['fk_mesh'] = self._neumann_obj_df['fk_mesh'].astype(np.intp)
        self._neumann_obj_df['fk_mapping'] = self._neumann_obj_df['fk_mapping'].astype(np.intp)

    def __str__(self):
        """
        Returns information about the Neumann-Manager when using print(instance)

        Returns
        -------
        info_of_Neumann-Manager: string
        #     no of condition definitions, list of applied conditions with linked objects, Address in RAM
        """
        return "--- Info about Neumann-Manager ---\nNo of conditions definitions: {0:,>}\n" \
               "List of applied conditions:\n{1:,>}\nAddress in RAM: {2}\n"\
               .format(self.no_of_condition_definitions, self._neumann_df.to_string(columns=['name', 'property_names']),
                       id(self))

    @property
    def no_of_condition_definitions(self):
        """
        Returns the number of neumann conditions that have different names

        Returns
        -------
        n: int
            number of neumann condidtions that have different names
        """
        return len(self._neumann_df['name'].unique())

    @property
    def fields(self):
        fields_list = list(set([field for ele_obj in self._neumann_obj_df['neumann_obj'].unique()
                                for field in ele_obj.fields()]))
        fields_list.sort()
        return fields_list

    def assign_neumann_by_eleids(self, neumannobj, eleidxes, ele_shapes, property_names, tag, name):
        dfindex = self._neumann_df.index.max() + 1
        if pd.isnull(dfindex):
            dfindex = 0

        neumann_prototypes = {element_shape: deepcopy(neumannobj) for element_shape in self.BOUNDARYELEMENTSHAPES}
        for element_shape, neumann_obj in neumann_prototypes.items():
            neumann_obj.set_element(element_shape)

        # Create pointer for each element
        neumann_objects = np.array([neumann_prototypes[ele_shape] for ele_shape in ele_shapes])
        # Create new rows for neumann_obj_df
        df = pd.DataFrame(
            {'neumann_obj': neumann_objects, 'fk_mesh': eleidxes,
             'fk_neumann_df': np.ones(len(neumann_objects), dtype=np.intp) * dfindex,
             'fk_mapping': -1*np.ones(len(neumann_objects), dtype=np.intp)}
        )
        self._neumann_obj_df = pd.concat([self._neumann_obj_df, df], ignore_index=True, verify_integrity=True)

        # Create entry for neumann_df describing the whole b.c.
        df_data = {'name': name, 'tag': tag, 'property_names': [property_names], 'neumann_obj': neumannobj}
        self._neumann_df = pd.concat([self._neumann_df, pd.DataFrame(df_data, index=[dfindex])], sort=True)

    @property
    def el_df(self):
        return self._neumann_obj_df

    def write_mapping_key(self, fk, local_id):
        """
        Write a foreign key info to a mapping element

        Parameters
        ----------
        fk : int
            foreign key to a mapping class that contains mapping info
        local_id : int
            index of the neumann obj that shall get the new mapping info

        Returns
        -------
        None

        """
        self._neumann_obj_df.loc[local_id, 'fk_mapping'] = fk

    def get_ele_obj_fk_mesh_and_fk_mapping(self):
        """
        Returns neumann objects, their foreign keys to mesh elements and their foreign keys to mapping infos

        Returns
        -------
        neumann_obj : iterable
            iterable containing neumann_objects
        fk_mesh : iterable
            iterable containing the mesh foreign keys to elements
        fk_mapping : iterable
            iterable containing mapping foreign keys to mapping information of a mapping object
        """
        values = self._neumann_obj_df[['neumann_obj', 'fk_mesh', 'fk_mapping']].values
        return values[:, 0], values[:, 1], values[:, 2]

    @staticmethod
    def create_fixed_direction_neumann(direction, time_func=lambda t: 1):
        direction = np.array(direction, dtype=np.float64)
        return FixedDirectionNeumann(direction, time_func)

    @staticmethod
    def create_normal_following_neumann(time_func=lambda t: 1):
        return NormalFollowingNeumann(time_func)

    @staticmethod
    def create_projected_area_neumann(direction, time_func=lambda t: 1):
        direction = np.array(direction, dtype=np.float64)
        return ProjectedAreaNeumann(direction, time_func)
