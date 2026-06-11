"""
# State 
# Includes parameter, precondition and action
# parameters are ojects as o1, o2, o3..
# precondition have 4 components
# 1. object symbols z0-z1-z2-z3 
# 2. object existence z4
# 3. relation symbols r0-r1-r2-r3-r4
# 4. relation existence r5
"""

"""
Initial condition:

symbols of objects are defined (z0-3)
all z4 is 0
all r0-5 is 0

-------

New steps:

after each step add objects z4=1
new relations are added 
r5 for those relations are 1

--------

preconditions include:

symbols of existing and to be added object
all new object related realtions r0-5 as 0 
existing object o4 as 1
new objects o4 as 0

"""
from copy import deepcopy
import numpy as np
import torch


class State:
    def __init__(self, obj_tensor, existing_obj_tensor, rel_tensor, cfg, act_tensor=None, act_symbol=None):
        """
        obj_tensor: List of discrete IDs(size=1) or Symbols representing objects
                    [N_obj, OBJ_SYM_SIZE] - Object symbols (z)
        rel_tensor: [REL_SYM_SIZE, N_obj, N_obj] - Relational symbols (r)
        cfg:        yaml config dict (must contain cfg["model"]["symbol_size"])
        act_symbol: int - A discrete ID/or symbols representing the action ("1010")
        """
        self.n_obj = obj_tensor.shape[0]
        self.n_rel = cfg["model"]["symbol_size"]

        rel_tensor = self.update_rel_tensor(rel_tensor, self.n_obj)
        n_pddl_rel = self.n_rel + 1
        self.n_pddl_rel = n_pddl_rel
        relations = [{} for _ in range(n_pddl_rel)]

        obj_tensor = self.update_obj_tensor(obj_tensor, existing_obj_tensor)
        obj_dict = {}

        cnt = 0
        for j in range(1, self.n_obj):
            for i in range(j):
                for k in range(n_pddl_rel):
                    relations[k][(i, j)] = int(rel_tensor[cnt, k].item())
                cnt += 1

        for i in range(self.n_obj):
            obj_dict[i] = tuple(obj_tensor[i].int().tolist())

        """if act_symbol is not None:
            action = act_symbol
        else:
            action = tuple(act_tensor.int().tolist())"""
        if isinstance(act_symbol, torch.Tensor):
            action = tuple(act_symbol.squeeze().int().tolist())
        else:
            action = tuple(act_symbol)

        self.obj_dict = obj_dict
        self.relations = relations
        self.action = action

    def update_rel_tensor(self, rel_tensor, n):
        """
        Adds extra bit to existing relations as 1.
        Adds sym_size + 1 bit relations with new objects all 0.
        """
        expected_pairs = (n * (n - 1)) // 2

        if isinstance(rel_tensor, torch.Tensor) and rel_tensor.numel() > 0:
            flat_rel = rel_tensor.view(-1, self.n_rel)
            current_len = flat_rel.shape[0]
            existence_column = torch.ones((current_len, 1), device=rel_tensor.device)
            flat_rel = torch.cat([flat_rel, existence_column], dim=1)
        else:
            flat_rel = torch.empty((0, self.n_rel + 1), device='cuda:0')
            current_len = 0

        needed_extra = max(0, expected_pairs - current_len)

        if needed_extra > 0:
            padding = torch.zeros((needed_extra, self.n_rel + 1), device=flat_rel.device)
            full_rel_tensor = torch.cat([flat_rel, padding], dim=0)
        else:
            full_rel_tensor = flat_rel

        return full_rel_tensor

    def update_obj_tensor(self, obj_tensor, ex_obj_tensors):
        """
        Adds an extra existence bit: 1 for already-existing objects, 0 for new ones.
        """
        num_existing = ex_obj_tensors.size(0)
        num_total = obj_tensor.size(0)

        indicator = torch.zeros((num_total, 1),
                                device=obj_tensor.device,
                                dtype=obj_tensor.dtype)
        indicator[:num_existing, 0] = 1

        updated_tensor = torch.cat((obj_tensor, indicator), dim=1)
        return updated_tensor

    def get_params(self):
        params = []
        for key in self.obj_dict:
            params.append(key)
        for k, rel_dict in enumerate(self.relations):
            for (key1, key2) in rel_dict:
                params.append(key1)
                params.append(key2)
        return tuple(np.unique(params).tolist())

    def __eq__(self, other):
        return (self.obj_dict == other.obj_dict and
                self.action == other.action and
                self.relations == other.relations)

    def __hash__(self):
        repr = str(sorted(tuple(self.obj_dict.items()))) + \
               str(tuple(sorted(tuple(k.items())) for k in self.relations)) + \
               str(self.action)
        return hash(repr)

    def substitute(self, delta):
        obj_dict = {}
        relations = [{} for _ in range(self.n_rel)]
        for idx in delta:
            key = delta[idx]
            obj_dict[key] = self.obj_dict[idx]
            for idx2 in delta:
                if idx2 > idx:
                    key2 = delta[idx2]
                    for k in range(self.n_rel):
                        relations[k][(key, key2)] = self.relations[k][idx, idx2]

        action = self.action
        new_state = deepcopy(self)
        new_state.obj_dict = obj_dict
        new_state.relations = relations
        new_state.action = action
        return new_state