from itertools import permutations
from copy import deepcopy

import torch


class Effect:
    def __init__(self, z_i, r_i, z_f, r_f, cfg):
        """
        Compute the effect of a transition.
        z_i, z_f: Lists of object IDs (initial and final)
        r_i, r_f: Relation tensors (initial and final)
        cfg:      yaml config dict (must contain cfg["model"]["symbol_size"])
        """
        self.n_rel = cfg["model"]["symbol_size"]
        n_pddl_rel = self.n_rel + 1

        n_f = len(z_f)  # number of final objects
        n_i = len(z_i)  # number of initial objects

        z_i, z_f = self.update_z(z_i, z_f)
        r_i, r_f = self.update_r(r_i, r_f, n_f)

        obj_diff_idx = torch.where(z_i != z_f)[0].unique()  # 0 is the row id
        obj_dict = {}
        for idx in obj_diff_idx:
            obj_dict[idx.item()] = tuple((z_f[idx].int() - z_i[idx].int()).tolist())
        self.z_eff = obj_dict

        relations = [{} for _ in range(n_pddl_rel)]

        ri_new = [{} for _ in range(n_pddl_rel)]
        rf_new = [{} for _ in range(n_pddl_rel)]

        cnt = 0
        for j in range(1, n_f):
            for i in range(j):
                for k in range(n_pddl_rel):
                    ri_new[k][(i, j)] = int(r_i[cnt, k].item())
                    rf_new[k][(i, j)] = int(r_f[cnt, k].item())
                cnt += 1

        for i in range(n_f):
            for j in range(i + 1, n_f):
                for k in range(n_pddl_rel):
                    val_i = ri_new[k][(i, j)]
                    val_f = rf_new[k][(i, j)]
                    if val_i != val_f:
                        relations[k][(i, j)] = int(val_f) - int(val_i)
        self.r_eff = relations

    def update_z(self, z_i, z_f):
        """
        Adds an extra existence bit: 1 for all final objects, 0 for
        objects that did not exist in the initial state.
        """
        num_existing = z_i.size(0)
        num_total = z_f.size(0)

        indicator = torch.ones((num_total, 1),
                               device=z_f.device,
                               dtype=z_f.dtype)

        updated_zf = torch.cat((z_f, indicator), dim=1)

        indicator[num_existing:, 0] = 0

        updated_zi = torch.cat((z_f, indicator), dim=1)

        return updated_zi, updated_zf

    def update_r(self, r_i, r_f, n):
        expected_pairs = (n * (n - 1)) // 2

        if isinstance(r_i, torch.Tensor) and r_i.numel() > 0:
            flat_rel = r_i.view(-1, self.n_rel)
            current_len = flat_rel.shape[0]
            existence_column = torch.ones((current_len, 1), device=r_i.device)
            flat_rel = torch.cat([flat_rel, existence_column], dim=1)
        else:
            flat_rel = torch.empty((0, self.n_rel + 1), device='cuda:0')
            current_len = 0

        needed_extra = max(0, expected_pairs - current_len)

        if needed_extra > 0:
            padding = torch.zeros((needed_extra, self.n_rel + 1), device=flat_rel.device)
            updated_ri = torch.cat([flat_rel, padding], dim=0)
        else:
            updated_ri = flat_rel

        flat_rel_rf = r_f.view(-1, self.n_rel)
        current_len = flat_rel_rf.shape[0]
        existence_column = torch.ones((current_len, 1), device=r_i.device)
        updated_rf = torch.cat([flat_rel_rf, existence_column], dim=1)

        return updated_ri, updated_rf

    def substitute(self, delta):
        z_eff = {}
        for k, v in self.z_eff.items():
            z_eff[delta[k]] = v
        r_eff = []
        for rel_dict in self.r_eff:
            new_rel_dict = {}
            for k, v in rel_dict.items():
                key = tuple(delta[k_i] for k_i in k)
                new_rel_dict[key] = v
            r_eff.append(new_rel_dict)
        new_effect = deepcopy(self)
        new_effect.z_eff = z_eff
        new_effect.r_eff = r_eff
        return new_effect

    def __hash__(self):
        repr = str(sorted(tuple(self.z_eff.items()))) + \
               str(tuple(sorted(tuple(k.items())) for k in self.r_eff))
        return hash(repr)

    def __eq__(self, other):
        return (self.z_eff == other.z_eff and
                self.r_eff == other.r_eff)

    def __repr__(self):
        return f"Object Effects: {self.z_eff}\nRelation Effects: {self.r_eff}"