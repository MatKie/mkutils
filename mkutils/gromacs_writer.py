from .mie import mie
import json


class FFWriter(object):
    """
    Writes Forcefield information for GROMACS from a json file with all
    the necessary data. 

    Parameters
    ----------
    param_file : str, path
        File path to the .json file holding the parameter values.
        Basically a dict like this: 
        {'atomtypes' : {'Atom1': {'epsilon': float (in Kelvin), 
        'sigma': float (in nm), 'l_r': float, 'l_a': float, 
        'MW': float }}, 
        'crossint': {frozenset('Atom1', 'Atom2'): float (k_ij)}
        }
    
    Methods
    -------
    add_atomtype():

    add_crossint():
    """

    def __init__(self, param_file):
        self.param_file = param_file

    def add_atomtype(self, name, l_r, l_a, eps, sig, MW, update=False):
        _dict = {name: {"l_r": l_r, "l_a": l_a, "eps": eps, "sig": sig, "MW": MW}}
        _ff_dict = self._open_json()
        if name in _ff_dict.get("atomtypes").keys():
            if not update:
                raise ValueError(
                    "Atomtype already in forcefield. Explicitly \
                    set update flag! "
                )
            print("Updating {:s} atom type".format(name))

        _ff_dict.get("atomtypes").update(_dict)

        self._close_json(_ff_dict)

    def add_crossint(self, name1, name2, kij, update=False):
        _fr_set = frozenset([name1, name2])
        _dict = {_fr_set: kij}

        _ff_dict = self._open_json()
        for name in [name1, name2]:
            if name not in _ff_dict.get("atomtypes").keys():
                raise ValueError("{:s} not an atomtype!".format(name))

        if _fr_set in _ff_dict.get("crossint").keys():
            if not update:
                raise ValueError(
                    "Crossint already in forcefield. Explicitly \
                    set update flag! "
                )
            print("Updating {:s} - {:s} cross interaction.".format(name1, name2))

        _ff_dict.get("crossint").update(_dict)

        self._close_json(_ff_dict)

    def _open_json(self):
        with open(self.param_file, "r") as f:
            ff_dict = json.load(f)
        ff_dict["crossint"] = {
            frozenset(key.split(":")): value
            for key, value in ff_dict.get("crossint").items()
        }
        return ff_dict

    def _close_json(self, ff_dict):
        ff_dict["crossint"] = {
            ":".join(key): value for key, value in ff_dict.get("crossint").items()
        }
        with open(self.param_file, "w") as f:
            json.dump(ff_dict, f)
