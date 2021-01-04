from .mie import mie
import json
import os


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
        'crossints': {frozenset('Atom1', 'Atom2'): float (k_ij)}
        }
    
    Methods
    -------
    add_atomtype():

    add_crossint():
    """

    def __init__(self, param_file):
        self.param_file = param_file

    def add_atomtype(self, name, l_r, l_a, eps, sig, MW, at_num, update=False):
        _dict = {
            name: {
                "l_r": l_r,
                "l_a": l_a,
                "eps": eps,
                "sig": sig,
                "MW": MW,
                "at_num": at_num,
            }
        }
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

    def add_crossint(self, name1, name2, k=0.0, eps_mix=False, update=False):
        _fr_set = frozenset([name1, name2])
        _dict = {_fr_set: {"k": k, "eps_mix": eps_mix}}

        _ff_dict = self._open_json()
        for name in [name1, name2]:
            if name not in _ff_dict.get("atomtypes").keys():
                raise ValueError("{:s} not an atomtype!".format(name))

        if _fr_set in _ff_dict.get("crossints").keys():
            if not update:
                raise ValueError(
                    "Crossint already in forcefield. Explicitly \
                    set update flag! "
                )
            print("Updating {:s} - {:s} cross interaction.".format(name1, name2))

        _ff_dict.get("crossints").update(_dict)

        self._close_json(_ff_dict)

    def write_forcefield(self, outfile="ffnonbonded.itp"):
        atomtypes = self.print_atomtypes()
        crossints = self.print_crossints()
        with open(outfile, "w") as f:
            f.write(atomtypes)
            f.write("\n\n")
            f.write(crossints)

    def print_atomtypes(self):
        _fl = "[ atomtypes ]"
        _sl = ";   {:<16s}{:<8s}{:<12s}{:<8s}{:<8s}{:<16s}{:<16s}{:<s}".format(
            "name", "at.num", "mass", "charge", "ptype", "V(C6)", "W(Cm)", "Ref"
        )

        atomtypes = self._open_json().get("atomtypes")
        return_string = "\n".join([_fl, _sl])
        for _name, atomtype in atomtypes.items():
            _at_num = atomtype.get("at_num")
            _mass = atomtype.get("MW")
            _args = tuple(atomtype.get(key) for key in ["l_r", "l_a", "eps", "sig"])
            _mie = mie(*_args)
            C_a = _mie.get_C_attr()
            C_r = _mie.get_C_rep()

            _l = "    {:<16s}{:<8d}{:<12.5}{:<8.4f}{:<8s}{:<16.6e}{:<16.6e}{:<s}".format(
                _name, _at_num, _mass, 0.0, "A", C_a, C_r, ";"
            )
            return_string = "\n".join([return_string, _l])

        return return_string

    def write_tables(self, shift=True, cutoff=2.0, double_prec=False):
        ff_dict = self._open_json()
        atomtypes = ff_dict.get("atomtypes")
        keys = list(atomtypes.keys())

        if not os.path.isdir("tables"):
            os.makedirs("tables")
        os.chdir("tables")
        for i, type_i in enumerate(keys):
            for type_j in keys:
                atomtype_i = atomtypes.get(type_i)
                atomtype_j = atomtypes.get(type_j)
                _argsi = tuple(
                    atomtype_i.get(key) for key in ["l_r", "l_a", "eps", "sig"]
                )
                _argsj = tuple(
                    atomtype_j.get(key) for key in ["l_r", "l_a", "eps", "sig"]
                )
                _mie_i = mie(*_argsi)
                _mie_j = mie(*_argsj)

                mix = mie.mix(_mie_i, _mie_j, rc=cutoff, shift=shift)

                mix.write_table(
                    names=(type_i, type_j),
                    double_prec=double_prec,
                    cutoff=cutoff,
                    shift=shift,
                )
        os.chdir("..")

    def print_crossints(self):
        _fl = "[ nonbond_params ]"
        _sl = ";   {:<16s}{:<16s}{:<8s}{:<16s}{:<16s}{:<s}".format(
            "i", "j", "func", "V(C6)", "W(Cm)", "Ref"
        )

        return_string = "\n".join([_fl, _sl])

        ff_dict = self._open_json()
        atomtypes = ff_dict.get("atomtypes")
        crossints = ff_dict.get("crossints")
        for _namei, atomtypei in atomtypes.items():
            for _namej, atomtypej in atomtypes.items():

                _argsi = tuple(
                    atomtypei.get(key) for key in ["l_r", "l_a", "eps", "sig"]
                )
                _argsj = tuple(
                    atomtypej.get(key) for key in ["l_r", "l_a", "eps", "sig"]
                )
                _mie_i = mie(*_argsi)
                _mie_j = mie(*_argsj)
                crossint_id = frozenset([_namei, _namej])
                if crossint_id in crossints.keys():
                    _kwargs_mix = crossints.get(crossint_id)
                else:
                    _kwargs_mix = {"k": 0.0, "eps_mix": False}

                mix = mie.mix(_mie_i, _mie_j, rule="SAFT", **_kwargs_mix)
                C_a = mix.get_C_attr()
                C_r = mix.get_C_rep()
                _l = "    {:<16s}{:<16s}{:<8d}{:<16.6e}{:<16.6e}{:<s}".format(
                    _namei, _namej, 1, C_a, C_r, ";"
                )
                return_string = "\n".join([return_string, _l])

        return return_string

    def _open_json(self):
        with open(self.param_file, "r") as f:
            ff_dict = json.load(f)
        ff_dict["crossints"] = {
            frozenset(key.split(":")): value
            for key, value in ff_dict.get("crossints").items()
        }
        return ff_dict

    def _close_json(self, ff_dict):
        ff_dict["crossints"] = {
            ":".join(key): value for key, value in ff_dict.get("crossints").items()
        }
        with open(self.param_file, "w") as f:
            json.dump(ff_dict, f)
