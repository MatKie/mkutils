from .mie import mie
import json
import os


class FFWriter(object):
    def __init__(self, param_file, rule="SAFT"):
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
        rule : str, optional
            Either SAFT or SDK (for the SDK/SPICA forcefield). By default SAFT. 
        Methods
        -------
        add_atomtype():
            Add an atomtype, e.g. CM
        add_crossint():
            Add a cross interaction, e.g. CM - CT
        """
        self.param_file = param_file
        self.rule = rule

    def add_atomtype(self, name, l_r, l_a, eps, sig, MW, at_num, update=False):
        """
        Add an atomtype/group to the forcefield. Bascially all the self
        paramters.

        Parameters
        ----------
        name : str
            Name of the group/atomtype to add
        l_r : float
            repulsive exponent of mie potential.
        l_a : float
            attractive exponent of mie potential.
        eps : float
            well depth of mie potential in Kelvin.
        sig : float
            sigma parameter of mie potential in nm.
        MW : float
            molecular weight of group. 
        at_num : int
            The proton number of a group. I think it's unused but better
            save than sorry.
        update : bool, optional
            This flag needs to be set when we have already defined the atomtype to not accidentally overwrite it, by default False

        Raises
        ------
        ValueError
            If the atomtype is already in the parameter file (.json file)
            and we did not explicitly set the update flag.
        """
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

    def add_crossint(
        self, name1, name2, k=0.0, eps_mix=False, sig_mix=False, update=False, wca=False
    ):
        """
        Add specific crossinteractions if they don't follow mixing rules.
        Can either be specified by k_ij or directly the well depth. 
        Be a nice kid and don't specify both as I'm too lazy checking
        with one has priority.

        Parameters
        ----------
        name1 : str
            group 1 name.
        name2 : str
            group 2 name.
        k : float, optional
            k_ij as defined by: eps_mix = (1-k_ij)*eps_mix_rule, by default 0.0.
        eps_mix : bool or float, optional
            If float well depth of this interaction in Kelvin , by default False.
        sig_mix : bool or float, optional
            If given, sigma parameter in nm. Don't know if it works for SAFT rule as it was designed for SDK rule. So check if it's giving you the right results, by default False.
        update : bool, optional
            This flag needs to be set when we have already defined the atomtype to not accidentally overwrite it, by default False.
        wca : bool, optional
            if True, the potential is shifted by the well depth at the location of the potential well. This results in a purely repulsive potential (Weeks-chandler-anderson for 12-6), by default False.

        Raises
        ------
        ValueError
            When you didn't put update flag to true and changing things around.
        ValueError
            If one of the groups is not already added as an atomtype.
        """
        _fr_set = frozenset([name1, name2])
        _dict = {_fr_set: {"k": k, "eps_mix": eps_mix, "sig_mix": sig_mix, "wca": wca}}

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
        """
        Write the nonbonded part of the forcefield from all the info
        we put in so far.

        The important calculations are made in mie.get_C_attr() and mie.get_C_rep() in mie.py.

        C_rep  = C_mie*eps*sig^l_r 
        C_attr = C_mie*eps*sig^l_a
        
        These later get multiplied with what's in the tables:

        C_attr g(r) + C_rep h(r), where
        g(r) = -(1/r)^l_a
        h(r) = (1/r)^l_r

        Parameters
        ----------
        outfile : str, optional
            how to name the outputfile, by default "ffnonbonded.itp"
        """
        atomtypes = self.print_atomtypes()
        crossints = self.print_crossints()
        with open(outfile, "w") as f:
            f.write(atomtypes)
            f.write("\n\n")
            f.write(crossints)

    def print_atomtypes(self):
        """
        Generates the self interaction. passes all relevant info to mie from mie.py.

        Returns
        -------
        string
            string of the self interactions.
        """
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
        """
        Write tables for all self and cross interactions into the folder 'tables'. All possible tables are generated (CM-CT and CT-CM) so we don't have to care how we define it later on in the energygroups.

        This makes use of mie.mix & mie.write_table from mie.py.

        V(r) = q_i*q_j/eps0 f(r) + C_la g(r) + C_lr h(r)

        -> Tables include r, f(r), -f'(r), g(r), -g'(r), h(r), h'(r)

        For a 8-6 potential g, h and derivatives are:

        g(r) = -(1/r)^6, -g'(r) = -6*(1/r)^7 (force, attractive)
        h(r) = (1/r)^8, -h'(r) = 8*(1/r)^9 (force, repulsive)

        Parameters
        ----------
        shift : bool, optional
            Whether or not to shift the potential at curoff, by default True
        cutoff : float, optional
            last point in tables (not including table extension), by default 2.0
        double_prec : bool, optional
            Wheter or not to write double precision tables which are more closely spaced, by default False
        """
        ff_dict = self._open_json()
        atomtypes = ff_dict.get("atomtypes")
        crossints = ff_dict.get("crossints")
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

                this_ci = crossints.get(frozenset([type_i, type_j]), {})
                sig_mix = this_ci.get("sig_mix", False)
                wca = this_ci.get("wca", False)

                # eps_mix is not actually used for tables
                mix = mie.mix(
                    _mie_i,
                    _mie_j,
                    eps_mix=1e6,
                    sig_mix=sig_mix,
                    rule=self.rule,
                    rc=cutoff,
                    shift=shift,
                    wca=wca,
                )

                mix.write_table(
                    names=(type_i, type_j),
                    double_prec=double_prec,
                    cutoff=cutoff,
                    shift=shift,
                )
        os.chdir("..")

    def print_crossints(self):
        """
        Calculate a mie potential for the crossinteraction via mie.mix from mie.py and then the attractive and repulsive exponents vie _get_C_attr() and _get_C_rep(). Also print some comments like eps-mix and l_r-mix.

        Returns
        -------
        string
            string of all the crossints.
        """
        _fl = "[ nonbond_params ]"
        _sl = ";   {:<16s}{:<16s}{:<8s}{:<16s}{:<16s}{:<4s}\t{:<7s}\t{:<7s}".format(
            "i", "j", "func", "V(C6)", "W(Cm)", "Ref", "eps/K", "l_r"
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
                    _kwargs_mix = {"k": 0.0, "eps_mix": False, "sig_mix": False}

                mix = mie.mix(_mie_i, _mie_j, rule=self.rule, **_kwargs_mix)
                C_a = mix.get_C_attr()
                C_r = mix.get_C_rep()
                eps_mix = mix.eps
                lr_mix = mix.l_r
                _l = "    {:<16s}{:<16s}{:<8d}{:<16.6e}{:<16.6e}{:<s}{:<s}\t{:<7.2f}\t{:<7.2f}".format(
                    _namei, _namej, 1, C_a, C_r, ";", "N/A", eps_mix, lr_mix
                )
                return_string = "\n".join([return_string, _l])

        return return_string

    def get_crossints(self):
        """
        Calculates all parameters, also from mixing rules.

        Returns
        -------
        dict
            dictionary with all cross interactions (and self interactions). Also the ones acc. to mixing rules.
        """
        return_dict = {}
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
                    _kwargs_mix = {"k": 0.0, "eps_mix": False, "sig_mix": False}

                mix = mie.mix(_mie_i, _mie_j, rule=self.rule, **_kwargs_mix)
                sig_mix = mix.sig
                eps_mix = mix.eps
                lr_mix = mix.l_r
                la_mix = mix.l_a
                temp_dict = {
                    "eps": eps_mix,
                    "sig": sig_mix,
                    "l_r": lr_mix,
                    "l_a": la_mix,
                }
                name_dict = {crossint_id: temp_dict}
                return_dict.update(name_dict)

        return return_dict

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
            json.dump(ff_dict, f, indent=4, sort_keys=True)
