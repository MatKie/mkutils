from mkutils.gromacs_writer import FFWriter
from pytest import approx
import pytest
import json
import os


test_dict = {"crossint": {}, "atomtypes": {}}

with open("test.json", "w") as f:
    json.dump(test_dict, f)

json_file = os.path.join("test.json")
this_ff = FFWriter(json_file)

this_ff.add_atomtype("W", 8, 6, 305.21, 0.29016, 18.015028, 18)

this_ff.add_atomtype("T", 15.947, 6, 358.37, 0.45012, 43.08698, 43)

this_ff.add_atomtype("M", 16.433, 6, 377.14, 0.4184, 42.07914, 42)

this_ff.add_crossint("W", "T", 0.7)

this_ff.add_atomtype("Na", 8, 6, 50, 0.23, 23.0, 23)


def test_crossint1():
    with pytest.raises(ValueError, match=r"Crossint already .*"):
        this_ff.add_crossint("W", "T", 1000)


def test_crossint2():
    with pytest.raises(ValueError, match=r"T3 not an atomtype!"):
        this_ff.add_crossint("W", "T3", 0.5)


def test_crossint3():
    this_ff.add_crossint("W", "T", 0.31, update=True)
    with open(json_file, "r") as f:
        _ff_dict = json.load(f)
        try:
            assert _ff_dict.get("crossint").get("W:T").get("k") == approx(0.31)
        except:
            assert _ff_dict.get("crossint").get("T:W").get("k") == approx(0.31)


def test_atomtype1():
    with pytest.raises(ValueError, match=r"Atomtype already .*"):
        this_ff.add_atomtype("Na", 8, 6, 55, 0.24, 23.1, 23)


def test_atomtype2():
    this_ff.add_atomtype("Na", 9, 7, 55, 0.24, 23.1, 23, update=True)
    with open(json_file, "r") as g:
        _ff_dict = json.load(g)
        _dict = _ff_dict.get("atomtypes").get("Na")
        assert _dict.get("eps") == approx(55)
        assert _dict.get("sig") == approx(0.24)
        assert _dict.get("l_r") == approx(9)
        assert _dict.get("l_a") == approx(7)
        assert _dict.get("MW") == approx(23.1)


this_ff.add_crossint("W", "T", 0.31, update=True)


def test_write_atomtypes():
    print(this_ff.print_atomtypes())


def test_write_crossints():
    print(this_ff.print_crossints())


def test_write_all():
    this_ff.write_forcefield()


test_write_atomtypes()
test_write_crossints()
test_write_all()
