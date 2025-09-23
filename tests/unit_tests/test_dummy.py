import pytest

# class DummyTest:

def test_1_equals_1():
	assert 1==1

def test_add_numbers():
	from easydynamics.dummy_code import add_numbers
	assert add_numbers(2, 3) == 5
	assert add_numbers(-1, 1) == 0
	assert add_numbers(0, 0) == 0
		
