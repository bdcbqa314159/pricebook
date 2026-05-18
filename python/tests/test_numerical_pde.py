"""Tests for numerical._pde."""
import pytest, numpy as np
from pricebook.numerical._pde import hundsdorfer_verwer, psor_2d

class TestImports:
    def test_hv_callable(self):
        assert callable(hundsdorfer_verwer)
    def test_psor_callable(self):
        assert callable(psor_2d)
