import unittest
import echelon_calc
import copy
import numpy as np

IN_ECHELON_B = [
    [1, 4, 0, 2],
    [0, 1, -5/7, 3/7],
    [0, 0, 1, 5],
    [0, 0, 0, 1]
]


class TestEchelonCalc(unittest.TestCase):

    def test_swap_function(self):
        b = copy.deepcopy(echelon_calc.B)
        b = echelon_calc.swap(b, 0)
        self.assertEqual(list(b[0]), [2, 8, 0, 4])
        self.assertEqual(list(b[1]), [0, 7, -5, 3])

    def test_fix_function_on_first_row(self):
        b = copy.deepcopy(echelon_calc.B)
        b = echelon_calc.swap(b, 0)
        row_0 = echelon_calc.fix(b, b[0], 0)
        self.assertEqual(list(row_0), [1, 4, 0, 2])

    def test_fix_function_on_second_row(self):
        b = copy.deepcopy(echelon_calc.B)
        b = echelon_calc.swap(b, 0)
        b[0] = np.array([1, 4, 0, 2])
        row_1 = echelon_calc.fix(b, b[1], 1)
        self.assertCountEqual(list(row_1), [0, 1, -5/7, 3/7]) 
    
    def test_echelon_form_conversion(self):
        b = copy.deepcopy(echelon_calc.B)
        b = echelon_calc.convert_to_echelon(b)
        for i in range(0,4):
            self.assertEqual(list(b[i]), IN_ECHELON_B[i])

    def test_in_echelon_form_b(self):
        b = copy.deepcopy(echelon_calc.B)
        b = echelon_calc.convert_to_echelon(b)
        self.assertTrue(echelon_calc.in_echelon_form(b))
    
    def test_in_echelon_form_a(self):
        a = copy.deepcopy(echelon_calc.A)
        a = echelon_calc.convert_to_echelon(a)
        self.assertTrue(echelon_calc.in_echelon_form(a)) 