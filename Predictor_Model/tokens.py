# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 01:43:12 2024

@author: edgar
"""

class tokens_table(object):
    def __init__(self):
        tokens = ['C', '1', '=', 'N', '(', 'S', ')', '2', 'O', '3', '4', 'F',
                  '[C@@H]', '#', 'Cl', '5', '6', '7', '8', '[C@H]', 'Br', '/', '\\',
                  '[C@@]', '[N+]', '[O-]', '.', 'P', '[Br-]', '9', 'I', '[C@]', '[Pt]', 'B', ' ']
        self.table = tokens
        self.table_len = len(self.table)