"""
Usage: python remove_output.py notebook.ipynb
Python version: 3.9.X
Author: rauls19
"""
import sys
import io
import os
import nbformat as nbf

def removeCellOutputs(data):
    for cell in data.cells:
        print('Cell: ',cell)
        if cell.cell_type == 'code':
            print('Esborrada')
            cell.outputs = []
    return data

if __name__ == '__main__':
    #file_name = sys.argv[1]
    file_name = 'model.ipynb'
    with open(file_name, 'r', encoding='utf8') as f:
        nb = nbf.read(f, nbf.NO_CONVERT)
    print('obert')
    nb = removeCellOutputs(nb)
    with open(file_name, 'w', encoding='utf8') as f:
        nbf.write(nb, f, nbf.NO_CONVERT)