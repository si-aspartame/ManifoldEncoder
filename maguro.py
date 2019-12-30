import os
import subprocess

def compr7zip(filename, zipname, verbose=False):
    if verbose:
        print(f'-- COMPRESS {filename} --')
    system = subprocess.Popen(["C:/Program Files/7-Zip/7zip.exe", "a", zipname, filename])
    return(system.communicate())

def extr7zip(zipname, verbose=False):
    if verbose:
        print(f'-- EXTRACT {filename} --')
    system = subprocess.Popen(["C:/Program Files/7-Zip/7zip.exe", "e", zipname])
    return(system.communicate())
