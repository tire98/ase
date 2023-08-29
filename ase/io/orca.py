import os
import re
from io import StringIO
from pathlib import Path

import numpy as np

from ase.io import read
from ase.atoms import Atoms
from ase.units import Bohr, Hartree
from ase.utils import reader, writer

# Made from NWChem interface


@reader
def read_geom_orcainp(fd):
    """Method to read geometry from an ORCA input file."""
    lines = fd.readlines()

    # Find geometry region of input file.
    stopline = 0
    for index, line in enumerate(lines):
        if line[1:].startswith('xyz '):
            startline = index + 1
            stopline = -1
        elif (line.startswith('end') and stopline == -1):
            stopline = index
        elif (line.startswith('*') and stopline == -1):
            stopline = index
    # Format and send to read_xyz.
    xyz_text = '%i\n' % (stopline - startline)
    xyz_text += ' geometry\n'
    for line in lines[startline:stopline]:
        xyz_text += line
    atoms = read(StringIO(xyz_text), format='xyz')
    atoms.set_cell((0., 0., 0.))  # no unit cell defined

    return atoms


@writer
def write_orca(fd, atoms, params):
    # conventional filename: '<name>.inp'
    fd.write("! %s \n" % params['orcasimpleinput'])
    fd.write("%s \n" % params['orcablocks'])

    fd.write('*xyz')
    fd.write(" %d" % params['charge'])
    fd.write(" %d \n" % params['mult'])
    for atom in atoms:
        if atom.tag == 71:  # 71 is ascii G (Ghost)
            symbol = atom.symbol + ' : '
        else:
            symbol = atom.symbol + '   '
        fd.write(symbol +
                 str(atom.position[0]) + ' ' +
                 str(atom.position[1]) + ' ' +
                 str(atom.position[2]) + '\n')
    fd.write('*\n')


@reader
def read_orca_energy(outputfile: str) -> float:
    """Read Energy from ORCA output file."""
    re_energy = re.compile(r"FINAL SINGLE POINT ENERGY.*\n")
    re_not_converged = re.compile(r"Wavefunction not fully converged")

    found_line = re_energy.finditer(outputfile)
    energy = float('nan')
    for match in found_line:
        if not re_not_converged.search(match.group()):
            energy = float(match.group().split()[-1])
    if np.isnan(energy):
        raise RuntimeError('No energy')
    else:
        return energy


@reader
def read_orca_forces(fd):
    """Read Forces from ORCA output file."""
    getgrad = False
    gradients = []
    tempgrad = []
    for i, line in enumerate(fd):
        if line.find('# The current gradient') >= 0:
            getgrad = True
            gradients = []
            tempgrad = []
            continue
        if getgrad and "#" not in line:
            grad = line.split()[-1]
            tempgrad.append(float(grad))
            if len(tempgrad) == 3:
                gradients.append(tempgrad)
                tempgrad = []
        if '# The at' in line:
            getgrad = False

    forces = -np.array(gradients) * Hartree / Bohr
    return forces

def get_number_of_jobs(output_file: str):
    """Read number of jobs from ORCA output file."""
    re_jobs = re.compile(r"THERE\s*ARE\s*(\d+)\s*JOBS\s*TO\s*BE\s*PROCESSED\s*THIS\s*RUN")
    match_line = re_jobs.search(output_file)
    if match_line is None:
        return 1
    else:
        return int(match_line.group(1))

def get_base_name(output_file: str) -> str:
    """Get base name of the calculation"""
    base_name_pattern = re.compile(r"BaseName\s*\(.gbw .S,...\)\s*...\s*([A-Za-z0-9_]+)")
    match_line = base_name_pattern.search(output_file)
    if match_line is None:
        return None
    else:
        return match_line.group(1)


def get_task(output_file: str) -> str:
    """Read tasks from ORCA output file."""
    tasks_regex = {"geometry_optimisation": r"\s* Geometry Optimization Run\s*", "frequency": r"\s*Energy+Gradient Calculation\s*",
    "single_point": r"\s*Single Point Calculation\s*"}
    for task in tasks_regex:
        pattern = re.compile(tasks_regex[task])
        if pattern.search(output_file):
            return task
    

def split_jobs(output_file: str):
    """Split ORCA output file into jobs."""
    jobs = re.split(r"JOB\s*NUMBER\s*(\d+)", output_file)[1:]
    return jobs


def get_cartesian_coordinates(output_file: str, task: str):
    """Read cartesian coordinates from ORCA output file."""
    # if geometry optimisation, check if converged
    if task == "geometry_optimisation":
        converged_pattern = re.compile(r"\s*THE OPTIMIZATION HAS CONVERGED\s*")
        if not converged_pattern.search(output_file):
            raise RuntimeError('Geometry optimization did not converge')
        output_file = re.split(converged_pattern, output_file)[1]
    cartesian_pattern = re.compile(r"CARTESIAN COORDINATES.*?\n-{2,}\n((?:\s+[A-Za-z]+\s+[\d\-\.]+\s+[\d\-\.]+\s+[\d\-\.]+\n)+)")
    coords_match = cartesian_pattern.findall(output_file)
    energy_pattern = re.compile(r"FINAL SINGLE POINT ENERGY.*\n")
    energy_match = energy_pattern.search(output_file)
    final_energy = float(energy_match.group().split()[-1])
    for match in coords_match:
        element, x, y, z = [], [], [], []
        rows = match.split("\n")
        for row in rows if rows[-1] != "" else rows[:-1]:
            entries = row.split()
            element.append(entries[0])
            x.append(entries[1])
            y.append(entries[2])
            z.append(entries[3])
        
        coordinates = np.array([x, y, z], dtype=float).T
    return [coordinates, element], final_energy

@reader
def read_orca_outputs(out_file, job_number: int = 1):
    results = {}
    # get path to out_file
    file = out_file.read()
    # with open(out_file, 'r') as f:
        # file = f.read()
    num_jobs = get_number_of_jobs(file)
    if num_jobs > 1:
        jobs = split_jobs(file)
        file = jobs[job_number * 2 - 1]
    task = get_task(file)
    base_name = get_base_name(file)
    if base_name is None:
        base_name = out_file.name
    coordinates, energy = get_cartesian_coordinates(file, task)
    atoms = Atoms(coordinates[1], coordinates[0])
    results['task'] = task
    results['base_name'] = base_name
    results['job_number'] = job_number
    results['energy'] = energy
    results['free_energy'] = energy
    results['atoms'] = atoms
    results['no_jobs'] = num_jobs

    # Does engrad always exist? - No!
    # Will there be other files -No -> We should just take engrad
    # as a direct argument.  Or maybe this function does not even need to
    # exist.
    engrad_path = Path(base_name).with_suffix('.engrad')
    if os.path.isfile(engrad_path):
        results['forces'] = read_orca_forces(engrad_path)
    return results
