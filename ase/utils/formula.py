import re
import sys
from ase.utils import gcd
from ase.data import chemical_symbols

if sys.version_info >= (3, 6):
    ordereddict = dict
else:
    from collections import OrderedDict as ordereddict


# Tree = Typedef('Tree', str, Tuple['Tree', int], List['Tree'])


class Formula:
    def __init__(self, formula: str = '',
                 _tree=None,  # Tree
                 _count=None):  # Dict[str, int]
        """Chemical formula object.

        Parameters
        ----------
        formula: str
            Text string representation of formula.  Examples: '6CO2',
            '30Cu+2CO', 'Pt(CO)6'.

        Examples
        --------
        >>> f = Formula('H2O')
        >>> f.count()
        {'H': 2, 'O': 1}
        >>> 'H' in f
        True
        >>> f == 'HOH'
        True
        >>> f'{f:latex}'
        'H$_{2}$O'
        >>> f.format('latex')
        'H$_{2}$O'
        >>> divmod(6 * f + 'Cu', 'H2O')
        (6, Formula('Cu'))

        Raises
        ------
        `ValueError` on malformed formula.
        """
        self._formula = formula
        self._tree = _tree or parse(formula)
        self._count = _count or count_tree(self._tree)

    def count(self):  # -> Dict[str, int]
        """Return dictionary mapping chemical symbol to number of atoms.

        Example
        -------
        >>> Formula('H2O').count()
        {'H': 2, 'O': 1}
        """
        return self._count.copy()

    @staticmethod
    def from_dict(dct):  # (Dict[str, int]) -> Formula
        """Convert dict to Formula."""
        return Formula(dict2str(dct),
                       _tree=[([(symb, n) for symb, n in dct.items()], 1)],
                       _count=dict(dct))

    def __iter__(self, tree=None):
        if tree is None:
            tree = self._tree
        if isinstance(tree, str):
            yield tree
        elif isinstance(tree, tuple):
            tree, N = tree
            for _ in range(N):
                yield from self.__iter__(tree)
        else:
            for tree in tree:
                yield from self.__iter__(tree)

    def __str__(self):
        return self._formula

    def __repr__(self):
        return 'Formula({!r})'.format(self._formula)

    def format(self, fmt: str = '') -> str:
        """Format formula as string.

        Formats:

        * hill: alphabetically ordered with C and H first
        * metal: alphabetically ordered with metals first
        * latex: LaTeX representation
        * html: HTML representation
        * rest: reStructuredText representation

        Example
        -------
        >>> Formula('H2O').format(html)
        'H<sub>2>/sub>O'
        """
        return format(self, fmt)

    def __getitem__(self, symb: str) -> int:
        """Number of atoms with chemical symbol *symb*."""
        return self._count.get(symb, 0)

    def __contains__(self, f) -> bool:  # (Union[str, Formula])
        """Check if formula contains chemical symbols in *f*.

        Type of *f* must be str or Formula.

        Example
        -------
        >>> 'OH' in Formula('H2O')
        True
        """
        if isinstance(f, str):
            f = Formula(f)
        for symb, n in f._count.items():
            if self[symb] < n:
                return False
        return True

    def __eq__(self, other):
        """Equality check.

        Note that order is not important.

        Example
        -------
        >>> Formula('CO') == Formula('OC')
        True
        """
        if isinstance(other, str):
            other = Formula(other)
        return self._count == other._count

    def __divmod__(self, other):
        """Return the tuple (self // other, self % other).

        Invariant::

            div, mod = divmod(self, other)
            div * other + mod == self
        """
        if isinstance(other, str):
            other = Formula(other)
        N = min(self[symb] // n for symb, n in other._count.items())
        dct = self.count()
        if N:
            for symb, n in other._count.items():
                dct[symb] -= n * N
                if dct[symb] == 0:
                    del dct[symb]
        return N, self.from_dict(dct)

    def __rdivmod__(self, other):
        return divmod(Formula(other), self)

    def __mod__(self, other):
        return divmod(self, other)[1]

    def __rmod__(self, other):
        return Formula(other) % self

    def __floordiv__(self, other):
        return divmod(self, other)[0]

    def __rfloordiv__(self, other):
        return Formula(other) // self

    def __add__(self, other):  # (Union[str, Formula]) -> Formula
        """Add two formulas."""
        if not isinstance(other, str):
            other = other._formula
        return Formula(self._formula + '+' + other)

    def __radd__(self, other: str):  # -> Formula
        return Formula(other) + self

    def __mul__(self, N: int):  # -> Formula
        """Repeat formula `N` times."""
        return self.from_dict({symb: n * N
                               for symb, n in self._count.items()})

    def __rmul__(self, N: int):  # -> Formula
        return self * N

    def __len__(self):
        """Number of atoms."""
        return sum(self._count.values())

    def compact(self):
        """"""
        return self.from_dict(self._count)

    def _reduce(self):
        N = 0
        for n in self._count.values():
            if N == 0:
                N = n
            else:
                N = gcd(n, N)
        dct = {symb: n // N for symb, n in self._count.items()}
        return dct, N

    def reduce(self):
        """Reduce formula.

        Returns
        -------
        formula: Formula
            Reduced formula.
        n: int
            Number of reduced formula units.

        Example
        -------
        >>> Formula('2H2O').reduce()
        (Formula('H2O'), 2)
        """
        dct, N = self._reduce()
        return self.from_dict(dct), N

    def stoichiometry(self):
        """Reduce to unique stoichiomerty using "chemical symbols" A, B, B, ...

        Examples
        --------
        >>> Formula('CO2').stoichiometry()
        (Formula('A2B'), Formula('O2C'), 1)
        >>> Formula('(H2O)4').stoichiometry()
        (Formula('A2B'), Formula('H2O'), 4)
        """
        count1, N = self._reduce()
        c = ord('A')
        count2 = ordereddict()
        count3 = ordereddict()
        for n, symb in sorted((-n, symb)
                              for symb, n in count1.items()):
            count2[chr(c)] = -n
            count3[symb] = -n
            c += 1
        return self.from_dict(count2), self.from_dict(count3), N

    def __format__(self, fmt: str) -> str:
        """Format Formula as str.

        Possible formats: hill, metal, latex, html, rest.

        Example
        -------
        >>> f = Formula('OH2')
        >>> '{f}, {f:hill}, {f:latex}'.format(f=f)
        'OH2, H2O, OH$_{2}$'
        """

        if fmt == 'hill':
            count = self.count()
            count2 = ordereddict()
            for symb in 'CH':
                if symb in count:
                    count2[symb] = count.pop(symb)
            for symb, n in sorted(count.items()):
                count2[symb] = n
            return dict2str(count2)

        if fmt == 'metal':
            count = self.count()
            result2 = [(s, count.pop(s)) for s in non_metals if s in count]
            result = [(s, count[s]) for s in sorted(count)]
            result += sorted(result2)
            return dict2str(ordereddict(result))

        if fmt == 'latex':
            return self._tostr('$_{', '}$')
        if fmt == 'html':
            return self._tostr('<sub>', '</sub>')
        if fmt == 'rest':
            return self._tostr(r'\ :sub`', r'`\ ')
        if fmt == '':
            return self._formula
        raise ValueError('Invalid format specifier')

    def _tostr(self, sub1, sub2):
        parts = []
        for tree, n in self._tree:
            s = tree2str(tree, sub1, sub2)
            if s[0] == '(' and s[-1] == ')':
                s = s[1:-1]
            if n > 1:
                s = str(n) + s
            parts.append(s)
        return '+'.join(parts)


def dict2str(dct):
    return ''.join(symb + (str(n) if n > 1 else '')
                   for symb, n in dct.items())


def parse(f: str):  # -> Tree
    if not f:
        return []
    parts = f.split('+')
    result = []
    for part in parts:
        n, f = strip_number(part)
        result.append((parse2(f), n))
    return result


def parse2(f: str):  # -> Tree
    units = []
    while f:
        if f[0] == '(':
            level = 0
            for i, c in enumerate(f[1:], 1):
                if c == '(':
                    level += 1
                elif c == ')':
                    if level == 0:
                        break
                    level -= 1
            else:
                raise ValueError
            f2 = f[1:i]
            n, f = strip_number(f[i + 1:])
            unit = (parse2(f2), n)
        else:
            m = re.match('([A-Z][a-z]?)([0-9]*)', f)
            if m is None:
                raise ValueError
            symb = m.group(1)
            number = m.group(2)
            if number:
                unit = (symb, int(number))
            else:
                unit = symb
            f = f[m.end():]
        units.append(unit)
    if len(units) == 1:
        return unit
    return units


def strip_number(s: str):  # -> Tuple[int, str]:
    m = re.match('[0-9]*', s)
    return int(m.group() or 1), s[m.end():]


def tree2str(tree,  # Tree
             sub1: str, sub2: str):  # -> str
    if isinstance(tree, str):
        return tree
    if isinstance(tree, tuple):
        tree, N = tree
        s = tree2str(tree, sub1, sub2)
        if N == 1:
            if s[0] == '(' and s[-1] == ')':
                return s[1:-1]
            return s
        return s + sub1 + str(N) + sub2
    return '(' + ''.join(tree2str(tree, sub1, sub2) for tree in tree) + ')'


def count_tree(tree):  # (Tree) -> Dict[str, int]
    if isinstance(tree, str):
        return {tree: 1}
    if isinstance(tree, tuple):
        tree, N = tree
        return {symb: n * N for symb, n in count_tree(tree).items()}
    dct = {}
    for tree in tree:
        for symb, n in count_tree(tree).items():
            m = dct.get(symb, 0)
            dct[symb] = m + n
    return dct


# non metals, half-metals/metalloid, halogen, noble gas:
non_metals = ['H', 'He', 'B', 'C', 'N', 'O', 'F', 'Ne',
              'Si', 'P', 'S', 'Cl', 'Ar',
              'Ge', 'As', 'Se', 'Br', 'Kr',
              'Sb', 'Te', 'I', 'Xe',
              'Po', 'At', 'Rn']


# Backwards compatibility:
def formula_hill(numbers, empirical=False):
    """Convert list of atomic numbers to a chemical formula as a string.

    Elements are alphabetically ordered with C and H first.

    If argument `empirical`, element counts will be divided by greatest common
    divisor to yield an empirical formula"""
    symbols = [chemical_symbols[Z] for Z in numbers]
    f = Formula('', [(symbols, 1)])
    if empirical:
        f, _ = f.reduce()
    return f.format('hill')


# Backwards compatibility:
def formula_metal(numbers, empirical=False):
    """Convert list of atomic numbers to a chemical formula as a string.

    Elements are alphabetically ordered with metals first.

    If argument `empirical`, element counts will be divided by greatest common
    divisor to yield an empirical formula"""
    symbols = [chemical_symbols[Z] for Z in numbers]
    f = Formula('', [(symbols, 1)])
    if empirical:
        f, _ = f.reduce()
    return f.format('metal')
