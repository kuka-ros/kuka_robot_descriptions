# Copyright 2026 KUKA Hungaria Kft.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python3
"""Check that every published inertia tensor is physically realisable.

For each robot the README "What is verified?" table marks as having verified
inertial values, this reads the robot's macro and checks every <inertial>:

  * mass is positive
  * the inertia tensor is positive definite (no zero or negative principal
    moment, and none that is zero to working precision)
  * the principal moments satisfy the triangle inequality a + b >= c

It only reads what is published and never proposes a value. Robots without
verified inertia are skipped. No ROS environment is needed (numpy only), so it
runs as a plain CI step from the repository root:

    python3 check_inertia.py [--root PATH] [--robot NAME ...]

Exit code 0 when every checked tensor passes, 1 otherwise.
"""

import argparse
import pathlib
import sys
import xml.etree.ElementTree as ET

import numpy as np

# A principal moment below this fraction of the largest is treated as zero.
REL_ZERO = 1e-9
# Relative slack on the triangle inequality, for values rounded on export.
TRIANGLE_SLACK = 1e-6
INERTIA_KEYS = ('ixx', 'ixy', 'ixz', 'iyy', 'iyz', 'izz')


def verified_robots(readme):
    """Robot names whose 'Inertial values' cell in the README table is ticked."""
    lines = readme.read_text(encoding='utf-8').splitlines()
    try:
        start = next(i for i, line in enumerate(lines)
                     if line.strip().lower().startswith('## what is verified'))
    except StopIteration:
        sys.exit(f'{readme}: no "What is verified?" section found')
    header, col, robots = None, None, []
    for line in lines[start + 1:]:
        if line.startswith('## '):
            break
        if not line.strip().startswith('|'):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if header is None:
            header = [c.lower() for c in cells]
            if 'inertial values' not in header:
                sys.exit(f'{readme}: no "Inertial values" column')
            col = header.index('inertial values')
            continue
        if set(''.join(cells)) <= set('-: '):
            continue  # the |---|:---:| separator row
        if col < len(cells) and cells[col]:
            robots.append(cells[0])
    if not robots:
        sys.exit(f'{readme}: the verified table lists no robot with inertial values')
    return robots


def find_macro(root, robot):
    hits = sorted(root.glob(f'*/urdf/{robot}_macro.xacro'))
    return hits[0] if len(hits) == 1 else None


def check_macro(path):
    """Yield (link, problem) for every failing <inertial> in one macro."""
    tree = ET.parse(path)
    for link in tree.iter('link'):
        name = link.get('name', '?').replace('${prefix}', '')
        inertial = link.find('inertial')
        if inertial is None:
            continue
        mass_el, inertia_el = inertial.find('mass'), inertial.find('inertia')
        if mass_el is None or inertia_el is None:
            yield name, 'inertial without <mass> or <inertia>'
            continue
        raw = [mass_el.get('value')] + [inertia_el.get(k) for k in INERTIA_KEYS]
        try:
            mass, ixx, ixy, ixz, iyy, iyz, izz = (float(v) for v in raw)
        except (TypeError, ValueError):
            yield name, f'not a literal number, cannot check {raw}'
            continue
        if mass <= 0:
            yield name, f'mass {mass:g} is not positive'
        tensor = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
        a, b, c = np.linalg.eigvalsh(tensor)  # ascending
        moments = f'principal moments [{a:.6g}, {b:.6g}, {c:.6g}]'
        if c <= 0 or a <= REL_ZERO * c:
            yield name, f'not positive definite, {moments}'
        elif a + b < c * (1 - TRIANGLE_SLACK):
            yield name, f'triangle inequality broken, {moments}, a + b = {a + b:.6g} < c'


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--root', type=pathlib.Path, default=pathlib.Path('.'),
                        help='repository root (default: current directory)')
    parser.add_argument('--robot', action='append',
                        help='check only this robot (repeatable); must be in the verified table')
    args = parser.parse_args()
    root = args.root.resolve()

    robots = verified_robots(root / 'README.md')
    if args.robot:
        unknown = sorted(set(args.robot) - set(robots))
        if unknown:
            sys.exit(f'not marked as verified inertia in the README: {", ".join(unknown)}')
        robots = args.robot

    failures = 0
    for robot in robots:
        macro = find_macro(root, robot)
        if macro is None:
            print(f'FAIL {robot}: no single */urdf/{robot}_macro.xacro found')
            failures += 1
            continue
        problems = list(check_macro(macro))
        rel = macro.relative_to(root).as_posix()
        if not problems:
            print(f'ok   {robot} ({rel})')
        for link, problem in problems:
            print(f'FAIL {robot} {link} ({rel}): {problem}')
            failures += 1

    print(f'{len(robots)} robots checked, {failures} problem(s)')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
