"""Exact algebra examples; no checkpoints, datasets, optimization or accuracy."""

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path


def matrix(rows):
    return [[F(x) for x in row] for row in rows]


def add(a, b):
    return [[x + y for x, y in zip(ar, br)] for ar, br in zip(a, b)]


def scale(a, c):
    return [[c * x for x in row] for row in a]


def multiply(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def square(a):
    return [[x * x for x in row] for row in a]


def encode(a):
    return [[str(x) for x in row] for row in a]


def main():
    identity = matrix([[1, 0], [0, 1]])
    zero = matrix([[0, 0], [0, 0]])
    p1 = matrix([[1, 0], [0, 0]])
    c1 = add(identity, scale(p1, -1))
    p2 = scale(matrix([[81, 36], [36, 16]]), F(1, 97))
    c2 = add(identity, scale(p2, -1))
    anchor1 = matrix([[2, 1], [0, 1]])
    anchor2 = matrix([[1, 0]])
    task1 = matrix([[1, 0], [2, 0]])
    task2 = scale(matrix([[9, 4]]), F(1, 97))
    expert1 = add(anchor1, task1)
    expert2 = add(anchor2, task2)
    edit1 = matrix([[0, 5], [0, -3]])
    edit2 = matrix([[36, -81]])
    release1 = add(expert1, edit1)
    release2 = add(expert2, edit2)

    for p, c in [(p1, c1), (p2, c2)]:
        assert multiply(p, p) == p
        assert multiply(p, c) == zero
    assert multiply(task1, c1) == zero
    assert multiply(task2, c2) == matrix([[0, 0]])
    assert multiply(edit1, p1) == zero
    assert multiply(edit2, p2) == matrix([[0, 0]])

    cleaned1 = add(multiply(release1, p1), multiply(anchor1, c1))
    cleaned2 = add(multiply(release2, p2), multiply(anchor2, c2))
    assert cleaned1 == expert1 and cleaned2 == expert2
    native_coefficient = multiply(expert2, square(multiply(expert1, matrix([[1], [0]]))))
    release_coefficient = multiply(release2, square(multiply(release1, matrix([[1], [0]]))))
    assert native_coefficient == release_coefficient == matrix([[10]])

    # Partner enters this illustrative merge only, not the edit definitions.
    partner1 = matrix([[2, 0], [4, 1]])
    partner2 = anchor2
    unprotected1 = scale(add(expert1, partner1), F(1, 2))
    unprotected2 = scale(add(expert2, partner2), F(1, 2))
    protected1 = scale(add(release1, partner1), F(1, 2))
    protected2 = scale(add(release2, partner2), F(1, 2))
    x = matrix([[1], [0]])
    original_output = multiply(unprotected2, square(multiply(unprotected1, x)))
    released_output = multiply(protected2, square(multiply(protected1, x)))
    assert add(released_output, scale(original_output, -1)) == matrix([[-252]])
    restored1 = scale(add(cleaned1, partner1), F(1, 2))
    restored2 = scale(add(cleaned2, partner2), F(1, 2))
    assert multiply(restored2, square(multiply(restored1, x))) == original_output

    # Known permutation symmetry preserves the final map, not each task input map.
    swap = matrix([[0, 1], [1, 0]])
    assert multiply(swap, swap) == identity
    active_change = multiply(add(swap, scale(identity, -1)), p1)
    assert active_change != zero
    for vector in [matrix([[2], [-3]]), matrix([[F(1, 3)], [5]])]:
        assert multiply(swap, square(multiply(swap, vector))) == square(vector)

    result = {
        "scope": "Own exact-rational illustrative square-activation network; not CLIP or a defense candidate",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "native_input_family": "x=(t,0); native output is 10*t^2 for original and release",
        "input_projectors": {"first": encode(p1), "second": encode(p2)},
        "task_update_complement": "Both tau_i C_i equal zero exactly",
        "edit_task_component": "Both E_i P_i equal zero exactly",
        "public_completion": "Each Wprime_i P_i + W0_i C_i equals the original expert W_i exactly",
        "merged_original_output_at_t1": str(original_output[0][0]),
        "merged_release_output_at_t1": str(released_output[0][0]),
        "merged_output_difference": "-252",
        "merged_restoration": "Exact original output after public completion",
        "global_permutation_active_change": encode(active_change),
        "permutation_novelty": "Known symmetry; illustrates preservation conditions, not a new candidate",
        "all_checks_passed": True,
        "new_CLIP_forwards": 0,
        "new_protected_checkpoints": 0,
        "new_classification_results": 0,
        "limits": "No real utility guarantee; no all-complement impossibility; approximate alignment, finite calibration, floating-point arithmetic, output-side complements and only-final-output preservation require separate conditions",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
