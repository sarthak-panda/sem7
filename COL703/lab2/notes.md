Given h : A = B:

rw [h] replaces occurrences of A in the goal with B.

rw [← h] replaces occurrences of B in the goal with A.

rw [h.symm] is equivalent to rw [← h]