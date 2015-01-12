#!/usr/bin/env python2

import sys

# helper function, print a simply progress bar
def pbar(title, n, N):
    # outputs title: n / N
    output_str = str(title) + ": " + str(n) + " / " + str(N)
    sys.stdout.write("\b" * (len(output_str)+1))
    sys.stdout.flush()
    sys.stdout.write(str(output_str))
    sys.stdout.flush()
