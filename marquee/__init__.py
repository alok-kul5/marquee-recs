import os

# implicit is much faster with a single-threaded BLAS
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
