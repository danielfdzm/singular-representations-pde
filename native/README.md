# Native helper

`tanh_second_derivative_optimizer.c` implements the long mixed-precision
finite-difference optimizer used by the Gaussian collision experiment. The
Python driver compiles it locally with the compiler selected by `CC` (or the
first supported compiler it discovers).

Compiled libraries are platform-specific and are written under the ignored
`outputs/` directory; they are never part of the reference artifact.
