
#ifndef EXTERNAL_CONCAT
#include "mandelstructs.h"
#endif

// add macro to detect if gcc or opencl and use corresponding builtins?
inline FPN _abs(FPN x) { return x > 0 ? x : -x; }

inline FPN _min(FPN a, FPN b) { return a < b ? a : b; }

Complex_t complex_add(Complex_t a, Complex_t b) {
  Complex_t c;
  c.re = a.re + b.re;
  c.im = a.im + b.im;
  return c;
}

Complex_t complex_mult(Complex_t a, Complex_t b) {
  Complex_t c;
  c.re = a.re * b.re - a.im * b.im;
  c.im = a.im * b.re + a.re * b.im;
  return c;
}

Complex_t complex_pow(Complex_t z, int n) {
  Complex_t p = z;
  for (int i = 1; i < n; i++) {
    p = complex_mult(p, z);
  }
  return p;
}

// function which we recurse
//>>
inline Complex_t f(Complex_t z, Complex_t c) {
  return complex_add(complex_pow(z, 2), c);
}
//<<

int in_circle(Complex_t z, Complex_t z0, FPN r) {
  FPN dre = z.re - z0.re;
  FPN dim = z.im - z0.im;
  return dre * dre + dim * dim < r * r;
}

inline int in_box(Complex_t z, Box_t b) {
  return z.re > b.left && z.re < b.right && z.im > b.bot && z.im < b.top;
}

inline int in_bounds(Complex_t z) {
  return in_circle(z, (Complex_t){FZERO, FZERO}, 2);
}

FPN proximity(Complex_t z, ProxType_t PROXTYPE)
// Various things we can measure distance from...
{
  FPN res = 1000 * FONE;
  if (PROXTYPE.to_unit_circ) {
    res =
        _min(res, z.re * z.re + z.im * z.im); // normalise to be more in the
                                              // same value range as other two?
  }
  if (PROXTYPE.to_horizontal) {
    res = _min(res, _abs(z.re));
  }
  if (PROXTYPE.to_vertical) {
    res = _min(res, _abs(z.im));
  }
  return res;
}

int _escape_iter(Complex_t z, Complex_t c, int MAXITER) {

  int i = 0;
  while (i < MAXITER && in_bounds(z)) {
    z = f(z, c);
    i += 1;
  }

  return i;
}

FPN _minprox(Complex_t z, Complex_t c, int MAXITER, ProxType_t PROXTYPE)
// more of a distance field?
{

  int i = 0;
  FPN dist = proximity(z, PROXTYPE);
  while (i < MAXITER && in_bounds(z)) {
    z = f(z, c);
    dist = _min(dist, proximity(z, PROXTYPE));
    i += 1;
  }

  return dist;
}

Complex_t _orbit_trap(Complex_t z, Complex_t c, Box_t b, int MAXITER)
// returns UV coords in given box
{
  Complex_t res = {-b.left, -b.bot};

  int i = 0;
  while (i < MAXITER) {
    i += 1;
    z = f(z, c);
    if (in_box(z, b)) {
      res = complex_add(res, z);
      res.re /= (b.right - b.left);
      res.im /= (b.top - b.bot);
      return res;
    }
  }

  return (Complex_t){FZERO, FZERO};
}
