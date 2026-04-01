# References

Textbooks, papers, and standards cited across the library.

---

## Day Count Conventions

- OpenGamma, *Interest Rate Instruments and Market Conventions Guide*
- Brigo & Mercurio, *Interest Rate Models - Theory and Practice*, 2nd Edition, Ch.1

## Business Day Calendars

- OpenGamma, *Interest Rate Instruments and Market Conventions Guide*
- SIFMA, US Holiday Schedule

## Interpolation

- Press, Teukolsky, Vetterling & Flannery, *Numerical Recipes*, 3rd Edition, Ch.3

## Discount Curves

- Brigo & Mercurio, *Interest Rate Models - Theory and Practice*, 2nd Edition, Ch.1
- Ametrano & Bianchetti, *Everything You Always Wanted to Know About Multiple Interest Rate Curve Bootstrapping but Were Afraid to Ask*, 2013

## Monotone Interpolation

- Hyman, *Accurate Monotonicity Preserving Cubic Interpolation*, SIAM J. Sci. Stat. Comput., 1983 — used in `interpolation.py` (MonotoneCubicInterpolator)
- Fritsch & Carlson, *Monotone Piecewise Cubic Interpolation*, SIAM J. Numer. Anal., 1980 — used in `interpolation.py` (slope computation)

## SABR Model

- Hagan, Kumar, Lesniewski & Woodward, *Managing Smile Risk*, Wilmott Magazine, 2002 — used in `sabr.py` (Hagan approximation for implied vol)

## COS Method

- Fang & Oosterlee, *A Novel Pricing Method for European Options Based on Fourier-Cosine Series Expansions*, SIAM J. Sci. Comput., 2008 — used in `cos_method.py`

## Heston Model

- Heston, *A Closed-Form Solution for Options with Stochastic Volatility*, Review of Financial Studies, 1993 — used in `heston.py` (P1/P2 decomposition)

## AAD (Adjoint Algorithmic Differentiation)

- Savine, *Modern Computational Finance: AAD and Parallel Simulations*, Wiley, 2018 — used in `aad.py` (tape-based reverse-mode AD)

## General References

- Brigo & Mercurio, *Interest Rate Models — Theory and Practice*, Springer, 2006
- Andersen & Piterbarg, *Interest Rate Modeling*, Atlantic Financial Press, 2010
- Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2003
- de Boor, *A Practical Guide to Splines*, Springer, 2001
