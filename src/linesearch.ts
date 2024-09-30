import { dot, weightedSum } from './blas1'

/**
 * Searches along line 'pk' for a point that satisfies the Wolfe conditions.
 * See 'Numerical Optimization' by Nocedal and Wright p59-60.
 *
 * @param {function} f - Objective function.
 * @param {number[]} pk - Search direction.
 * @param {{ fx: number; fxprime: number[]; x: number[] }} current - Object containing current gradient/loss.
 * @param {{ fx: number; fxprime: number[]; x: number[] }} next - Output: contains next gradient/loss.
 * @param {number} [a=1] - Initial step size.
 * @param {number} [c1=1e-6] - Wolfe condition parameter.
 * @param {number} [c2=0.1] - Wolfe condition parameter.
 * @returns {number} Step size taken.
 */
export const wolfeLineSearch = (
  f: (x: number[], fxprime: number[]) => number,
  pk: number[],
  current: { fx: number; fxprime: number[]; x: number[] },
  next: { fx: number; fxprime: number[]; x: number[] },
  a = 1,
  c1 = 1e-6,
  c2 = 0.1
): number => {
  const phi0 = current.fx
  const phiPrime0 = dot(current.fxprime, pk)
  let phi = phi0
  let phi_old = phi0
  let phiPrime = phiPrime0
  let a0 = 0

  const zoom = (a_lo: number, a_high: number, phi_lo: number): number => {
    for (let iteration = 0; iteration < 16; ++iteration) {
      a = (a_lo + a_high) / 2
      weightedSum(next.x, 1.0, current.x, a, pk)
      phi = next.fx = f(next.x, next.fxprime)
      phiPrime = dot(next.fxprime, pk)

      if (phi > phi0 + c1 * a * phiPrime0 || phi >= phi_lo) {
        a_high = a
      } else {
        if (Math.abs(phiPrime) <= -c2 * phiPrime0) {
          return a
        }

        if (phiPrime * (a_high - a_lo) >= 0) {
          a_high = a_lo
        }

        a_lo = a
        phi_lo = phi
      }
    }

    return 0
  }

  for (let iteration = 0; iteration < 10; ++iteration) {
    weightedSum(next.x, 1.0, current.x, a, pk)
    phi = next.fx = f(next.x, next.fxprime)
    phiPrime = dot(next.fxprime, pk)
    if (phi > phi0 + c1 * a * phiPrime0 || (iteration && phi >= phi_old)) {
      return zoom(a0, a, phi_old)
    }

    if (Math.abs(phiPrime) <= -c2 * phiPrime0) {
      return a
    }

    if (phiPrime >= 0) {
      return zoom(a, a0, phi)
    }

    phi_old = phi
    a0 = a
    a *= 2
  }

  return a
}
