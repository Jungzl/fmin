import { dot, gemv, norm2, scale, weightedSum } from './blas1'
import { wolfeLineSearch } from './linesearch'

interface OptimizationResult {
  x: number[]
  fx: number
  fxprime: number[]
}

interface OptimizationParams {
  maxIterations?: number
  history?: Array<{
    x: number[]
    fx: number
    fxprime: number[]
    alpha: number
  }>
}

type ObjectiveFunction = (x: number[], fxprime: number[]) => number

export function conjugateGradient(
  f: ObjectiveFunction,
  initial: number[],
  params: OptimizationParams = {}
): OptimizationResult {
  // allocate all memory up front here, keep out of the loop for performance reasons
  let current: OptimizationResult = { x: initial.slice(), fx: 0, fxprime: initial.slice() }
  let next: OptimizationResult = { x: initial.slice(), fx: 0, fxprime: initial.slice() }
  const yk = initial.slice()
  let pk: number[]
  let temp: OptimizationResult
  let a = 1
  const maxIterations = params.maxIterations || initial.length * 20

  current.fx = f(current.x, current.fxprime)
  pk = current.fxprime.slice()
  scale(pk, current.fxprime, -1)

  for (let i = 0; i < maxIterations; ++i) {
    a = wolfeLineSearch(f, pk, current, next, a)

    if (params.history) {
      params.history.push({ x: current.x.slice(), fx: current.fx, fxprime: current.fxprime.slice(), alpha: a })
    }

    if (!a) {
      // failed to find point that satisfies wolfe conditions.
      // reset direction for next iteration
      scale(pk, current.fxprime, -1)
    } else {
      // update direction using Polak–Ribiere CG method
      weightedSum(yk, 1, next.fxprime, -1, current.fxprime)

      const delta_k = dot(current.fxprime, current.fxprime)
      const beta_k = Math.max(0, dot(yk, next.fxprime) / delta_k)

      weightedSum(pk, beta_k, pk, -1, next.fxprime)

      temp = current
      current = next
      next = temp
    }

    if (norm2(current.fxprime) <= 1e-5) {
      break
    }
  }

  if (params.history) {
    params.history.push({ x: current.x.slice(), fx: current.fx, fxprime: current.fxprime.slice(), alpha: a })
  }

  return current
}

/// Solves a system of linear equations Ax = b for x
/// using the conjugate gradient method.
export function conjugateGradientSolve(
  A: number[][],
  b: number[],
  x: number[],
  history?: Array<{ x: number[]; p: number[]; alpha: number }>
): number[] {
  const r = x.slice()
  const Ap = x.slice()
  let p: number[]
  let rsold: number
  let rsnew: number
  let alpha = 0 // Initialize alpha with a default value

  // r = b - A*x
  gemv(Ap, A, x)
  weightedSum(r, 1, b, -1, Ap)
  p = r.slice()
  rsold = dot(r, r)

  for (let i = 0; i < b.length; ++i) {
    gemv(Ap, A, p)
    alpha = rsold / dot(p, Ap)
    if (history) {
      history.push({ x: x.slice(), p: p.slice(), alpha: alpha })
    }

    //x = x + alpha * p;
    weightedSum(x, 1, x, alpha, p)

    // r = r - alpha * Ap;
    weightedSum(r, 1, r, -alpha, Ap)
    rsnew = dot(r, r)
    if (Math.sqrt(rsnew) <= 1e-10) break

    // p = r + (rsnew / rsold) * p;
    weightedSum(p, 1, r, rsnew / rsold, p)
    rsold = rsnew
  }
  if (history) {
    history.push({ x: x.slice(), p: p.slice(), alpha: alpha })
  }
  return x
}
