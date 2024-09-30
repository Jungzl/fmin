import { norm2, scale, weightedSum } from './blas1'
import { wolfeLineSearch } from './linesearch'

interface OptimizationResult {
  x: number[]
  fx: number
  fxprime: number[]
}

interface OptimizationParams {
  maxIterations?: number
  learnRate?: number
  history?: Array<{
    x: number[]
    fx: number
    fxprime: number[]
    functionCalls?: number[][]
    learnRate?: number
    alpha?: number
  }>
  c1?: number
  c2?: number
}

type ObjectiveFunction = (x: number[], fxprime: number[]) => number

export function gradientDescent(
  f: ObjectiveFunction,
  initial: number[],
  params: OptimizationParams = {}
): OptimizationResult {
  const maxIterations = params.maxIterations || initial.length * 100
  const learnRate = params.learnRate || 0.001
  const current: OptimizationResult = { x: initial.slice(), fx: 0, fxprime: initial.slice() }

  for (let i = 0; i < maxIterations; ++i) {
    current.fx = f(current.x, current.fxprime)
    if (params.history) {
      params.history.push({ x: current.x.slice(), fx: current.fx, fxprime: current.fxprime.slice() })
    }

    weightedSum(current.x, 1, current.x, -learnRate, current.fxprime)
    if (norm2(current.fxprime) <= 1e-5) {
      break
    }
  }

  return current
}

export function gradientDescentLineSearch(
  f: ObjectiveFunction,
  initial: number[],
  params: OptimizationParams = {}
): OptimizationResult {
  let current: OptimizationResult = { x: initial.slice(), fx: 0, fxprime: initial.slice() }
  let next: OptimizationResult = { x: initial.slice(), fx: 0, fxprime: initial.slice() }
  const maxIterations = params.maxIterations || initial.length * 100
  let learnRate = params.learnRate || 1
  const pk = initial.slice()
  const c1 = params.c1 || 1e-3
  const c2 = params.c2 || 0.1
  let temp: OptimizationResult
  let functionCalls: number[][] = []

  if (params.history) {
    // wrap the function call to track linesearch samples
    const inner = f
    f = (x: number[], fxprime: number[]): number => {
      functionCalls.push(x.slice())
      return inner(x, fxprime)
    }
  }

  current.fx = f(current.x, current.fxprime)
  for (let i = 0; i < maxIterations; ++i) {
    scale(pk, current.fxprime, -1)
    learnRate = wolfeLineSearch(f, pk, current, next, learnRate, c1, c2)

    if (params.history) {
      params.history.push({
        x: current.x.slice(),
        fx: current.fx,
        fxprime: current.fxprime.slice(),
        functionCalls: functionCalls,
        learnRate: learnRate,
        alpha: learnRate,
      })
      functionCalls = []
    }

    temp = current
    current = next
    next = temp

    if (learnRate === 0 || norm2(current.fxprime) < 1e-5) break
  }

  return current
}
