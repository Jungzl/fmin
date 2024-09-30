import { describe, expect, it } from 'vitest'
import {
  conjugateGradient,
  conjugateGradientSolve,
  gradientDescent,
  gradientDescentLineSearch,
  nelderMead,
} from '../src'

const SMALL = 1e-5

function nearlyEqual(left: number, right: number, tolerance = SMALL, message = 'nearlyEqual') {
  expect(Math.abs(left - right)).toBeLessThan(tolerance)
  console.log(`${message}: ${left} ~== ${right}`)
}

// function lessThan(left: number, right: number, message = 'lessThan') {
//   expect(left).toBeLessThan(right)
//   console.log(`${message}: ${left} < ${right}`)
// }

const optimizers = [nelderMead, gradientDescent, gradientDescentLineSearch, conjugateGradient]
const optimizerNames = ['Nelder Mead', 'Gradient Descent', 'Gradient Descent w/ Line Search', 'Conjugate Gradient']

describe('fmin', () => {
  it('himmelblau', () => {
    const x = 4.9515014216303825
    const y = 0.07301421370357275
    const params = { learnRate: 0.1 }

    function himmelblau(X: number[], fxprime?: number[]) {
      fxprime = fxprime || [0, 0]
      const [x, y] = X
      fxprime[0] = 2 * (x + 2 * y - 7) + 4 * (2 * x + y - 5)
      fxprime[1] = 4 * (x + 2 * y - 7) + 2 * (2 * x + y - 5)
      return Math.pow(x + 2 * y - 7, 2) + Math.pow(2 * x + y - 5, 2)
    }

    optimizers.forEach((optimizer, i) => {
      const solution = optimizer(himmelblau, [x, y], params)
      nearlyEqual(solution.fx, 0, SMALL, `himmelblau:${optimizerNames[i]}`)
    })
  })

  it('banana', () => {
    const x = 1.6084564160555601
    const y = -1.5980748860165477

    function banana(X: number[], fxprime?: number[]) {
      fxprime = fxprime || [0, 0]
      const [x, y] = X
      fxprime[0] = 400 * x * x * x - 400 * y * x + 2 * x - 2
      fxprime[1] = 200 * y - 200 * x * x
      return (1 - x) * (1 - x) + 100 * (y - x * x) * (y - x * x)
    }

    const params = { learnRate: 0.0003, maxIterations: 50000 }
    optimizers.forEach((optimizer, i) => {
      const solution = optimizer(banana, [x, y], params)
      nearlyEqual(solution.fx, 0, 1e-3, `banana:${optimizerNames[i]}`)
    })
  })

  it('quadratic1D', () => {
    const loss = (x: number[], xprime?: number[]) => {
      xprime = xprime || [0, 0]
      xprime[0] = 2 * (x[0] - 10)
      return (x[0] - 10) * (x[0] - 10)
    }

    const params = { learnRate: 0.5 }

    optimizers.forEach((optimizer, i) => {
      const solution = optimizer(loss, [0], params)
      nearlyEqual(solution.fx, 0, SMALL, `quadratic_1d:${optimizerNames[i]}`)
    })
  })

  it('nelderMead', () => {
    function loss(X: number[]) {
      const [x, y] = X
      return Math.sin(y) * x + Math.sin(x) * y + x * x + y * y
    }

    const solution = nelderMead(loss, [-3.5, 3.5])
    nearlyEqual(solution.fx, 0, SMALL, 'nelderMead')
  })

  it('conjugateGradientSolve', () => {
    // matyas function
    const A = [
      [0.52, -0.48],
      [-0.48, 0.52],
    ]
    const b = [0, 0]
    const initial = [-9.08, -7.83]
    const x = conjugateGradientSolve(A, b, initial)
    nearlyEqual(x[0], 0, SMALL, 'matyas.x')
    nearlyEqual(x[1], 0, SMALL, 'matyas.y')

    // booth's function
    const history: Array<{ x: number[]; p: number[]; alpha: number }> = []
    const A2 = [
      [10, 8],
      [8, 10],
    ]
    const b2 = [34, 38]
    const x2 = conjugateGradientSolve(A2, b2, initial, history)
    nearlyEqual(x2[0], 1, SMALL, 'booth.x')
    nearlyEqual(x2[1], 3, SMALL, 'booth.y')
    console.log(history)
  })
})
