// need some basic operations on vectors, rather than adding a dependency,
// just define here
export function zeros(x: number): number[] {
  return new Array(x).fill(0)
}

export function zerosM(x: number, y: number): number[][] {
  return zeros(x).map(() => zeros(y))
}

export function dot(a: number[], b: number[]): number {
  let ret = 0
  for (let i = 0; i < a.length; ++i) {
    ret += a[i] * b[i]
  }
  return ret
}

export function norm2(a: number[]): number {
  return Math.sqrt(dot(a, a))
}

export function scale(ret: number[], value: number[], c: number): void {
  for (let i = 0; i < value.length; ++i) {
    ret[i] = value[i] * c
  }
}

export function weightedSum(ret: number[], w1: number, v1: number[], w2: number, v2: number[]): void {
  for (let j = 0; j < ret.length; ++j) {
    ret[j] = w1 * v1[j] + w2 * v2[j]
  }
}

export function gemv(output: number[], A: number[][], x: number[]): void {
  for (let i = 0; i < output.length; ++i) {
    output[i] = dot(A[i], x)
  }
}
