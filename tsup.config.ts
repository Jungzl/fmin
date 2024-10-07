import { defineConfig } from 'tsup'

export default defineConfig({
  entry: {
    fmin: 'src/index.ts',
    // fmin_vis: 'src/visualizations/index.js',
  },
  format: ['cjs', 'esm'],
  treeshake: true,
  clean: true,
  dts: true,
  metafile: true,
})
