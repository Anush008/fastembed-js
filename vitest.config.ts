import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    testTimeout: 180000,
    threads: false
  }
})