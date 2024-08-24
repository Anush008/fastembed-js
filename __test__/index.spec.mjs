import test from 'ava'

import { sum, TextEmbedding, EmbeddingModel } from '../index.js'

test('sum from native', (t) => {
  t.is(sum(1, 2), 3)
})

test('embedding with TextEmbedding - No batch size', (t) => {
  const model = new TextEmbedding(
    {
      modelName: EmbeddingModel.AllMiniLML6V2,
      showDownloadProgress: true
    }
  )

  const embeddings = model.embed(["Hello people", "How are you?"]);
  t.is(embeddings.length, 2)
  t.is(embeddings[0].length, 384)
})

test('embedding with TextEmbedding - With batch size', (t) => {
  const model = new TextEmbedding(
    {
      modelName: EmbeddingModel.AllMiniLML6V2,
      showDownloadProgress: true
    }
  )

  const embeddings = model.embed(["Hello people", "How are you?"], 1);
  t.is(embeddings.length, 2)
  t.is(embeddings[0].length, 384)
})