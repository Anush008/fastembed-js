import { expect, test } from "vitest";
import { FlagEmbedding, EmbeddingModel } from "../src";

test("Init EmbeddingModel", async () => {
  const model = await FlagEmbedding.init({
    model: EmbeddingModel.BGEBaseENV15,
  });
  expect(model).toBeDefined();
});

test("FlagEmbedding embed", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.BGEBaseENV15,
    maxLength: 512,
  });
  const embeddings = (await flagEmbedding.embed(["This is a test"]).next())
    .value!;
  expect(embeddings).toBeDefined();
  expect(embeddings.length).toBe(1);
});

test("FlagEmbedding embed batch", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.BGEBaseENV15,

    maxLength: 512,
  });
  const embeddingsBatch = flagEmbedding.embed([
    "This is a test",
    "Some text",
    "Some more test",
    "This is a test",
    "Some text",
    "Some more test",
  ]);
  for await (const embeddings of embeddingsBatch) {
    expect(embeddings).toBeDefined();
    expect(embeddings.length).toBe(6);
    expect(embeddings[0].length).toBe(768);
  }
});

test("FlagEmbedding embed Base batch", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.BGEBaseENV15,
    maxLength: 512,
  });
  const embeddingsBatch = flagEmbedding.embed(
    [
      "This is a test",
      "Some text",
      "Some more test",
      "This is a test",
      "Some text",
      "Some more test",
    ],
    1
  );
  for await (const embeddings of embeddingsBatch) {
    expect(embeddings).toBeDefined();
    expect(embeddings.length).toBe(1);
    expect(embeddings[0].length).toBe(768);
  }
});

test("FlagEmbedding queryEmbed", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.BGEBaseENV15,
    maxLength: 512,
  });
  const embeddings = await flagEmbedding.queryEmbed("This is a test");
  expect(embeddings).toBeDefined();
  expect(embeddings.length).toBe(768);
});

test("FlagEmbedding passageEmbed", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.BGEBaseENV15,

    maxLength: 512,
  });
  const embeddings = (
    await flagEmbedding.passageEmbed(["This is a test"]).next()
  ).value!;
  expect(embeddings).toBeDefined();
  expect(embeddings.length).toBe(1);
});

test("FlagEmbedding canonical values", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.BGEBaseENV15,
    maxLength: 512,
  });
  const expected = [0.01129394, 0.05493144, 0.02615099, 0.00328772, 0.02996045];

  const embeddings = (await flagEmbedding.embed(["hello world"]).next()).value!;
  expect(embeddings).toBeDefined();
  for (let i = 0; i < expected.length; i++) {
    expect(embeddings[0][i]).toBeCloseTo(expected[i], 3);
  }
});
