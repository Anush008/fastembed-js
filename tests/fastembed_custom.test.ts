import path from "path";
import { expect, test } from "vitest";
import { EmbeddingModel, FlagEmbedding } from "../src";

test("Init EmbeddingModel", async () => {
  const pathModel = path.resolve(__dirname, "../models/customs");
  console.log("pathModel", pathModel);
  const model = await FlagEmbedding.init({
    model: EmbeddingModel.CUSTOM,
    modelAbsoluteDirPath: pathModel,
    modelName: "mymodel.onnx",
  });
  expect(model).toBeDefined();
});

test("FlagEmbedding embed", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,
    maxLength: 512,
  });
  const embeddings = (await flagEmbedding.embed(["This is a test"]).next())
    .value!;
  expect(embeddings).toBeDefined();
  expect(embeddings.length).toBe(1);
});

test("FlagEmbedding embed batch", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,

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
    expect(embeddings[0].length).toBe(384);
  }
});

test("FlagEmbedding embed small batch", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,
    maxLength: 512,
  });
  const embeddingsBatch = flagEmbedding.embed(
    ["This is a test", "Some text"],
    1
  );
  for await (const embeddings of embeddingsBatch) {
    expect(embeddings).toBeDefined();
    expect(embeddings.length).toBe(1);
    expect(embeddings[0].length).toBe(384);
  }
});

test("FlagEmbedding queryEmbed", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,
    maxLength: 512,
  });
  const embeddings = await flagEmbedding.queryEmbed("This is a test");
  expect(embeddings).toBeDefined();
  expect(embeddings.length).toBe(384);
});

test("FlagEmbedding passageEmbed", async () => {
  const flagEmbedding = await FlagEmbedding.init({
    model: EmbeddingModel.AllMiniLML6V2,

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
    model: EmbeddingModel.AllMiniLML6V2,
    maxLength: 512,
  });
  const expected = [
    0.02591, 0.00573, 0.01147, 0.03796, -0.0232, -0.0549, 0.01404, -0.0107,
    -0.0244, -0.01822,
  ];

  const embeddings = (await flagEmbedding.embed(["hello world"]).next()).value!;
  expect(embeddings).toBeDefined();
  for (let i = 0; i < expected.length; i++) {
    expect(embeddings[0][i]).toBeCloseTo(expected[i], 3);
  }
});
