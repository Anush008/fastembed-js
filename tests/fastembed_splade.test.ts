import { expect, test } from "vitest";
import { SparseTextEmbedding, SparseEmbeddingModel } from "../src";

test("Init SparseEmbeddingModel", async () => {
  const model = await SparseTextEmbedding.init({
    model: SparseEmbeddingModel.SpladePPEnV1,
  });
  expect(model).toBeDefined();
}, 60000); // Increased timeout for model download

test("SparseTextEmbedding embed", async () => {
  const sparseEmbedding = await SparseTextEmbedding.init({
    model: SparseEmbeddingModel.SpladePPEnV1,
    maxLength: 512,
  });
  const embeddings = (await sparseEmbedding.embed(["This is a test"]).next())
    .value!;

  expect(embeddings).toBeDefined();
  expect(embeddings.length).toBe(1);

  // Verify sparse vector structure
  const sparseVector = embeddings[0];
  expect(Array.isArray(sparseVector)).toBe(true);
  expect(sparseVector.length).toBeGreaterThan(0);

  // Verify each element has tokenId and score
  sparseVector.forEach((item) => {
    expect(item).toHaveProperty("tokenId");
    expect(item).toHaveProperty("score");
    expect(typeof item.tokenId).toBe("number");
    expect(typeof item.score).toBe("number");
    expect(item.score).toBeGreaterThan(0);
  });

  // Verify sorted by score descending
  for (let i = 1; i < sparseVector.length; i++) {
    expect(sparseVector[i - 1].score).toBeGreaterThanOrEqual(
      sparseVector[i].score
    );
  }
}, 60000);

test("SparseTextEmbedding embed batch", async () => {
  const sparseEmbedding = await SparseTextEmbedding.init({
    model: SparseEmbeddingModel.SpladePPEnV1,
    maxLength: 512,
  });

  const texts = [
    "This is a test",
    "Some text",
    "Some more test",
    "This is a test",
    "Some text",
    "Some more test",
  ];

  const embeddingsBatch = sparseEmbedding.embed(texts);

  for await (const embeddings of embeddingsBatch) {
    expect(embeddings).toBeDefined();
    expect(embeddings.length).toBe(texts.length);

    // Verify each embedding is a sparse vector
    embeddings.forEach((sparseVector) => {
      expect(Array.isArray(sparseVector)).toBe(true);
      expect(sparseVector.length).toBeGreaterThan(0);
    });
  }
}, 60000);

test("SparseTextEmbedding embed small batch", async () => {
  const sparseEmbedding = await SparseTextEmbedding.init({
    model: SparseEmbeddingModel.SpladePPEnV1,
    maxLength: 512,
  });

  const texts = [
    "This is a test",
    "Some text",
    "Some more test",
    "This is a test",
    "Some text",
    "Some more test",
  ];

  const embeddingsBatch = sparseEmbedding.embed(texts, 1);

  for await (const embeddings of embeddingsBatch) {
    expect(embeddings).toBeDefined();
    expect(embeddings.length).toBe(1);
    expect(Array.isArray(embeddings[0])).toBe(true);
    expect(embeddings[0].length).toBeGreaterThan(0);
  }
}, 60000);

test("SparseTextEmbedding queryEmbed", async () => {
  const sparseEmbedding = await SparseTextEmbedding.init({
    model: SparseEmbeddingModel.SpladePPEnV1,
    maxLength: 512,
  });

  const embedding = await sparseEmbedding.queryEmbed("This is a test");

  expect(embedding).toBeDefined();
  expect(Array.isArray(embedding)).toBe(true);
  expect(embedding.length).toBeGreaterThan(0);

  // Verify structure
  embedding.forEach((item) => {
    expect(item).toHaveProperty("tokenId");
    expect(item).toHaveProperty("score");
  });
}, 60000);

test("SparseTextEmbedding passageEmbed", async () => {
  const sparseEmbedding = await SparseTextEmbedding.init({
    model: SparseEmbeddingModel.SpladePPEnV1,
    maxLength: 512,
  });

  const embeddings = (
    await sparseEmbedding.passageEmbed(["This is a test"]).next()
  ).value!;

  expect(embeddings).toBeDefined();
  expect(embeddings.length).toBe(1);
  expect(Array.isArray(embeddings[0])).toBe(true);
}, 60000);

test("SparseTextEmbedding sparsity", async () => {
  const sparseEmbedding = await SparseTextEmbedding.init({
    model: SparseEmbeddingModel.SpladePPEnV1,
    maxLength: 512,
  });

  const embedding = await sparseEmbedding.queryEmbed("hello world");

  expect(embedding).toBeDefined();

  // SPLADE typically produces ~100-200 non-zero dimensions out of 30522
  // Verify it's actually sparse
  expect(embedding.length).toBeLessThan(1000); // Should be much less than vocab size
  expect(embedding.length).toBeGreaterThan(10); // But should have some meaningful tokens

  console.log(`Sparse vector has ${embedding.length} non-zero dimensions`);
  console.log(
    `Top 5 tokens: ${embedding
      .slice(0, 5)
      .map((e) => `${e.tokenId}:${e.score.toFixed(3)}`)
      .join(", ")}`
  );
}, 60000);