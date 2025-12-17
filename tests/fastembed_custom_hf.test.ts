import { describe, expect, test } from "vitest";
import { EmbeddingModel, FlagEmbedding } from "../src";

describe("FastEmbed Custom HuggingFace Model Tests", () => {
  test(
    "loads dense model using enum (standard approach)",
    async () => {
      const model = await FlagEmbedding.init({
        model: EmbeddingModel.BGESmallENV15,
      });
      expect(model).toBeDefined();

      const embeddings = (await model.embed(["test"]).next()).value!;
      expect(embeddings).toBeDefined();
      expect(embeddings.length).toBe(1);
      expect(embeddings[0].length).toBe(384);
    },
    120000
  );

  test(
    "loads dense model using actual HF repo ID",
    async () => {
      const model = await FlagEmbedding.init({
        model: "BAAI/bge-small-en-v1.5",
      });
      expect(model).toBeDefined();

      const embeddings = (await model.embed(["test"]).next()).value!;
      expect(embeddings).toBeDefined();
      expect(embeddings.length).toBe(1);
      expect(embeddings[0].length).toBe(384);
    },
    120000
  );

  test(
    "loads sentence-transformers model",
    async () => {
      const model = await FlagEmbedding.init({
        model: "sentence-transformers/all-MiniLM-L6-v2",
      });
      expect(model).toBeDefined();

      const embeddings = (await model.embed(["hello world"]).next()).value!;
      expect(embeddings).toBeDefined();
      expect(embeddings.length).toBe(1);
      expect(embeddings[0].length).toBe(384);
    },
    120000
  );

  test(
    "embeddings are normalized",
    async () => {
      const model = await FlagEmbedding.init({
        model: EmbeddingModel.BGESmallENV15,
      });

      const embeddings = (await model.embed(["test"]).next()).value!;
      const magnitude = Math.sqrt(
        embeddings[0].reduce((acc, val) => acc + val * val, 0)
      );
      expect(magnitude).toBeCloseTo(1.0, 5);
    },
    120000
  );
});
