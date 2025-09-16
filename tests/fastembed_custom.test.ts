import fs from "fs";
import path from "path";
import { beforeAll, describe, expect, test } from "vitest";
import { EmbeddingModel, FlagEmbedding } from "../src";
async function downloadFile(url: string, dest: string) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Failed to fetch ${url}`);
  const buffer = await res.arrayBuffer();
  fs.mkdirSync(path.dirname(dest), { recursive: true });
  fs.writeFileSync(dest, Buffer.from(buffer));
}
describe("FastEmbed Custom Model Tests with Download files", () => {
  beforeAll(async () => {
    const files = [
      {
        repoUrl:
          "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model_quint8_avx2.onnx",
        outputPath: "../models/customs/mymodel.onnx",
      },
      {
        repoUrl:
          "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json",
        outputPath: "../models/customs/config.json",
      },
      {
        repoUrl:
          "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/special_tokens_map.json",
        outputPath: "../models/customs/special_tokens_map.json",
      },
      {
        repoUrl:
          "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer.json",
        outputPath: "../models/customs/tokenizer.json",
      },
      {
        repoUrl:
          "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json",
        outputPath: "../models/customs/tokenizer_config.json",
      },
    ];
    for (const element of files) {
      await downloadFile(
        element.repoUrl,
        path.resolve(__dirname, element.outputPath)
      );
    }
  });

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
    const pathModel = path.resolve(__dirname, "../models/customs");
    const flagEmbedding = await FlagEmbedding.init({
      model: EmbeddingModel.CUSTOM,
      modelAbsoluteDirPath: pathModel,
      modelName: "mymodel.onnx",
      maxLength: 512,
    });
    const embeddings = (await flagEmbedding.embed(["This is a test"]).next())
      .value!;
    expect(embeddings).toBeDefined();
    expect(embeddings.length).toBe(1);
  });

  test("FlagEmbedding embed batch", async () => {
    const pathModel = path.resolve(__dirname, "../models/customs");
    const flagEmbedding = await FlagEmbedding.init({
      model: EmbeddingModel.CUSTOM,
      modelAbsoluteDirPath: pathModel,
      modelName: "mymodel.onnx",

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
    const pathModel = path.resolve(__dirname, "../models/customs");
    const flagEmbedding = await FlagEmbedding.init({
      model: EmbeddingModel.CUSTOM,
      modelAbsoluteDirPath: pathModel,
      modelName: "mymodel.onnx",
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
    const pathModel = path.resolve(__dirname, "../models/customs");
    const flagEmbedding = await FlagEmbedding.init({
      model: EmbeddingModel.CUSTOM,
      modelAbsoluteDirPath: pathModel,
      modelName: "mymodel.onnx",
      maxLength: 512,
    });
    const embeddings = await flagEmbedding.queryEmbed("This is a test");
    expect(embeddings).toBeDefined();
    expect(embeddings.length).toBe(384);
  });

  test("FlagEmbedding passageEmbed", async () => {
    const pathModel = path.resolve(__dirname, "../models/customs");
    const flagEmbedding = await FlagEmbedding.init({
      model: EmbeddingModel.CUSTOM,
      modelAbsoluteDirPath: pathModel,
      modelName: "mymodel.onnx",

      maxLength: 512,
    });
    const embeddings = (
      await flagEmbedding.passageEmbed(["This is a test"]).next()
    ).value!;
    expect(embeddings).toBeDefined();
    expect(embeddings.length).toBe(1);
  });

  test("FlagEmbedding canonical values", async () => {
    const pathModel = path.resolve(__dirname, "../models/customs");
    const flagEmbedding = await FlagEmbedding.init({
      model: EmbeddingModel.CUSTOM,
      modelAbsoluteDirPath: pathModel,
      modelName: "mymodel.onnx",
      maxLength: 512,
    });
    const expected = [
      0.025276897475123405, 0.013033483177423477, 0.005586996208876371,
      0.04152565822005272, -0.018848471343517303, -0.05523142218589783,
      0.018086062744259834, -0.000535094877704978, -0.013765564188361168,
      -0.016923097893595695,
    ];

    const embeddings = (await flagEmbedding.embed(["hello world"]).next())
      .value!;
    console.log("embeddings", embeddings[0].slice(0, 10));
    expect(embeddings).toBeDefined();
    for (let i = 0; i < expected.length; i++) {
      expect(embeddings[0][i]).toBeCloseTo(expected[i], 3);
    }
  });
});
