import fs, { PathLike } from "fs";
import https from "https";
import path from "path";
import Progress from "progress";
import tar from "tar";
import { Tokenizer } from "@anush008/tokenizers";
import * as ort from "onnxruntime-node";

export enum ExecutionProvider {
  CPU = "cpu",
  CUDA = "cuda",
  WebGL = "webgl",
  WASM = "wasm",
  XNNPACK = "xnnpack",
}

export enum EmbeddingModel {
  AllMiniLML6V2 = "sentence-transformers/all-MiniLM-L6-v2",
  BGEBaseEN = "BAAI/bge-base-en",
  BGESmallEN = "BAAI/bge-small-en",
}

export interface InitOptions {
  modelName: EmbeddingModel;
  executionProviders: ExecutionProvider[];
  maxLength: number;
  cacheDir: string;
  showDownloadProgress: boolean;
}

function normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((acc, val) => acc + val * val, 0));
  const epsilon = 1e-12;

  return v.map((val) => val / (norm + epsilon));
}

function create3DArray(
  inputArray: number[],
  dimensions: number[]
): number[][][] {
  const totalElements = dimensions.reduce((acc, val) => acc * val, 1);

  if (inputArray.length !== totalElements) {
    throw new Error(
      "Input array length does not match the specified dimensions."
    );
  }

  const resultArray = Array.from({ length: dimensions[0] }, (_, i) =>
    Array.from({ length: dimensions[1] }, (_, j) =>
      Array.from(
        { length: dimensions[2] },
        (_, k) =>
          inputArray[i * dimensions[1] * dimensions[2] + j * dimensions[2] + k]
      )
    )
  );

  return resultArray;
}

abstract class Embedding {
  abstract embed(
    texts: string[],
    batchSize?: number
  ): AsyncGenerator<number[][], void, unknown>;

  private static async downloadFileFromGCS(
    url: string,
    outputFilePath: PathLike,
    modelName: string,
    showDownloadProgress: boolean = true
  ): Promise<PathLike> {
    if (fs.existsSync(outputFilePath)) {
      return outputFilePath;
    }

    const fileStream = fs.createWriteStream(outputFilePath);

    return new Promise<PathLike>((resolve, reject) => {
      https
        .get(url, { headers: { "User-Agent": "Mozilla/5.0" } }, (response) => {
          if (response.statusCode === 403) {
            reject(
              new Error(
                "Authentication Error: You do not have permission to access this resource."
              )
            );
            return;
          }

          const totalSizeInBytes = parseInt(
            response.headers["content-length"] || "0",
            10
          );

          if (totalSizeInBytes === 0) {
            console.warn(
              `Warning: Content-length header is missing or zero in the response from ${url}.`
            );
          }

          if (showDownloadProgress) {
            const progressBar = new Progress(
              `Downloading ${modelName} [:bar] :percent :etas`,
              {
                complete: "=",
                width: 20,
                total: totalSizeInBytes,
              }
            );

            response.on("data", (chunk) => {
              progressBar.tick(chunk.length, { speed: "N/A" });
            });
          }
          response.on("error", (error) => {
            reject(error);
          });

          response.pipe(fileStream);

          fileStream.on("finish", () => {
            fileStream.close();
            resolve(outputFilePath);
          });

          fileStream.on("error", (error) => {
            reject(error);
          });
        })
        .on("error", (error) => {
          fs.unlink(outputFilePath, () => {
            reject(error);
          });
        });
    });
  }

  private static async decompressToCache(
    targzPath: PathLike,
    cacheDir: PathLike
  ) {
    // Implementation for decompressing a .tar.gz file to a cache directory
    if (path.extname(targzPath.toString()) === ".gz") {
      await tar.x({
        file: targzPath,
        // @ts-ignore
        cwd: cacheDir,
      });
    } else {
      throw new Error(`Unsupported file extension: ${targzPath}`);
    }
  }

  protected static async retrieveModel(
    model: EmbeddingModel,
    cacheDir: PathLike,
    showDownloadProgress: boolean = true
  ): Promise<PathLike> {
    if (!model.includes("/")) {
      throw new Error(
        "model_name must be in the format <org>/<model> e.g. BAAI/bge-base-en"
      );
    }

    let modelName = model.split("/").pop()!;
    let fastModelName = `fast-${modelName}`;

    if (!fs.existsSync(cacheDir)) {
      fs.mkdirSync(cacheDir, {
        mode: 0o777,
      });
    }

    let modelDir = path.join(cacheDir.toString(), fastModelName);

    if (fs.existsSync(modelDir)) {
      return modelDir;
    }

    let modelTarGz = path.join(cacheDir.toString(), `${fastModelName}.tar.gz`);
    await this.downloadFileFromGCS(
      `https://storage.googleapis.com/qdrant-fastembed/${fastModelName}.tar.gz`,
      modelTarGz,
      modelName,
      showDownloadProgress
    );
    await this.decompressToCache(modelTarGz, cacheDir);
    fs.unlinkSync(modelTarGz);
    return modelDir;
  }

  passageEmbed(texts: string[], batchSize: number = 256) {
    texts = texts.map((text) => `passage: ${text}`);
    return this.embed(texts, batchSize);
  }

  async queryEmbed(query: string) {
    return (await this.embed([`query: ${query}`]).next()).value!;
  }
}

export class FlagEmbedding extends Embedding {
  private constructor(
    private tokenizer: Tokenizer,
    private model: ort.InferenceSession
  ) {
    super();
  }
  static async init({
    modelName = EmbeddingModel.BGESmallEN,
    executionProviders = [ExecutionProvider.CPU],
    maxLength = 512,
    cacheDir = "local_cache",
    showDownloadProgress = true,
  }: Partial<InitOptions> = {}) {
    let modelDir = await Embedding.retrieveModel(modelName, cacheDir, showDownloadProgress);
    let tokenizerPath = path.join(modelDir.toString(), "tokenizer.json");
    if (!fs.existsSync(tokenizerPath)) {
      throw new Error(`Tokenizer file not found at ${tokenizerPath}`);
    }
    let modelPath = path.join(modelDir.toString(), "model_optimized.onnx");
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Model file not found at ${modelPath}`);
    }

    let tokenizer = Tokenizer.fromFile(tokenizerPath);
    tokenizer.setTruncation(maxLength!);
    tokenizer.setPadding({
      padToken: "[PAD]",
      maxLength,
    });
    let model = await ort.InferenceSession.create(modelPath, {
      executionProviders,
    });
    return new FlagEmbedding(tokenizer, model);
  }

  async *embed(textStrings: string[], batchSize: number = 256) {
    for (let i = 0; i < textStrings.length; i += batchSize) {
      const batchTexts = textStrings.slice(i, i + batchSize);

      const encodedTexts = await Promise.all(
        batchTexts.map((textString) => this.tokenizer.encode(textString))
      );

      const idsArray: bigint[][] = [];
      const maskArray: bigint[][] = [];
      const typeIdsArray: bigint[][] = [];

      encodedTexts.forEach((text) => {
        const ids = text.getIds().map(BigInt);
        const mask = text.getAttentionMask().map(BigInt);
        const typeIds = text.getTypeIds().map(BigInt);

        idsArray.push(ids);
        maskArray.push(mask);
        typeIdsArray.push(typeIds);
      });

      const maxLength = idsArray.reduce((max, arr) => Math.max(max, arr.length), 0);
      
      const batchInputIds = new ort.Tensor(
        "int64",
        idsArray.flat() as unknown as number[],
        [batchTexts.length, maxLength]
      );
      const batchAttentionMask = new ort.Tensor(
        "int64",
        maskArray.flat() as unknown as number[],
        [batchTexts.length, maxLength]
      );
      const batchTokenTypeId = new ort.Tensor(
        "int64",
        typeIdsArray.flat() as unknown as number[],
        [batchTexts.length, maxLength]
      );

      const output = await this.model.run({
        input_ids: batchInputIds,
        attention_mask: batchAttentionMask,
        token_type_ids: batchTokenTypeId,
      });

      const lastHiddenState: number[][][] = create3DArray(
        output.last_hidden_state.data as unknown[] as number[],
        output.last_hidden_state.dims as number[]
      );

      const embeddings = lastHiddenState.map((layer, layerIdx) => {
        const weightedSum = layer.reduce((acc, tokenEmbedding, idx) => {
          const attentionWeight = maskArray[layerIdx][idx];
          return acc.map(
            (val, i) => val + tokenEmbedding[i] * Number(attentionWeight)
          );
        }, new Array(layer[0].length).fill(0));

        const inputMaskSum = maskArray[layerIdx].reduce(
          (acc, attentionWeight) => acc + Number(attentionWeight),
          0
        );

        return weightedSum.map((val) => val / (inputMaskSum + 1e-9));
      });

      yield embeddings.map(normalize);
    }
  }
}
