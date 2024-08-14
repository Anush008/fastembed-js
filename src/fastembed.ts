import fs, { PathLike } from "fs";
import https from "https";
import path from "path";
import Progress from "progress";
import tar from "tar";
import { AddedToken, Tokenizer } from "@anush008/tokenizers";
import * as ort from "onnxruntime-node";

export enum ExecutionProvider {
  CPU = "cpu",
  CUDA = "cuda",
  WebGL = "webgl",
  WASM = "wasm",
  XNNPACK = "xnnpack",
}

export enum EmbeddingModel {
  AllMiniLML6V2 = "fast-all-MiniLM-L6-v2",
  BGEBaseEN = "fast-bge-base-en",
  BGEBaseENV15 = "fast-bge-base-en-v1.5",
  BGESmallEN = "fast-bge-small-en",
  BGESmallENV15 = "fast-bge-small-en-v1.5",
  BGESmallZH = "fast-bge-small-zh-v1.5",
  MLE5Large = "fast-multilingual-e5-large",
}

interface InitOptions {
  model: EmbeddingModel;
  executionProviders: ExecutionProvider[];
  maxLength: number;
  cacheDir: string;
  showDownloadProgress: boolean;
}

interface ModelInfo {
  model: EmbeddingModel;
  dim: number;
  description: string;
}

type ModelInput = Record<string, ort.Tensor>;

function normalize(v: number[]): number[] {
  const norm = Math.sqrt(v.reduce((acc, val) => acc + val * val, 0));
  const epsilon = 1e-12;

  return v.map((val) => val / Math.max(norm, epsilon));
}

function getEmbeddings(
  data: number[],
  dimensions: [number, number, number]
): number[][] {
  const [x, y, z] = dimensions;

  return new Array(x).fill(undefined).map((_, index) => {
    const startIndex = index * y * z;
    const endIndex = startIndex + z;
    return data.slice(startIndex, endIndex);
  });
}

// Remove attention pooling
// Ref: https://github.com/qdrant/fastembed/commit/a335c8898f11042fdb311fce2dab3acf50c23011
// function create3DArray(
//   inputArray: number[],
//   dimensions: number[]
// ): number[][][] {
//   const totalElements = dimensions.reduce((acc, val) => acc * val, 1);

//   if (inputArray.length !== totalElements) {
//     throw new Error(
//       "Input array length does not match the specified dimensions."
//     );
//   }

//   const resultArray = Array.from({ length: dimensions[0] }, (_, i) =>
//     Array.from({ length: dimensions[1] }, (_, j) =>
//       Array.from(
//         { length: dimensions[2] },
//         (_, k) =>
//           inputArray[i * dimensions[1] * dimensions[2] + j * dimensions[2] + k]
//       )
//     )
//   );

//   return resultArray;
// }

abstract class Embedding {
  abstract listSupportedModels(): ModelInfo[];

  abstract embed(
    texts: string[],
    batchSize?: number
  ): AsyncGenerator<number[][], void, unknown>;

  abstract passageEmbed(
    texts: string[],
    batchSize: number
  ): AsyncGenerator<number[][], void, unknown>;

  abstract queryEmbed(query: string): Promise<number[]>;
}

export class FlagEmbedding extends Embedding {
  private constructor(
    private tokenizer: Tokenizer,
    private session: ort.InferenceSession,
    private model: EmbeddingModel
  ) {
    super();
  }

  static async init({
    model = EmbeddingModel.BGESmallENV15,
    executionProviders = [ExecutionProvider.CPU],
    maxLength = 512,
    cacheDir = "local_cache",
    showDownloadProgress = true,
  }: Partial<InitOptions> = {}) {
    const modelDir = await FlagEmbedding.retrieveModel(
      model,
      cacheDir,
      showDownloadProgress
    );

    const tokenizer = this.loadTokenizer(modelDir, maxLength);

    const modelPath = path.join(modelDir.toString(), "model_optimized.onnx");
    if (!fs.existsSync(modelPath)) {
      throw new Error(`Model file not found at ${modelPath}`);
    }
    const session = await ort.InferenceSession.create(modelPath, {
      executionProviders,
      graphOptimizationLevel: "all",
    });
    return new FlagEmbedding(tokenizer, session, model);
  }

  private static loadTokenizer(
    modelDir: fs.PathLike,
    maxLength: number
  ): Tokenizer {
    const tokenizerPath = path.join(modelDir.toString(), "tokenizer.json");
    if (!fs.existsSync(tokenizerPath)) {
      throw new Error(`Tokenizer file not found at ${tokenizerPath}`);
    }

    const configPath = path.join(modelDir.toString(), "config.json");
    if (!fs.existsSync(configPath)) {
      throw new Error(`Config file not found at ${configPath}`);
    }
    const config = JSON.parse(fs.readFileSync(configPath, "utf-8"));

    const tokenizerFilePath = path.join(
      modelDir.toString(),
      "tokenizer_config.json"
    );
    if (!fs.existsSync(tokenizerFilePath)) {
      throw new Error(`Tokenizer file not found at ${tokenizerFilePath}`);
    }
    const tokenizerConfig = JSON.parse(
      fs.readFileSync(tokenizerFilePath, "utf-8")
    );
    maxLength = Math.min(maxLength, tokenizerConfig["model_max_length"]);

    const tokensMapPath = path.join(
      modelDir.toString(),
      "special_tokens_map.json"
    );
    if (!fs.existsSync(tokensMapPath)) {
      throw new Error(`Tokens map file not found at ${tokensMapPath}`);
    }
    const tokensMap = JSON.parse(fs.readFileSync(tokensMapPath, "utf-8"));

    const tokenizer = Tokenizer.fromFile(tokenizerPath);

    tokenizer.setTruncation(maxLength);
    tokenizer.setPadding({
      maxLength,
      padId: config["pad_token_id"],
      padToken: tokenizerConfig["pad_token"],
    });

    for (let token of Object.values(tokensMap)) {
      if (typeof token === "string") {
        tokenizer.addSpecialTokens([token]);
      } else if (isAddedTokenMap(token)) {
        const addedToken = new AddedToken(token["content"], true, {
          singleWord: token["single_word"],
          leftStrip: token["lstrip"],
          rightStrip: token["rstrip"],
          normalized: token["normalized"],
        });
        tokenizer.addAddedTokens([addedToken]);
      }
    }
    return tokenizer;
  }

  private static async downloadFileFromGCS(
    outputFilePath: PathLike,
    model: string,
    showDownloadProgress: boolean = true
  ): Promise<PathLike> {
    if (fs.existsSync(outputFilePath)) {
      return outputFilePath;
    }

    const url = `https://storage.googleapis.com/qdrant-fastembed/${model}.tar.gz`;
    const fileStream = fs.createWriteStream(outputFilePath);

    return new Promise<PathLike>((resolve, reject) => {
      https
        .get(url, { headers: { "User-Agent": "Mozilla/5.0" } }, (response) => {
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
              `Downloading ${model} [:bar] :percent :etas`,
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

  private static async retrieveModel(
    model: EmbeddingModel,
    cacheDir: PathLike,
    showDownloadProgress: boolean = true
  ): Promise<PathLike> {
    if (!fs.existsSync(cacheDir)) {
      fs.mkdirSync(cacheDir, {
        mode: 0o777,
      });
    }

    const modelDir = path.join(cacheDir.toString(), model);

    if (fs.existsSync(modelDir)) {
      return modelDir;
    }

    const modelTarGz = path.join(cacheDir.toString(), `${model}.tar.gz`);
    await this.downloadFileFromGCS(modelTarGz, model, showDownloadProgress);
    await this.decompressToCache(modelTarGz, cacheDir);
    fs.unlinkSync(modelTarGz);
    return modelDir;
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

      const maxLength = idsArray[0].length;

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

      const inputs: ModelInput = {
        input_ids: batchInputIds,
        attention_mask: batchAttentionMask,
        token_type_ids: batchTokenTypeId,
      };

      // Exclude token_type_ids for MLE5Large
      if (this.model === EmbeddingModel.MLE5Large) {
        delete inputs.token_type_ids;
      }

      const output = await this.session.run(inputs);

      // Remove attention pooling
      // Ref: https://github.com/qdrant/fastembed/commit/a335c8898f11042fdb311fce2dab3acf50c23011

      // const lastHiddenState: number[][][] = create3DArray(
      //   output.last_hidden_state.data as unknown[] as number[],
      //   output.last_hidden_state.dims as number[]
      // );

      // const embeddings = lastHiddenState.map((layer, layerIdx) => {
      //   const weightedSum = layer.reduce((acc, tokenEmbedding, idx) => {
      //     const attentionWeight = maskArray[layerIdx][idx];
      //     return acc.map(
      //       (val, i) => val + tokenEmbedding[i] * Number(attentionWeight)
      //     );
      //   }, new Array(layer[0].length).fill(0));

      //   const inputMaskSum = maskArray[layerIdx].reduce(
      //     (acc, attentionWeight) => acc + Number(attentionWeight),
      //     0
      //   );

      //   return weightedSum.map((val) => val / Math.max(inputMaskSum, 1e-9));
      // });

      // const embeddings = lastHiddenState.map((sentence) => sentence[0]);

      const embeddings = getEmbeddings(
        output.last_hidden_state.data as unknown[] as number[],
        output.last_hidden_state.dims as [number, number, number]
      );

      yield embeddings.map(normalize);
    }
  }

  passageEmbed(texts: string[], batchSize: number = 256) {
    texts = texts.map((text) => `passage: ${text}`);
    return this.embed(texts, batchSize);
  }

  async queryEmbed(query: string): Promise<number[]> {
    return (await this.embed([`query: ${query}`]).next()).value![0];
  }

  listSupportedModels(): ModelInfo[] {
    return [
      {
        model: EmbeddingModel.BGESmallEN,
        dim: 384,
        description: "Fast English model",
      },
      {
        model: EmbeddingModel.BGESmallENV15,
        dim: 384,
        description: "v1.5 release of the fast, default English model",
      },
      {
        model: EmbeddingModel.BGEBaseEN,
        dim: 768,
        description: "Base English model",
      },
      {
        model: EmbeddingModel.BGEBaseENV15,
        dim: 768,
        description: "v1.5 release of Base English model",
      },
      {
        model: EmbeddingModel.BGESmallZH,
        dim: 512,
        description: "v1.5 release of the fast, Chinese model",
      },
      {
        model: EmbeddingModel.AllMiniLML6V2,
        dim: 384,
        description: "Sentence Transformer model, MiniLM-L6-v2",
      },
      {
        model: EmbeddingModel.MLE5Large,
        dim: 1024,
        description:
          "Multilingual model, e5-large. Recommend using this model for non-English languages",
      },
    ];
  }
}

interface AddedTokenMap {
  content: string;
  single_word: boolean;
  lstrip: boolean;
  rstrip: boolean;
  normalized: boolean;
}

function isAddedTokenMap(token: any): token is AddedTokenMap {
  return (
    typeof token === "object" &&
    token !== null &&
    "token" in token &&
    "single_word" in token &&
    "rstrip" in token &&
    "lstrip" in token &&
    "normalized" in token
  );
}
