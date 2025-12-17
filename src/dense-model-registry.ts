export interface DenseModelMetadata {
  repoId: string; // Actual provider repo
  gcsUrl: string; // GCS fallback URL
  onnxFilePath: string; // Path within HF repo: "onnx/model.onnx"
  dim: number;
  description: string;
  requiresTokenTypeIds: boolean;
}

export const DENSE_MODEL_REGISTRY: Record<string, DenseModelMetadata> = {
  "sentence-transformers/all-MiniLM-L6-v2": {
    repoId: "sentence-transformers/all-MiniLM-L6-v2",
    gcsUrl:
      "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz",
    onnxFilePath: "onnx/model.onnx",
    dim: 384,
    description: "Sentence Transformer model, MiniLM-L6-v2",
    requiresTokenTypeIds: true,
  },
  "BAAI/bge-base-en": {
    repoId: "BAAI/bge-base-en",
    gcsUrl:
      "https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en.tar.gz",
    onnxFilePath: "onnx/model.onnx",
    dim: 768,
    description: "Base English model from BAAI",
    requiresTokenTypeIds: true,
  },
  "BAAI/bge-base-en-v1.5": {
    repoId: "BAAI/bge-base-en-v1.5",
    gcsUrl:
      "https://storage.googleapis.com/qdrant-fastembed/fast-bge-base-en-v1.5.tar.gz",
    onnxFilePath: "onnx/model.onnx",
    dim: 768,
    description: "v1.5 release of Base English model",
    requiresTokenTypeIds: true,
  },
  "BAAI/bge-small-en": {
    repoId: "BAAI/bge-small-en",
    gcsUrl:
      "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en.tar.gz",
    onnxFilePath: "onnx/model.onnx",
    dim: 384,
    description: "Small English model from BAAI",
    requiresTokenTypeIds: true,
  },
  "BAAI/bge-small-en-v1.5": {
    repoId: "BAAI/bge-small-en-v1.5",
    gcsUrl:
      "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz",
    onnxFilePath: "onnx/model.onnx",
    dim: 384,
    description: "v1.5 release of small English model",
    requiresTokenTypeIds: true,
  },
  "BAAI/bge-small-zh-v1.5": {
    repoId: "BAAI/bge-small-zh-v1.5",
    gcsUrl:
      "https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-zh-v1.5.tar.gz",
    onnxFilePath: "onnx/model.onnx",
    dim: 512,
    description: "v1.5 Chinese small model",
    requiresTokenTypeIds: true,
  },
  "intfloat/multilingual-e5-large": {
    repoId: "intfloat/multilingual-e5-large",
    gcsUrl:
      "https://storage.googleapis.com/qdrant-fastembed/fast-multilingual-e5-large.tar.gz",
    onnxFilePath: "onnx/model.onnx",
    dim: 1024,
    description: "Multilingual model, e5-large",
    requiresTokenTypeIds: false,
  },
};
