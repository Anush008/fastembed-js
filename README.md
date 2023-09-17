# ⚡️ FastEmbed-JS

FastEmbed-JS is a Typescript/NodeJS implementation of [@Qdrant/fastembed](https://github.com/qdrant/fastembed).

* Supports CommonJS and ESM.
* Uses [@anush008/tokenizers](https://github.com/Anush008/tokenizers) multi-arch native bindings for [@huggingface/tokenizers](https://github.com/huggingface/tokenizers).
* Supports batch embedddings with generators.

The default embedding supports "query" and "passage" prefixes for the input text. The default model is Flag Embedding, which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard.

1. Light
    - Quantized model weights
    - ONNX Runtime for inference
    - No hidden dependencies via Huggingface Transformers

2. Accuracy/Recall
    - Better than OpenAI Ada-002
    - Default is Flag Embedding, which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard

3. Fast
    - Lot faster for batches!
    - ONNX Runtime allows you to use dedicated runtimes for even higher throughput and lower latency 

## 🚀 Installation

To install the FastEmbed library, npm works: 

```bash
npm install fastembed
```

## 📖 Usage

```js
import { EmbeddingModel, FlagEmbedding } from "fastembed";
// For CommonJS
// const { EmbeddingModel, FlagEmbedding } = require("fastembed)

const embeddingModel = await FlagEmbedding.init({
    modelName: EmbeddingModel.BGEBaseEN
});

let documents = [
    "passage: Hello, World!",
    "query: Hello, World!", // these are two different embedding
    "passage: This is an example passage.",
    //# You can leave out the prefix but it's recommended
    "fastembed is supported by and maintained by Qdrant." 
];

const embeddings = embeddingModel.embed(documents, 2); //Optional batch size. Defaults to 256

for await (const batch of embeddings) {
    // batch is list of Float32 embeddings
    console.log(batch);
}

```

## 🚒 Under the hood

### Why fast?

It's important we justify the "fast" in FastEmbed. FastEmbed is fast because:

1. Quantized model weights
2. ONNX Runtime which allows for inference on CPU, GPU, and other dedicated runtimes

### Why light?
1. No hidden dependencies via Huggingface Transformers

### Why accurate?
1. Better than OpenAI Ada-002
2. Top of the Embedding leaderboards e.g. [MTEB](https://huggingface.co/spaces/mteb/leaderboard)