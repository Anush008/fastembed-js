# ⚡️ FastEmbed-JS

FastEmbed-JS is a Typescript/NodeJS implementation of [@Qdrant/fastembed](https://github.com/qdrant/fastembed).

* Supports CommonJS and ESM.
* Uses [@anush008/tokenizers](https://github.com/Anush008/tokenizers) multi-arch native bindings for [@huggingface/tokenizers](https://github.com/huggingface/tokenizers).
* Supports batch embedddings with generators.

The default embedding supports "query" and "passage" prefixes for the input text. The default model is Flag Embedding, which is top of the [MTEB](https://huggingface.co/spaces/mteb/leaderboard) leaderboard.

## 🤖 Models

- [**BAAI/bge-base-en-v1.5**](https://huggingface.co/BAAI/bge-base-en-v1.5)
- [**BAAI/bge-small-en-v1.5**](https://huggingface.co/BAAI/bge-small-en-v1.5) - Default
- [**sentence-transformers/all-MiniLM-L12-v2**](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2)



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
    model: EmbeddingModel.BGEBaseEN
});

let documents = [
    "passage: Hello, World!",
    "query: Hello, World!",
    "passage: This is an example passage.",
    // You can leave out the prefix but it's recommended
    "fastembed-js is licensed under MIT" 
];

const embeddings = embeddingModel.embed(documents, 2); //Optional batch size. Defaults to 256

for await (const batch of embeddings) {
    // batch is list of Float32 embeddings(number[][]) with length 2
    console.log(batch);
}

```

#### Supports passage and query embeddings for more accurate results
```ts
const embeddings = embeddingModel.passageEmbed(listOfLongTexts, 10); //Optional batch size. Defaults to 256

for await (const batch of embeddings) {
    // batch is list of Float32 passage embeddings(number[][]) with length 10
    console.log(batch);
}

const queryEmbeddings: number[] = await embeddingModel.queryEmbed(userQuery);
console.log(queryEmbeddings)

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
