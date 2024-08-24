#![deny(clippy::all)]

#[macro_use]
extern crate napi_derive;

use std::path::{Path, PathBuf};

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

const DEFAULT_MAX_LENGTH: i32 = 512;
const DEFAULT_CACHE_DIR: &str = ".fastembed_cache";
const DEFAULT_EMBEDDING_MODEL: JSEmbeddingModel = JSEmbeddingModel::BGESmallENV15;
const DEFAULT_SHOW_DOWNLOAD_PROGRESS: bool = true;

#[napi]
pub fn sum(a: i32, b: i32) -> i32 {
  a + b
}

#[napi(js_name = "TextEmbedding")]
pub struct JsTextEmbedding {
  embedding: TextEmbedding,
}

#[napi(js_name = "EmbeddingModel")]
pub enum JSEmbeddingModel {
  /// sentence-transformers/all-MiniLM-L6-v2
  AllMiniLML6V2,
  /// Quantized sentence-transformers/all-MiniLM-L6-v2
  AllMiniLML6V2Q,
  /// sentence-transformers/all-MiniLM-L12-v2
  AllMiniLML12V2,
  /// Quantized sentence-transformers/all-MiniLM-L12-v2
  AllMiniLML12V2Q,
  /// BAAI/bge-base-en-v1.5
  BGEBaseENV15,
  /// Quantized BAAI/bge-base-en-v1.5
  BGEBaseENV15Q,
  /// BAAI/bge-large-en-v1.5
  BGELargeENV15,
  /// Quantized BAAI/bge-large-en-v1.5
  BGELargeENV15Q,
  /// BAAI/bge-small-en-v1.5 - Default
  BGESmallENV15,
  /// Quantized BAAI/bge-small-en-v1.5
  BGESmallENV15Q,
  /// nomic-ai/nomic-embed-text-v1
  NomicEmbedTextV1,
  /// nomic-ai/nomic-embed-text-v1.5
  NomicEmbedTextV15,
  /// Quantized v1.5 nomic-ai/nomic-embed-text-v1.5
  NomicEmbedTextV15Q,
  /// sentence-transformers/paraphrase-MiniLM-L6-v2
  ParaphraseMLMiniLML12V2,
  /// Quantized sentence-transformers/paraphrase-MiniLM-L6-v2
  ParaphraseMLMiniLML12V2Q,
  /// sentence-transformers/paraphrase-mpnet-base-v2
  ParaphraseMLMpnetBaseV2,
  /// BAAI/bge-small-zh-v1.5
  BGESmallZHV15,
  /// intfloat/multilingual-e5-small
  MultilingualE5Small,
  /// intfloat/multilingual-e5-base
  MultilingualE5Base,
  /// intfloat/multilingual-e5-large
  MultilingualE5Large,
  /// mixedbread-ai/mxbai-embed-large-v1
  MxbaiEmbedLargeV1,
  /// Quantized mixedbread-ai/mxbai-embed-large-v1
  MxbaiEmbedLargeV1Q,
  /// Alibaba-NLP/gte-base-en-v1.5
  GTEBaseENV15,
  /// Quantized Alibaba-NLP/gte-base-en-v1.5
  GTEBaseENV15Q,
  /// Alibaba-NLP/gte-large-en-v1.5
  GTELargeENV15,
  /// Quantized Alibaba-NLP/gte-large-en-v1.5
  GTELargeENV15Q,
}

impl From<JSEmbeddingModel> for EmbeddingModel {
  fn from(model: JSEmbeddingModel) -> Self {
    match model {
      JSEmbeddingModel::AllMiniLML6V2 => EmbeddingModel::AllMiniLML6V2,
      JSEmbeddingModel::AllMiniLML6V2Q => EmbeddingModel::AllMiniLML6V2Q,
      JSEmbeddingModel::AllMiniLML12V2 => EmbeddingModel::AllMiniLML12V2,
      JSEmbeddingModel::AllMiniLML12V2Q => EmbeddingModel::AllMiniLML12V2Q,
      JSEmbeddingModel::BGEBaseENV15 => EmbeddingModel::BGEBaseENV15,
      JSEmbeddingModel::BGEBaseENV15Q => EmbeddingModel::BGEBaseENV15Q,
      JSEmbeddingModel::BGELargeENV15 => EmbeddingModel::BGELargeENV15,
      JSEmbeddingModel::BGELargeENV15Q => EmbeddingModel::BGELargeENV15Q,
      JSEmbeddingModel::BGESmallENV15 => EmbeddingModel::BGESmallENV15,
      JSEmbeddingModel::BGESmallENV15Q => EmbeddingModel::BGESmallENV15Q,
      JSEmbeddingModel::NomicEmbedTextV1 => EmbeddingModel::NomicEmbedTextV1,
      JSEmbeddingModel::NomicEmbedTextV15 => EmbeddingModel::NomicEmbedTextV15,
      JSEmbeddingModel::NomicEmbedTextV15Q => EmbeddingModel::NomicEmbedTextV15Q,
      JSEmbeddingModel::ParaphraseMLMiniLML12V2 => EmbeddingModel::ParaphraseMLMiniLML12V2,
      JSEmbeddingModel::ParaphraseMLMiniLML12V2Q => EmbeddingModel::ParaphraseMLMiniLML12V2Q,
      JSEmbeddingModel::ParaphraseMLMpnetBaseV2 => EmbeddingModel::ParaphraseMLMpnetBaseV2,
      JSEmbeddingModel::BGESmallZHV15 => EmbeddingModel::BGESmallZHV15,
      JSEmbeddingModel::MultilingualE5Small => EmbeddingModel::MultilingualE5Small,
      JSEmbeddingModel::MultilingualE5Base => EmbeddingModel::MultilingualE5Base,
      JSEmbeddingModel::MultilingualE5Large => EmbeddingModel::MultilingualE5Large,
      JSEmbeddingModel::MxbaiEmbedLargeV1 => EmbeddingModel::MxbaiEmbedLargeV1,
      JSEmbeddingModel::MxbaiEmbedLargeV1Q => EmbeddingModel::MxbaiEmbedLargeV1Q,
      JSEmbeddingModel::GTEBaseENV15 => EmbeddingModel::GTEBaseENV15,
      JSEmbeddingModel::GTEBaseENV15Q => EmbeddingModel::GTEBaseENV15Q,
      JSEmbeddingModel::GTELargeENV15 => EmbeddingModel::GTELargeENV15,
      JSEmbeddingModel::GTELargeENV15Q => EmbeddingModel::GTELargeENV15Q,
    }
  }
}

#[napi(object)]
pub struct JsInitOptions {
  pub model_name: Option<JSEmbeddingModel>,
  pub max_length: Option<i32>,
  pub cache_dir: Option<String>,
  pub show_download_progress: Option<bool>,
}

#[napi]
impl JsTextEmbedding {
  #[napi(constructor)]
  pub fn new(options: JsInitOptions) -> Self {
    let JsInitOptions {
      model_name,
      max_length,
      cache_dir,
      show_download_progress,
    } = options;

    let model_name: EmbeddingModel = model_name.unwrap_or(DEFAULT_EMBEDDING_MODEL).into();
    let cache_dir: String = cache_dir.unwrap_or(DEFAULT_CACHE_DIR.to_string());
    let cache_dir: PathBuf = Path::new(&cache_dir).to_path_buf();
    let max_length: usize = max_length.unwrap_or(DEFAULT_MAX_LENGTH) as usize;
    let show_download_progress = show_download_progress.unwrap_or(DEFAULT_SHOW_DOWNLOAD_PROGRESS);

    let options: InitOptions = InitOptions::new(model_name)
      .with_cache_dir(cache_dir)
      .with_max_length(max_length)
      .with_show_download_progress(show_download_progress);
    JsTextEmbedding {
      embedding: TextEmbedding::try_new(options).unwrap(),
    }
  }

  #[napi]
  pub fn embed(&self, texts: Vec<&str>, batch_size: Option<i32>) -> Vec<Vec<f32>> {
    self
      .embedding
      .embed(texts, batch_size.map(|x| x as usize))
      .unwrap()
  }
}
