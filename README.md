# RAG HYDE

Building a minimal yet powerful Retrieval-Augmented Generation (RAG) workflow inspired by the techniques in the articles already shared.

## Setup

- Install [pixi](https://pixi.sh/latest/installation/)
- Clone this repo
- For Linux: `pixi install -e linux`
- For Mac: `pixi install -e mac`


### Prerequisites on Mac

To install with GPU accelerated libraries on Mac with Apple Silicon processors, here's what's needed:

- gcc and [zlib](https://formulae.brew.sh/formula/zlib) installed (either xcode/developer tools/brew etc.)
  - if gcc, is not found, you might be prompted to install developer tools or something similar.
  - If installed zlib via homebrew, make sure the library is in path. If not, run `ZLIB_ROOT=$(brew --prefix zlib) pixi install -e mac`

_(I had limited access to a mac mini M1 and successfully tested it (via notebook) after the above steps.)_

## Run


The code for each of the tasks can be run in two ways: (a) CLI (b) gradio.

`MODEL_COMBOS` in [config.py](src/config.py) provides multiple variants of embedding & generative LLM model combinations keeping in mind the host system's limitations & capabilities. Pass it via CLI or set them in gradio UI, if you wish to change from the default.

For gradio:

```bash
PYTHONPATH='.' pixi run -e [linux|mac] gradio app.py
```

To run from CLI or for help or to tweak parameters:
```bash
PYTHONPATH='.' pixi run -e [linux|mac] python src/main.py [--help]
```

### Debug

logs will be stored in `logs.log` realtime for debugging & monitoring.

---

## Tips/Observations:

- Extracting text as a markdown greatly preserved the structure and continuity of the text. This resulted in better logical chunking which in turn led to better embeddings and as a consequence, better search results.

- Reading the document via `docling` extracted more and correct text compared to `pymupdf4llm` but at a bit of an expense of speed. It is enabled by default for prioritising accuracy.
  - This proved esp. useful in extracting data containing lots of tables spread over multiple pages.
  - You can pass `--fast-extract` from CLI or tick a box via gradio UI to use pymupdf instead.

- Increasing the model size (coupled with correct text extraction in markdown) greatly improved performance. The Qwen3 models very much adhered to instructions but the smaller variants instead of hallucinating simply fell back to saying _'I don't know'_ (as per instructions). The `4B` variant understood the user intent which sometimes was vague and yet managed to give relevant results. The base variant is huge and it wouldn't have been fit and run fast enough on a consumer grade laptop GPU. Loading the `AWQ` variant of it helped as it occupied substantially less memory compared to the original without much loss in performance.

  - This model also showed great multilingual capabilities. User can upload document in one language and ask questions in another. Or they could upload multilingual documents and ask multilingual queries. For the demo, I tested mostly in English & German.

- The data is now stored in datasets format that allows for better storage & scaling (arrow) along with indexing (FAISS) for querying.

---

## Limitations / Known Issues

- Even though `docling` with mostly default options proved to be better than `pymupdf4llm` to extract text, it's not perfect everytime. There're instances where _pymupdf_ extracted text from an embedded image inside a PDF better than docling. However, docling is highly configurable and allows for deep customization via 'pipelines'. And it also comes with a very permissive license for commercial use compared to PyMuPDF.
  - docling comes with `easyocr` by default for text OCR. It's not powerful enough compared to _tesseract_ or similar models. But since installing the latter and linking it with docling involves touching system config, it's not pursued.

- When user uploads multiple PDFs, we can improve load times by reading them asynchronously. Attempts to do that with `docling` sometimes resulted in pages with ordering different than the original. So it's dropped for the demo. More investigation is needed later.


## Next Steps

- Checkout [EmbeddingGemma](https://huggingface.co/blog/embeddinggemma) for embeddings
- Checkout [fastembed](https://github.com/qdrant/fastembed) to generate embeddings faster
- Improve text extraction via docling pipeline
- Checkout `GGUF` models for CPU Inferencing
