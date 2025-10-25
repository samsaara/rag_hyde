import gc
import json
import torch
import pymupdf, pymupdf4llm

from ast import literal_eval
from pathlib import Path
from datasets import Dataset
from collections import defaultdict
from docling.document_converter import DocumentConverter

from src.config import log
from pathlib import Path


def empty_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.mps.is_available():
        torch.mps.empty_cache()


def extract_text(file, markdown=False, backend="pymupdf", **kwargs):
    if backend == "pymupdf":
        if not markdown:
            with pymupdf.open(file, filetype="pdf") as doc:
                return "\n".join(page.get_text(**kwargs) for page in doc)
        else:
            log.debug("\n\n using pymupdf4llm \n\n")
            return pymupdf4llm.to_markdown(file, show_progress=True, **kwargs)

    elif backend == "docling":
        converter = DocumentConverter(allowed_formats=["pdf"])
        doc = converter.convert(file, **kwargs).document
        res = doc.export_to_markdown() if markdown else doc.export_to_text()
        del converter, doc
        empty_cache()
        return res


def _load_pdf_sync(file, markdown=True, fast=False, **kwargs):
    """Synchronous PDF loading function for thread pool execution"""
    text = extract_text(
        file,
        markdown,
        backend="docling"
        if ((not fast) and (torch.cuda.is_available() or torch.mps.is_available()))
        else "pymupdf",
        **kwargs,
    )

    return (Path(file).stem, text)


def load_pdfs(files, markdown=True, fast_extract=False, **kwargs):
    """
    Load multiple PDF files

    Args:
        files: PDF filepaths
        markdown: whether to extract text in markdown
        fast_extract: whether to use pymupdf to extract text in markdown
    Returns:
        list: List of tuples containing (filename, extracted_text)
    """
    # # Use ThreadPoolExecutor to run synchronous operations concurrently
    # loop = asyncio.get_event_loop()

    # # Create executor with limited workers
    # with ThreadPoolExecutor(max_workers=max_concurrence) as executor:
    #     # Submit all PDF processing tasks
    #     futures = [
    #         loop.run_in_executor(executor, _load_pdf_sync, file, markdown, fast_extract, **kwargs) for file in files if file is not None
    #     ]

    #     results = await asyncio.gather(*futures, return_exceptions=True)

    # valid_results = [result for result in results if not isinstance(result, Exception)]

    # log.debug(f"Successfully processed {len(valid_results)} out of {len(files)} PDFs")
    # return valid_results

    results = []
    for file in files:
        results.append(_load_pdf_sync(file, markdown, fast_extract, **kwargs))
    return results


def find_think_tag_in_each_row(tensor):
    # look for `</think>` tag
    res = dict((tensor == 151668).nonzero().tolist())
    if not res:
        return [0] * len(tensor)
    idxs = []
    for idx in range(len(tensor)):
        idxs.append(res.get(idx, -1))
    return [x + 1 for x in idxs]


def build_corpus(pdfs, text_splitter, **load_pdf_kwargs):
    texts = load_pdfs(pdfs, **load_pdf_kwargs)
    corpus_with_meta = []
    _id = 0
    for file_name, raw_text in texts:
        chunks = text_splitter.split_text(raw_text)
        for idx, chunk in enumerate(chunks):
            corpus_with_meta.append(
                {
                    "id": _id,
                    "file": Path(file_name).stem,
                    "chunk_id": idx,
                    "chunk": chunk,
                }
            )
            _id += 1
    return Dataset.from_list(corpus_with_meta)


def reciprocal_rank_fusion(indices, top_k=3, denom=50):
    scores = defaultdict(int)
    for row in indices:
        for rank, idx in enumerate(row):
            scores[idx] += 1 / (rank + denom)
    results = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [idx for idx, _ in results]


def clean_rewrite_resp(resp):
    try:
        resp = json.loads(resp)  # Parse JSON
    except json.JSONDecodeError:
        try:
            resp = literal_eval(resp)  # Fallback parse
        except Exception:
            pass  # Keep resp as-is if both fail

    # Ensure resp is a string before strip and slicing
    if isinstance(resp, str):
        resp = resp.strip()
        if resp:
            start = resp.find("{")
            if start != -1:
                end = resp[::-1].find("}")
                if end != -1:
                    resp = resp[start : len(resp) - end]
                    return clean_rewrite_resp(resp)
    return resp
