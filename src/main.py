import string
import faiss
import yaml
import re
import sys
import torch
import click

from time import time
from random import shuffle
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_text_splitters import MarkdownTextSplitter

from src.config import PROMPTS_FILEPATH, MODEL_COMBOS, log
from src.utils import (
    empty_cache,
    find_think_tag_in_each_row,
    build_corpus,
    reciprocal_rank_fusion,
    clean_rewrite_resp,
)


def generate_text(
    tokenizer, model, user_prompts, system_prompt=None, model_name="", **llm_kwargs
):
    assert model_name, "pass on generative model name"
    if system_prompt is None or "":
        system_prompt = "You are a helpful assistant."

    if isinstance(user_prompts, str):
        user_prompts = [user_prompts]

    messages = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for user_prompt in user_prompts
    ]

    if "mlx" in model_name.lower() and sys.platform == "darwin":
        from mlx_lm import generate

        texts = [
            tokenizer.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for message in messages
        ]
        responses = [
            generate(
                model,
                tokenizer,
                prompt=text,
                verbose=False,
                max_tokens=llm_kwargs.pop("max_new_tokens", 32768),
            )
            for text in texts
        ]
    else:
        texts = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        model_inputs = tokenizer(
            texts, return_tensors="pt", truncation=True, padding=True
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=llm_kwargs.pop("max_new_tokens", 32768),
                temperature=llm_kwargs.pop("temperature", 0.7),
                top_p=llm_kwargs.pop("top_p", 0.8),
                top_k=llm_kwargs.pop("top_k", 20),
                min_p=llm_kwargs.pop("min_p", 0),
                **llm_kwargs,
            )

        output_ids = generated_ids[:, model_inputs.input_ids.shape[1] :]
        idxs = find_think_tag_in_each_row(output_ids)
        thinking_contents = [
            tokenizer.decode(output_ids[i][:idx], skip_special_tokens=True).strip("\n")
            for i, idx in enumerate(idxs)
        ]
        contents = [
            tokenizer.decode(output_ids[i][idx:], skip_special_tokens=True).strip("\n")
            for i, idx in enumerate(idxs)
        ]
        responses = [
            f"{think_resp}{cont}"
            for think_resp, cont in zip(thinking_contents, contents)
        ]

    return responses[0] if len(user_prompts) == 1 else responses


def load_models(embed_model_name: str, gen_model_name: str, device: str = None):
    # This will take some time to run for the first time if the model(s) don't exist locally.
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    dtype = torch.bfloat16 if device == "cuda" else torch.float16
    if device != "mps" or (device == "mps" and "mlx" not in gen_model_name.lower()):
        tok = AutoTokenizer.from_pretrained(gen_model_name, padding_side="left")
        # sometimes loading an AWQ model on my local machine fails for the first time
        try:
            gen = AutoModelForCausalLM.from_pretrained(
                gen_model_name, dtype=dtype, device_map=device
            ).eval()
        except ImportError:
            gen = AutoModelForCausalLM.from_pretrained(
                gen_model_name, dtype=dtype, device_map=device
            ).eval()
    else:
        from mlx_lm import load

        gen, tok = load(gen_model_name)

    embedder = SentenceTransformer(
        embed_model_name,
        device=device,
        model_kwargs={"dtype": dtype},
    )
    return embedder, tok, gen


def make_query_variants(
    tokenizer, model, query: str, prompt: str, n: int = 3, model_name="", **llm_kwargs
):
    # instructions = f"\n\n(Now give me at least {n} diverse variations of user query in the same language as the user provided query)"
    # query += instructions
    resp = generate_text(
        tokenizer, model, query.format(n=n), prompt, model_name=model_name, **llm_kwargs
    )
    clean_resp = re.sub(r"^\d+\.\s*", "", resp, flags=re.MULTILINE).split("\n")
    queries = [q.strip() for q in clean_resp if q.strip()]
    return [query.lower().strip()] + sorted(
        set(map(lambda x: str.lower(x).strip(), queries))
    )


def transform_query(
    tokenizer, model, query: str, rewrite_prompt: str, model_name="", **llm_kwargs
) -> dict:
    """split the query into things to search and actions to take"""
    resp = generate_text(
        tokenizer, model, query, rewrite_prompt, model_name=model_name, **llm_kwargs
    )
    try:
        resp = clean_rewrite_resp(resp)
    except:
        pass
    return resp


def aggregate_queries_and_tasks(
    tokenizer,
    model,
    orig_query,
    rewrite_prompt,
    variants_prompt,
    n_variations=3,
    gen_model_name="",
    **llm_kwargs,
):
    # make variations for the original query as is
    queries = make_query_variants(
        tokenizer,
        model,
        orig_query.strip(),
        variants_prompt,
        n_variations,
        gen_model_name,
        **llm_kwargs,
    )[: n_variations + 1]
    tr_q = transform_query(
        tokenizer, model, orig_query.strip(), rewrite_prompt, gen_model_name
    )

    # transformed query might have multiple things to search and tasks to perform depending on user query
    # recursively get variations for each of the search queries but keep the tasks as is.
    tasks = []
    if isinstance(tr_q, dict):
        search_results, tasks = tr_q.get("search", []), tr_q.get("tasks", [])
        for search_result in search_results:
            queries.extend(
                make_query_variants(
                    tokenizer,
                    model,
                    search_result,
                    variants_prompt,
                    n_variations,
                    gen_model_name,
                    **llm_kwargs,
                )
            )

    # keep the original user query as is (if in case LLM messes up the original query) and pick some after shuffling the rest
    q, qq = queries[0], queries[1:]
    shuffle(qq)
    queries = [q] + sorted(
        set(map(lambda x: str.lower(x).strip(string.punctuation), qq[:n_variations]))
    )
    tasks = sorted(set(map(lambda x: str.lower(x).strip(string.punctuation), tasks)))

    return queries, tasks


class HyDeRAGFusion:
    def __init__(
        self,
        embed_model: str,
        generator_llm_model: str,
        embed_batch_size: int = 8,
    ):
        self.embed_batch_size = embed_batch_size
        self.gen_model_name = generator_llm_model
        self.embedder, self.tok, self.gen = load_models(
            embed_model, generator_llm_model
        )
        self.text_splitter = MarkdownTextSplitter(chunk_overlap=450, chunk_size=3000)
        with open(PROMPTS_FILEPATH) as fl:
            self.prompts = yaml.safe_load(fl)

    def get_embeddings(self, texts, **kwargs):
        log.debug(f"batching size: {len(texts)} aka {self.embed_batch_size}")
        return self.embedder.encode(
            texts, batch_size=self.embed_batch_size, normalize_embeddings=True, **kwargs
        )

    @lru_cache(maxsize=2)
    def preprocess_pdfs(self, pdfs, **data_load_kwargs):
        log.debug(f"\n\n{'@@@@' * 20}\n\n preprocessing {pdfs=}")
        empty_cache()
        self.dataset = build_corpus(pdfs, self.text_splitter, **data_load_kwargs)
        empty_cache()
        self.dataset = self.dataset.map(
            lambda x: {
                "embeddings": self.get_embeddings(
                    x["chunk"], prompt_name="query", show_progress_bar=False
                )
            },
            batched=True,
            batch_size=self.embed_batch_size,
        )
        empty_cache()
        self.dataset.add_faiss_index(
            "embeddings", metric_type=faiss.METRIC_INNER_PRODUCT
        )

    def get_filtered_entries(self, idxs):
        # We need to drop the index before filtering/selecting the desired indices and re-add it later
        # Since it's FAISS and we index very little data, it's not noticeable
        self.dataset.drop_index("embeddings")
        entries = self.dataset.select(idxs)
        self.dataset.add_faiss_index(
            "embeddings", metric_type=faiss.METRIC_INNER_PRODUCT
        )
        return entries

    def retrieve(
        self, query, n_variants=3, top_k_per_variant=5, top_k_retrieve=3, **llm_kwargs
    ):
        queries, tasks = aggregate_queries_and_tasks(
            self.tok,
            self.gen,
            query.strip(),
            self.prompts["rewrite"],
            self.prompts["variants"],
            n_variants,
            self.gen_model_name,
            **llm_kwargs,
        )
        hyde_docs = generate_text(
            self.tok,
            self.gen,
            queries,
            self.prompts["hyde"],
            self.gen_model_name,
            **llm_kwargs,
        )
        chunks = []
        for hyde_doc in hyde_docs:
            chunks.extend(self.text_splitter.split_text(hyde_doc))
        q_emb = self.get_embeddings(chunks)
        matches = self.dataset.get_nearest_examples_batch(
            "embeddings", q_emb, top_k_per_variant
        )
        indices = [match["id"] for match in matches.total_examples]
        top_idxs = reciprocal_rank_fusion(indices, top_k_retrieve)
        return top_idxs, tasks

    def answer(self, query, idxs, tasks, max_ctx_chars=32768):
        total, text, prompt_length = 0, "", 10000
        sep = "\n\n-----\n\n"
        tasks = ", ".join(tasks) if tasks else ""
        log.debug("filtering entries")
        entries = self.get_filtered_entries(idxs)
        for chunk in entries["chunk"]:
            ctx = f"{sep}\n\n{chunk}"
            if total + len(ctx) + len(tasks) + len(sep) + prompt_length > max_ctx_chars:
                log.warning("context overflow")
                break

            text += ctx
            total = len(text)

        text += f"{sep}{tasks}"

        instruction = "go ahead and answer!"
        user_query = f"\nq: {query}\n\nctx:{text}" + f"\n\n{instruction}\n\n"
        resp = generate_text(
            self.tok,
            self.gen,
            user_query,
            self.prompts["final_answer"],
            self.gen_model_name,
        )

        sources = ""
        for idx, entry in enumerate(entries):
            source = f'<h2 style="color: cyan;">Source {idx + 1} :: {entry["file"]}:{entry["chunk_id"]}</h2>'
            sources += f"{sep}{source}\n\n{entry['chunk']}"

        return resp, sources.replace("```", "`")


def initial_setup(model_combo_key):
    models = MODEL_COMBOS[model_combo_key]
    hrf = HyDeRAGFusion(models["embed_model"], models["gen_model"])
    hrf._model_combo_key = model_combo_key
    return hrf


HRF = None


@click.command(context_settings=dict(show_default=True))
@click.option(
    "--pdfs",
    multiple=True,
    type=click.Path(exists=True),
    help="list of PDF filepaths to extract text from",
)
@click.option("--query", help="user query")
@click.option(
    "--model-combo-key",
    type=click.Choice(["linux", "HF-mid"]),
    default="linux",
    help="embedder and generator llm models combination to load (see config.py)",
)
@click.option(
    "--fast-extract/--no-fast-extract",
    default=False,
    help="Extract markdown text quickly (uses pymupdf if set, else docling if available)",
)
@click.option("--n-variants", default=3, help="no. of query variants")
@click.option(
    "--top-k-per-variant",
    default=5,
    help="top `k` hits per each query variant to consider for RRF",
)
@click.option(
    "--top-k-retrieve", default=3, help="top `k` chunks to retrieve after RRF"
)
@click.option("--temperature", default=0.7, help="LLM Model Temperature")
def ask(
    pdfs,
    query,
    model_combo_key,
    fast_extract,
    n_variants,
    top_k_per_variant,
    top_k_retrieve,
    temperature,
):
    pdfs = tuple(sorted(pdfs))
    log.debug(
        f"{pdfs=}, {query=}, {model_combo_key=}, {fast_extract=}, {n_variants=}, {top_k_per_variant=}, {top_k_retrieve=}, {temperature=}"
    )
    global HRF
    if HRF is None or HRF._model_combo_key != model_combo_key:
        if HRF is not None:
            log.debug("deleting HRF object")
            del HRF
        log.debug("emptying cache")
        empty_cache()
        log.debug(f"\n\n{'=:-:' * 20}\n\n initializing")
        start = time()
        HRF = initial_setup(model_combo_key)
        end = time()
        msg = f"init took {(end - start):.1f} seconds"
        log.debug(msg)

    start = time()
    if pdfs:
        HRF.preprocess_pdfs(pdfs, fast_extract=fast_extract)

    if query and query.strip():
        top_idxs, tasks = HRF.retrieve(
            query.strip(),
            int(n_variants),
            int(top_k_per_variant),
            int(top_k_retrieve),
            temperature=temperature,
        )

        log.debug("retrieving")
        resp, sources = HRF.answer(query, top_idxs, tasks)
        end = time()
        final_response = f"\nSearch took {(end - start):.1f} seconds\n\n{resp}{sources}"
        log.debug(final_response)
        return final_response

    return ""


if __name__ == "__main__":
    ask()
