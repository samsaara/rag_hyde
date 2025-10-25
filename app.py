import sys
import string
import gradio as gr
from src.main import ask
from src.utils import empty_cache
from src.config import log


def ask_wrapper(
    pdfs,
    query,
    model_combo_key,
    fast_extract,
    n_variants,
    top_k_per_variant,
    top_k_retrieve,
    temperature,
):
    resp = ask.callback(
        pdfs,
        query,
        model_combo_key,
        fast_extract,
        n_variants,
        top_k_per_variant,
        top_k_retrieve,
        temperature,
    )

    return f"## Final Answer:\n\n{resp}\n"


def reset(pdfs):
    """
    Reset text input and empty cache
    """
    log.warning("emptying cache")
    empty_cache()
    return ""


# Enable the button only when both fields are nonempty
def _enable_submit_if_filled(pdfs, query):
    status = bool(pdfs) and bool(len(query.strip(string.punctuation + " ")) > 10)
    return gr.update(interactive=status)


def disable_button():
    return gr.update(interactive=False, value="Processing...")


def enable_button():
    return gr.update(interactive=True, value="Submit")


with gr.Blocks(title="RAG with HYDE") as demo:
    gr.Markdown("# RAG with HYDE")
    with gr.Row():
        pdf_input = gr.File(
            label="upload PDF(s)",
            file_types=[".pdf"],
            file_count="multiple",
        )
        query = gr.Textbox(label="Question (Enter at least 10 valid characters)")

    with gr.Accordion("Advanced Settings", open=False):
        gr.Markdown(
            "*These parameters have sensible defaults but can be customized if needed*"
        )
        with gr.Row():
            _default_combo = "linux" if sys.platform == "linux" else "mac"
            model_combo_key = gr.Dropdown(
                label="Model Combo Key",
                choices=[_default_combo, "HF-mid"],
                value=_default_combo,
            )
            fast_extract = gr.Checkbox(
                value=False, label="Use PyMuPDF to extract content in markdown"
            )
            n_variants = gr.Number(
                value=3, minimum=1, maximum=5, label="no. of query variants"
            )
        with gr.Row():
            top_k_per_variant = gr.Number(
                value=5,
                minimum=2,
                maximum=10,
                label="top `k` hits per query variant for RRF",
            )
            top_k_retrieve = gr.Number(
                value=3,
                minimum=1,
                maximum=5,
                label="top `k` chunks to retrieve after RRF",
            )
            temperature = gr.Slider(
                value=0.7, minimum=0.1, maximum=1.0, step=0.1, label="temperature"
            )

    gr.Markdown(
        "### *Please be patient after hitting the submit button* esp. for the first question after uploading new document(s)"
    )
    submit_btn = gr.Button("Submit", variant="primary", interactive=False)
    answer = gr.Markdown(label="## Answer")

    pdf_input.change(
        _enable_submit_if_filled, [pdf_input, query], submit_btn, queue=False
    )
    query.change(_enable_submit_if_filled, [pdf_input, query], submit_btn, queue=False)

    submit_btn.click(fn=disable_button, outputs=submit_btn).then(
        fn=ask_wrapper,
        inputs=[
            pdf_input,
            query,
            model_combo_key,
            fast_extract,
            n_variants,
            top_k_per_variant,
            top_k_retrieve,
            temperature,
        ],
        outputs=answer,
    ).then(fn=enable_button, outputs=submit_btn)

    query.submit(fn=disable_button, outputs=submit_btn).then(
        fn=ask_wrapper,
        inputs=[
            pdf_input,
            query,
            model_combo_key,
            fast_extract,
            n_variants,
            top_k_per_variant,
            top_k_retrieve,
            temperature,
        ],
        outputs=answer,
    ).then(fn=enable_button, outputs=submit_btn)

    pdf_input.change(reset, pdf_input, query)
    demo.load(reset, pdf_input, query)

if __name__ == "__main__":
    demo.launch()
