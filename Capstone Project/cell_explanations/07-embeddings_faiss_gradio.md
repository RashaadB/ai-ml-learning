# 07 Embeddings, FAISS, and Gradio app

Overview

This cell builds semantic embeddings for product titles, indexes them in FAISS for fast similarity search, implements a fallback fuzzy matcher, constructs prompts for a language model, and builds a small Gradio web interface to generate product descriptions.

Purpose

- Demonstrate a retrieval augmented generation prototype that uses embeddings for context.
- Offer a small interactive demo via Gradio to generate marketing copy for product titles.
- Save a toy fine tuning dataset in JSON Lines format for future use.

Line by line explanation

1. `#section 4` and imports
   - Import pandas and numpy for tabular and numeric handling.
   - Import `SentenceTransformer` for sentence level embeddings.
   - Import `faiss` for fast nearest neighbor search on numeric vectors.
   - Import `rapidfuzz` as a fallback fuzzy string matcher.
   - Import `openai` for the external LLM call and `gradio` for the demo web interface.
   - Import `Dataset` from Hugging Face `datasets` to write a small JSONL dataset.

2. Read product title data
   - `df = pd.read_csv(..., usecols=[...])` reads only the product id, title, and category to limit memory use.
   - `df.dropna(...)` removes incomplete rows.
   - `df = df.head(1000)` keeps the example compact for demos.

3. Clean titles
   - Lowercase and remove punctuation from the product title to produce `clean_title` which improves embedding stability and fuzzy matching.

4. Embedding and FAISS setup
   - Load a compact model `all-MiniLM-L6-v2` which provides good semantic embeddings at low compute cost.
   - Compute embeddings for each cleaned title. The result is a numeric vector per title.
   - Initialize a FAISS index using L2 distance with the vector dimension set from the embeddings and add all vectors to the index.
   - Save `product_metadata` as a DataFrame to lookup titles and categories for returned indices.

5. Fuzzy matching fallback
   - Define `fuzzy_match_product` that uses `rapidfuzz.process.extract` to find close string matches when a semantic search is insufficient or when you want a quick string match.

6. Prompt construction for LLM
   - `create_prompt` returns a formatted prompt instructing the language model to write a persuasive product description using the provided product title and category.

7. LLM completion wrapper
   - `generate_description` calls `openai.ChatCompletion.create` with `gpt-4` and returns the text of the response. Parameters `temperature` and `top_p` control randomness and nucleus sampling.
   - This call requires an environment variable or config with a valid OpenAI API key to succeed.

8. RAG retrieval function
   - `search_faiss` encodes a query, performs a FAISS search to retrieve the top k nearest titles, and returns corresponding rows from `product_metadata`.

9. Build RAG prompt
   - `build_rag_prompt` composes a prompt that includes the query and a bullet list of relevant product titles and categories returned by FAISS, so the LLM has context to ground its output.

10. `rag_generate` combines retrieval and generation
   - It first retrieves similar products and then calls the LLM with the built prompt to produce a description. This creates a retrieval augmented generation flow.

11. Fine tuning dataset preparation
   - Create a small DataFrame `df_train` with columns `instruction`, `input`, and `output` suitable for supervised fine tuning format. Save it to `product_data.jsonl` so it can be reused later.

12. Gradio interface
   - `run_app` is a wrapper that creates a prompt from user supplied title and category and calls `generate_description` to get the model output.
   - `gr.Interface` builds a simple web UI with three input text boxes and a single text output. `interface.launch()` starts the local web server for the demo.

Inputs and outputs

- Inputs: product title and category entered by the user, plus an OpenAI API key for generation.
- Outputs: generated product description text returned by the LLM and displayed in the Gradio UI.

Security and cost notes

- The OpenAI API call sends prompts to a remote service and may incur cost. Make sure you have an API key set in your environment before running this cell.
- Do not commit API keys to version control. Use environment variables or secure secret management.

Notes and tips

- The FAISS index is in memory. For larger datasets persist indexes to disk using a FAISS save call and reload them for later use.
- The sentence transformer model chosen balances speed and quality. For higher quality at more compute cost, choose a larger model.
- When demoing, set `df = df.head(100)` to keep encoding and search fast for presentations.

---

End of 07 Embeddings, FAISS, and Gradio app
