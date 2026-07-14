# Information Retrieval

A collection of university coursework projects for an **Information Retrieval** course, implementing the core machinery of a search engine **from scratch** in Python — and then benchmarking it against modern machine-learning and transformer baselines.

Everything that a textbook IR system is made of is here: tokenization, positional inverted indexes, blocked sort-based indexing with γ-compressed postings, wildcard queries over a permuterm trie, spelling correction via edit distance, three classical ranking models (Vector Space, BM25/Probabilistic, Language Model) evaluated on the Cranfield collection, and finally text classification with Naive Bayes, Word2Vec, LSA, SVM and fine-tuned BERT.

> Almost none of the retrieval logic relies on a library. Linked-list postings, γ-codes, tries, edit-distance DP tables, tf-idf matrices and BM25 scores are all hand-rolled. `nltk` is used only for tokenization/stemming/stopwords, and `scikit-learn`/`gensim`/`transformers` only appear in the final ML comparison chapters.

---

## Table of Contents

- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Datasets you need to supply](#datasets-you-need-to-supply)
- [Project 1 — Boolean Search with a Positional Inverted Index](#project-1--boolean-search-with-a-positional-inverted-index-ir-booleansearchipynb)
- [Project 2 — BSBI Indexing with γ-Code Compression](#project-2--bsbi-indexing-with-γ-code-compression-bsbiipynb)
- [Project 3 — Wildcard Queries & Spelling Correction](#project-3--wildcard-queries--spelling-correction-wiledcard-spellcorrectionipynb)
- [Project 4 — Document Ranking & Evaluation on Cranfield](#project-4--document-ranking--evaluation-on-cranfield-doc-rankingipynb)
- [Project 5 — Word2Vec, LSA and Naive Bayes for Text Classification](#project-5--word2vec-lsa-and-naive-bayes-for-text-classification-word2vec-lsaipynb)
- [Project 6 — Persian Author Identification (Final Project)](#project-6--persian-author-identification-final-project-authorclassification)
- [Results at a glance](#results-at-a-glance)
- [Known issues, quirks and caveats](#known-issues-quirks-and-caveats)
- [Ideas for improvement](#ideas-for-improvement)
- [Authors](#authors)

---

## Repository structure

```
InformationRetrieval/
│
├── IR-BooleanSearch.ipynb            # Positional inverted index (linked lists) + Boolean/proximity queries
├── BSBI.ipynb                        # Blocked Sort-Based Indexing + gap encoding with Elias γ-codes
├── WiledCard-SpellCorrection.ipynb   # Permuterm trie → wildcard queries; edit distance → spell correction
├── doc-ranking.ipynb                 # VSM / BM25 / Language Model + 11-point interpolated precision (Cranfield)
├── Word2Vec-LSA.ipynb                # Naive Bayes vs. Word2Vec+SVM vs. LSA+SVM (binary sentiment corpus)
│
└── AuthorClassification/             # Final project: author identification in Persian literature
    ├── CODES/
    │   ├── preprocess.py             # hazm-based Persian preprocessing pipeline
    │   ├── NB.ipynb                  # Multinomial Naive Bayes on a trie-based class inverted index
    │   ├── SVM.ipynb                 # TF-IDF + linear SVM with 5-fold cross-validation
    │   ├── BERT.ipynb                # Fine-tuning BERT / XLM-RoBERTa / DistilBERT for 10-class classification
    │   └── CFV_Version.ipynb         # Same, with 5-fold cross-validation
    ├── DATASET/
    │   ├── Labels/y_data.csv         # 1,381 rows: Author, Author_ID (1–10)
    │   └── Stemmed/                  # X_data_stemmed.csv + train/test splits (stemmed Persian text)
    ├── Report/                       # Full project report (.docx and .pdf)
    └── AuthorClassification.pdf      # Original assignment brief
```

---

## Setup

The code is written for Python 3.9+ and runs in Jupyter. A minimal environment:

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate

pip install jupyter numpy pandas matplotlib seaborn nltk scikit-learn gensim pillow
pip install torch transformers                        # only for the BERT notebooks
pip install hazm                                      # only for Persian preprocessing
```

The NLTK notebooks expect a few corpora to be present. Run once:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

The transformer notebooks (`BERT.ipynb`, `CFV_Version.ipynb`) were written for **Google Colab** with a GPU — they read from `/content/` and fine-tuning on CPU is impractical.

---

## Datasets you need to supply

The first five notebooks read their data from the working directory, and that data is **not** committed to the repo. Place the following next to the notebook before running it:

| Notebook | Expects | What it is |
|---|---|---|
| `IR-BooleanSearch.ipynb` | `*.txt` files in the CWD | Any small plain-text document collection (the course used 15 documents) |
| `BSBI.ipynb` | `*.txt` files in the CWD, plus `test-people.jpg` | Same collection; the image is just a screenshot used for cross-checking results against Project 1 |
| `WiledCard-SpellCorrection.ipynb` | `*.txt` files in the CWD | Same collection |
| `doc-ranking.ipynb` | `cran.all.1400`, `cran.qry`, `cranqrel` | The **Cranfield** collection: 1,400 aerodynamics abstracts, 225+ queries, and human relevance judgements |
| `Word2Vec-LSA.ipynb` | `train_pos.csv`, `train_neg.csv`, `test_pos.csv`, `test_neg.csv` | A balanced binary-sentiment review corpus (25k train / 25k test — an IMDB-style split); the review text must be the second column |
| `AuthorClassification/CODES/preprocess.py` | `data2.csv` | Raw scraped corpus with `Text`, `Author`, `Author_ID` columns (the already-preprocessed output *is* committed under `DATASET/`) |

Document IDs in `doc-ranking.ipynb` are 1-based to match `cranqrel`.

---

## Project 1 — Boolean Search with a Positional Inverted Index (`IR-BooleanSearch.ipynb`)

A complete Boolean retrieval engine built on a hand-written linked list.

**Index structure.** The dictionary maps every term to a `LinkedList` of `Node`s. Each node stores one `doc_id` **and** the list of token positions at which the term occurs in that document — i.e. a *positional* index, which is what makes proximity search possible.

```
"people" → [doc 0 | pos: 12, 87, 240] → [doc 1 | pos: 5] → [doc 8 | pos: 33, 91] → None
```

Because documents are processed in increasing ID order, every postings list comes out **sorted by doc_id for free**, which is exactly the invariant the merge algorithms below rely on.

**Preprocessing** (`tokenize`): regex tokenization with `\d+,\d+|\w+` (so `1,500` survives as a number, with the comma later stripped), lowercasing, removal of non-alphanumeric tokens, and English stop-word removal.

**Query language.**

| Query form | Handler | Algorithm |
|---|---|---|
| `word` | `handling_term` | Walk the postings list |
| `NOT word` | `handling_NOT` | Complement of the postings list against all doc IDs |
| `w1 AND w2` | `handling_AND` | Classic two-pointer **intersection** — advance the smaller doc_id |
| `w1 OR w2` | `handling_OR` | Two-pointer **union**, then drain the remaining list |
| `w1 NEAR/k w2` | `handling_proximity` | Intersect the doc lists, then check position lists for a pair within distance *k* |

The merges run in **O(len(p₁) + len(p₂))** rather than the naive O(n·m), which is the whole point of keeping postings sorted. Proximity checking (`distance_checking`) exploits sorted position lists with an early `break` once the second position overshoots the first.

**Run it:** execute all cells, then type a query such as `information AND retrieval`, `NOT people`, or `data NEAR/5 mining` at the input prompt.

---

## Project 2 — BSBI Indexing with γ-Code Compression (`BSBI.ipynb`)

This notebook rebuilds the index the way a real system would when the collection doesn't fit in memory, and then squeezes the postings down to a bit string.

**Blocked Sort-Based Indexing.** Documents are processed in **blocks of 3**. An inverted index is built per block, and the blocks are then merged pairwise into a single dictionary (`merge_inverted_index`) — the in-memory analogue of BSBI's external merge.

**Gap encoding + Elias γ-codes.** Instead of storing doc IDs, each postings list stores the *gaps* between consecutive IDs, and each gap is encoded as an Elias γ-code, concatenated into one long string of bits:

- `to_code(n)` — writes `n` as γ-code: `(len(binary)-1)` ones, a `0` separator, then the binary of `n` **without its leading 1** (the offset).
- `to_id(s)` — decodes an entire γ-code string and returns the **last** document ID it represents (needed to compute the next gap).
- `gamma_code_connect(s, doc_id)` — appends a new doc ID by decoding the current tail, computing the gap, and encoding it.

**Worked example** from the notebook — the stem `peopl` is stored as `0011010011000100`:

| Code | Decodes to | Running doc ID |
|---|---|---|
| `0` | 1 | **1** |
| `0` | 1 | **2** |
| `11010` | 6 | **8** |
| `0` | 1 | **9** |
| `11000` | 4 | **13** |
| `100` | 2 | **15** |

→ postings `{1, 2, 8, 9, 13, 15}` (1-based). The notebook cross-checks this against Project 1's uncompressed output for the same term.

**Note:** unlike Project 1, this pipeline applies the **Porter stemmer**, so terms are stems (`people` → `peopl`).

---

## Project 3 — Wildcard Queries & Spelling Correction (`WiledCard-SpellCorrection.ipynb`)

Tolerant retrieval: the query no longer has to be spelled correctly, or even completely.

### Permuterm index on a trie

Every term is suffixed with `$` and **all its rotations** are inserted into a trie:

```
"hello" → "hello$" → hello$, ello$h, llo$he, lo$hel, o$hell
```

Each terminal `TrieNode` keeps a `doc` dictionary — `{doc_id: [positions]}` — so the trie doubles as a full positional index, replacing the linked lists of Project 1.

**Wildcard resolution** (`clean_word` → `wildcard` → `make_original`):

1. Append `$`, then **rotate the query until the `*` sits at the far right**, turning any wildcard into a prefix search. `he*o` → `o$he*`, so the prefix to look up is `o$he`.
2. With **two** stars, the rotation direction is chosen so that the *longest* usable prefix is obtained, and the substring between the stars is kept as an extra filter.
3. Descend the trie to the prefix node, then `Trie_traverse` recursively collects every completion beneath it.
4. Completions are filtered (for the two-star case) and rotated back to their original form with `make_original`.

Supports `*ello`, `hell*`, `he*lo`, and `h*ll*` styles.

### Spelling correction

`edit_distance` is a textbook **Levenshtein DP** (insert / delete / substitute, unit cost, full `(m+1)×(n+1)` table). `spell_correction` scores the query word against every unique term in the collection, keeps candidates with distance < 5, and returns **all terms tied at the minimum distance**.

### Putting it together

`information_retreival` first *normalizes* each query term — expanding wildcards or generating spelling candidates — and then runs the Boolean/proximity machinery (`AND`, `OR`, `NOT`, `NEAR/k`) over the **Cartesian product** of the candidate terms, printing results for each combination. Given a plain multi-word query with no operators, it instead behaves as a pure query-correction service and prints every corrected phrasing.

---

## Project 4 — Document Ranking & Evaluation on Cranfield (`doc-ranking.ipynb`)

Three ranked-retrieval models, implemented from their formulas, evaluated head-to-head on the classic **Cranfield** test collection.

The parser splits `cran.all.1400` on the `.I` markers and pulls out the `.T` (title) and `.W` (abstract) fields with regex; `cranqrel` is parsed into `{query_id: {doc_id: relevance}}`.

### 1. Vector Space Model
A dense `n_terms × n_documents` **tf-idf matrix**, with

$$\text{tf-idf}_{t,d} = \text{tf}_{t,d}\cdot\log_2\!\left(\frac{N}{\text{df}_t}\right)$$

and each document column L2-normalized. Queries are mapped into the same space and scored by **cosine similarity** (`cosine_score`).

### 2. Probabilistic Model (BM25 / RSV)
Full BM25 retrieval status value, with tf smoothed by +0.05:

$$RSV_d = \sum_{t\in q}\log\frac{N}{\text{df}_t}\cdot\frac{(k_1+1)\,\text{tf}_{td}}{k_1\big((1-b)+b\frac{L_d}{L_{ave}}\big)+\text{tf}_{td}}\cdot\frac{(k_3+1)\,\text{tf}_{tq}}{k_3+\text{tf}_{tq}}$$

Run with **k₁ = 1.2, k₃ = 2, b = 0.75**. The query-saturation term is switched off (`k₃ = 0`) for queries shorter than 20 tokens, since it only matters for long queries.

### 3. Language Model (query likelihood, Jelinek–Mercer smoothing)

$$\log P(q|d) \propto \sum_{t\in q}\log\Big(\lambda\frac{\text{tf}_{t,d}}{L_d} + (1-\lambda)\frac{\text{cf}_t}{T}\Big)$$

Run with **λ = 0.9** (10% of the mass from the collection model). Scores are accumulated in log-space to avoid underflow.

### Evaluation
`elev_point_interpolated_ave_per` walks the ranked list, recomputing recall/precision at each rank and recording precision at the first point where recall reaches each of 0.0, 0.1, …, 1.0 — the **11-point interpolated precision** curve. `sketch_plot` overlays all three models on one precision–recall graph, and the notebook plots individual queries (1, 2, 23, 73, 157) plus an average.

**Finding:** on this collection the **Probabilistic (BM25) model wins**, with the Language Model second and the Vector Space Model last on most queries — the ordering is inverted only occasionally (e.g. query 73, where the LM comes out ahead).

---

## Project 5 — Word2Vec, LSA and Naive Bayes for Text Classification (`Word2Vec-LSA.ipynb`)

A binary-sentiment classification bake-off on a 25k/25k positive–negative review corpus, comparing a from-scratch generative model against two embedding pipelines.

| Approach | How | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
| **Multinomial Naive Bayes** (hand-written) | Laplace-smoothed $P(t|c)=\frac{T_{t,c}+1}{\sum_{t'} T_{t',c}+1}$, log-summed over the document | **0.815** | — | — | — |
| **Word2Vec + SVM** | `gensim` Word2Vec (`vector_size=200`, `window=20`, `min_count=1`) trained on the training split; each document = **mean of its word vectors**; RBF `SVC` on top | **0.837** | 0.839 | 0.833 | 0.836 |
| **LSA + SVM** | `TfidfVectorizer` → `TruncatedSVD(n_components=200)` → `SVC` | **0.866** | 0.871 | 0.860 | 0.865 |

**Takeaway:** averaging Word2Vec vectors is a lossy way to represent a whole document — the order and salience of words are thrown away — and it barely beats a bag-of-words Naive Bayes. Truncated SVD over tf-idf (i.e. **Latent Semantic Analysis**) keeps the discriminative term weights while denoising them into 200 latent topics, and comes out on top.

Shared preprocessing (stopword removal, Porter stemming, alphanumeric filtering) mirrors the earlier notebooks.

---

## Project 6 — Persian Author Identification (Final Project) (`AuthorClassification/`)

> **Task:** given a ~500-word passage of **Persian** prose, identify which of 10 authors wrote it.

### The dataset

Hand-curated (deliberately **not** scraped) from Persian translations of mystery/supernatural fiction. **1,381 documents, 10 classes:**

| Author ID | Author | Docs |
|---:|---|---:|
| 1 | Artemis Fowl | 159 |
| 2 | Anthony Horowitz | 109 |
| 3 | Brandon Malle | 257 |
| 4 | John Flanagan | 129 |
| 5 | D.G. McHale | 173 |
| 6 | Rick Riordan | 228 |
| 7 | D.B. Reynolds | 76 |
| 8 | J.K. Rowling | 92 |
| 9 | R.L. Stine | 56 |
| 10 | C.S. Lewis | 102 |

Stratified 80/20 split (`random_state=42`) → **1,104 train / 277 test**.

### Preprocessing (`preprocess.py`)

Built on **`hazm`**, the Persian counterpart of NLTK:

1. Strip special characters (`re.sub(r'[^\w\s]', '', text)`).
2. Strip Latin characters and digits — the text should be purely Persian.
3. Remove Persian stop words (`hazm.stopwords_list()`).
4. **Stem** every token (`hazm.Stemmer`).
5. Lemmatization was implemented but **deliberately dropped** — `hazm`'s lemmatizer degraded quality, so the committed corpus is stemmed only.

The output lives in `DATASET/Stemmed/X_data_stemmed.csv`, aligned row-for-row with `DATASET/Labels/y_data.csv`.

### Models

**`NB.ipynb` — Naive Bayes over a trie-backed class index.**
One `Trie` per class, storing term frequencies (and posting sets) per author. Prediction scores every class with the Laplace-smoothed multinomial NB rule

$$\hat{c} = \arg\max_c \sum_{t\in d}\log_2\frac{\text{tf}_{t,c}+1}{T_c + |V|}$$

and picks the argmax. Trains in minutes, no embeddings, no GPU.

**`SVM.ipynb` — TF-IDF + linear SVM.**
`TfidfVectorizer(max_features=100)` → `SVC(kernel='linear', C=1e7)`, wrapped in `StratifiedKFold(5)` with accuracy / macro-F1 / macro-precision / macro-recall reported per fold.

**`BERT.ipynb` / `CFV_Version.ipynb` — transformer fine-tuning.**
`BertForSequenceClassification` with `num_labels=10`, `max_length=128`, `AdamW(lr=1e-5)`, batch size 32, 3–6 epochs. Several backbones were tried; `CFV_Version.ipynb` repeats the winner under 5-fold cross-validation.

### Results (from the project report)

| Model | Accuracy | Precision | Recall | F1 | Fine-tuning time |
|---|---:|---:|---:|---:|---|
| DistilBERT (multilingual) | 33% | — | — | — | ~20 min (3 epochs) |
| DistilRoBERTa | 18% | — | — | — | ~15 min (3 epochs) |
| BERT Base Multilingual | 50% | — | — | — | ~1 hr (3 epochs) |
| XLM-RoBERTa base | ~19% | — | — | — | (reproduced in `BERT.ipynb`) |
| **BERT Base Persian** (`HooshvareLab/bert-fa-base-uncased`), 3 epochs | 65% | — | — | — | ~1 hr |
| **BERT Base Persian**, 6 epochs | **88%** | 89% | 88% | 87% | ~2 hr |
| **BERT Base Persian**, 5-fold CV | **93%** | 92% | 90% | 88% | ~9–10 hr |
| TF-IDF + linear SVM | 87% | 87% | 87% | 87% | ~15 min |
| **Naive Bayes** (trie index) | **99%** | 99% | 99% | 98% | ~7–8 min |

**The headline result is that a hand-written Naive Bayes beats a fine-tuned Persian BERT** — 99% vs. 93%, at roughly 1/80th of the compute. The report explicitly re-audited the data for leakage (overlapping splits, author names leaking into the text) and found none. The plausible explanation is that authorship in this corpus is signalled almost entirely by **lexical fingerprints** — recurring character names, translator vocabulary, function-word habits — which a bag-of-words model captures perfectly, while BERT is additionally handicapped by a 128-token truncation that discards ~75% of each 500-word document.

The report (`Report/Report_IR_Final_Project.pdf`) also discusses the effect of learning rate, of removing stop words (which can strip exactly the stylistic cues authorship attribution depends on), and of document length.

---

## Results at a glance

| Project | Task | Best result |
|---|---|---|
| Boolean Search | Exact Boolean + proximity retrieval | Positional index with O(n+m) merges |
| BSBI | Index compression | Gap + Elias γ encoding, verified against the uncompressed index |
| Wildcard / Spell | Tolerant retrieval | Permuterm trie + Levenshtein candidates |
| Doc Ranking | Ranked retrieval (Cranfield) | **BM25 > Language Model > Vector Space** |
| Word2Vec / LSA | Binary sentiment classification | **LSA + SVM: 86.6%** (vs. W2V+SVM 83.7%, NB 81.5%) |
| Author Classification | 10-class Persian authorship | **Naive Bayes: 99%** (vs. BERT-fa 93%, SVM 87%) |

---

## Known issues, quirks and caveats

Worth knowing before you run or reuse this code:

- **`preprocess.py` does not run as committed** — `def split_data(self)` is missing its colon, and the file ends mid-function. It's best read as documentation of the pipeline that produced `DATASET/Stemmed/`, not as an executable script.
- **`doc-ranking.ipynb`, average-precision cell:** the loop iterates `query_id` over `range(1, 226)` but passes the literal `157` to `query_IR`, so the "average" curve is really query 157 evaluated 225 times. Passing `query_id` fixes it — but note that `elev_point_interpolated_ave_per` will `KeyError` on queries absent from `cranqrel`, so guard with `if query_id in que_doc_rel`.
- **Hardcoded ID offsets:** the ranking functions patch document indices around two empty Cranfield documents with `if index >= 471: index += 1` (and again at 995). This is brittle — it's compensating for documents dropped at parse time rather than preserving the original IDs.
- **Double length normalization** in `cosine_score`: the tf-idf columns are already L2-normalized, and the score is then divided by document length again. This is deliberate in the notebook's narrative but it is *not* standard cosine similarity, and it biases the VSM against long documents. It likely contributes to VSM finishing last.
- **`SVM.ipynb` evaluates with `cross_val_predict` on the test set** (`cross_val_predict(clf, X_test_tfidf, y_test, cv=5)`), which cross-validates *within* the held-out data rather than predicting from the model trained on the training set. The reported 87% should be read with that in mind. `max_features=100` is also a very aggressive vocabulary cap.
- **`Word2Vec-LSA.ipynb`:** a markdown cell says `vector_size=500`; the code uses `200`. The code is what ran.
- **`BERT.ipynb` as committed runs XLM-RoBERTa**, not the Persian BERT — the `HooshvareLab` lines are commented out. Its saved output (≈19% accuracy, everything predicted as class 2) is the *failed* XLM-R run, not the 88% headline. Swap the commented lines back in to reproduce the main model.
- **`CFV_Version.ipynb` reuses one `model` object across all 5 folds** without re-instantiating it, so folds 2–5 continue training an already-fitted model. The 93% figure should be treated as optimistic.
- **Filename typo:** `WiledCard-SpellCorrection.ipynb` (should be *Wildcard*).
- The first three notebooks call `input()` and print results, so they're interactive by design — there is no programmatic API.

---

## Ideas for improvement

- Replace `input()` prompts with functions returning result lists, and add a thin CLI (`python search.py "info AND retrieval"`).
- Support arbitrary Boolean expressions (parentheses, `n`-ary AND/OR) via a proper query parser instead of the current 1/2/3-token pattern match.
- Store postings as a **skip list** to speed up intersection, and persist the γ-encoded index to disk so BSBI actually spills to external storage.
- Vectorize `cosine_score` and `RSV` with NumPy/SciPy sparse matrices; the current triple loop is O(terms × documents) per query.
- Add MAP, nDCG and P@k to the Cranfield evaluation alongside the 11-point curve.
- For authorship: increase BERT's `max_length` (or use a long-context/hierarchical model) so the full 500-word document is seen, and add a character-n-gram baseline — traditionally the strongest classical feature set for authorship attribution.
- Pin dependencies in a `requirements.txt`.

---

## Authors

Coursework by **Arad Vazir Panah**, **Aaron Bateni**, **Nima Niroomand**, and **Yazdan Zandiye Vakili**.
Final project supervised by **Dr. BabaAli**.

The full write-up for the author identification project — dataset construction, model selection, confusion matrices, and the ablation discussion — is in [`AuthorClassification/Report/Report_IR_Final_Project.pdf`](AuthorClassification/Report/Report_IR_Final_Project.pdf).

---

## License

No license file is currently included. Without one, the code is under exclusive copyright by default; if you'd like others to reuse it, consider adding a `LICENSE` (MIT is a common choice for coursework).
