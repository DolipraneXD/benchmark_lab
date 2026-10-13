# Own retrieval data

Put your evaluation set here (both paths are git-ignored):

```
retrieval/queries.jsonl   one JSON object per line: {"id": "q1", "query": "...", "relevant_doc_ids": ["doc-a", "doc-b"]}
retrieval/docs/           one file per document (id = file name without extension),
                          or a single docs.jsonl with {"id": "...", "text": "...", "title": "..."} per line
```

`benchlab run --model qemb --bench own` embeds every document and query with the
embedding model, retrieves the top-1000 by cosine similarity, scores recall@10 /
nDCG@10 with mteb's evaluator, then re-ranks the top-100 with
`settings.reranker_model` and scores again (`own_rec`, `own_ndcg`, `own_rec_rr`,
`own_ndcg_rr`). Every `relevant_doc_ids` entry must exist in `docs/`; the run
fails early otherwise.
