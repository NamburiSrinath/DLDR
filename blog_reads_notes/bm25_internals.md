<b> Link: </b> https://medium.com/@rutikaambadkar1/okapi-bm25-the-probabilistic-ranking-function-that-powers-search-engines-4ce2e43798ed

---
- TF value is specific to a single document `d` while IDF depends on entire corpus
- BM25: Higher value of k -> increases the term frequency's contribution i.e multiple mentions matter more!!
- Higher value of b -> longer documents are penalized more
- Offline (index construction phase): Convert document (a document can be chunk, paragraph etc;) to "sparse" vector (feature vector) which can be later used for similarity search 
- During inference, we compute the score (TF-IDF or BM25) of query `Q` for each document `d` (corpus has `D` documents) and the top k documents will be retrieved! So, it's basically Score(`Q`, `d`, `D`) i.e for every token `q` in `Q`, compute \sum TF(`q`, `d`) (that'll be TF score for a particular document) and IDF(`q`) which will be be IDF score for that term (across all documents). 
- Postgres has `to_tsvector` to convert a document to tokens (https://www.postgresql.org/docs/current/textsearch-controls.html). This basically stems, lemmatizes and gets the count of each lexeme! and returns a `tsvector` type
- It also has `to_tsquery` type which converts a given query to `tsquery` type
```
SELECT to_tsquery('english', 'fit <-> fitterz') && to_tsvector('english', 'fit fat fitterz)
``` 
will return false as <-> means that fit should be followed by fitterz but the vector doesn't have that exact follow-up
- But these booleans are not always useful for agentic chatbots as users won't usually type boolean logic (unlike SEO search engines) and `websearch_to_tsquery` has some support (default is A AND B AND C etc; ie AND is applied to text by default, so make sure to update to OR to be more expressive).
- [`ts_rank_cd` gives the **cover density** score between a query (of `tsquery` type) and document (of `tsvector` type). 
- Lack of global information: But these ranking functions don't use any global information (so can be viewed like sort-of like precursor to TF-IDF, BM25) and this can be a limiting factor. 
- Computational expensiveness: Also, for very long queries, simple 'OR'-ing makes the search computationally expensive when looking at millions of documents. 
- Efficient BM25 implementation: BM25 score can be viewed as similarity score between 2 sparse vectors i.e representing query and document as sparse vectors and the efficiency can come from HNSW lookups, thus only computing the query embeddings during inference.
- **Read:** HNSW, Iterative scan, relaxed and strict order!!
- **Read** RRF (Reciprocal rank fusion) vs interleave
