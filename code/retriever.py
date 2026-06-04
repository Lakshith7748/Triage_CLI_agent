"""BM25 retrieval over data/corpus.json.

CLI: python retriever.py "<query>" [company]
"""

import os
import re
import json

from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CORPUS_PATH = os.path.join(BASE_DIR, '..', 'data', 'corpus.json')

MIN_TOKEN_LENGTH = 2

STOPWORDS = frozenset("""
a an the and or but if then else of to in on at by for with from as is are
was were be been being have has had do does did i you he she it we they
this that these those my your his her our their me him us them so not no
yes can will would could should may might must shall how what why where
when which who whom whose about into out up down over under again further
""".split())

# Tuned by manual inspection: real matches score 5–28, off-topic queries 0–1.
CONFIDENCE_THRESHOLD = 3.0

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    if not text:
        return []
    tokens = _TOKEN_RE.findall(text.lower())
    return [t for t in tokens if len(t) >= MIN_TOKEN_LENGTH and t not in STOPWORDS]


class Retriever:
    def __init__(self, corpus_path=CORPUS_PATH):
        self.corpus_path = corpus_path
        self.docs = self._load_corpus()
        self._tokenized = [tokenize(d['searchable_text']) for d in self.docs]
        self._bm25 = BM25Okapi(self._tokenized)

    def _load_corpus(self):
        if not os.path.exists(self.corpus_path):
            raise FileNotFoundError(
                f"Corpus not found at {self.corpus_path}. Run build_corpus.py first."
            )
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def search(self, query, top_k=3, company=None):
        """Return top-k hits as dicts with keys: doc, score, confident, product_area.

        ``company`` is a soft 1.5x score boost, not a hard filter — cross-domain
        tickets ("paid for Claude with my Visa") still need to reach the right corpus.
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        if company:
            company_lc = company.lower()
            scores = [
                s * 1.5 if self.docs[i]['company'].lower() == company_lc else s
                for i, s in enumerate(scores)
            ]

        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        hits = []
        for i in ranked:
            score = float(scores[i])
            doc = self.docs[i]
            hits.append({
                'doc': doc,
                'score': score,
                'confident': score >= CONFIDENCE_THRESHOLD,
                'product_area': self._product_area_for(doc),
            })
        return hits

    @staticmethod
    def _product_area_for(doc):
        sub = doc.get('subcategory')
        if sub:
            return sub.split('/')[0]
        return doc['company'].lower()


def _cli():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python retriever.py <query> [company]")
        sys.exit(1)
    query = sys.argv[1]
    company = sys.argv[2] if len(sys.argv) > 2 else None

    r = Retriever()
    hits = r.search(query, top_k=3, company=company)
    if not hits:
        print("(no results)")
        return
    for rank, hit in enumerate(hits, 1):
        confident = "OK " if hit['confident'] else "LOW"
        print(
            f"#{rank} [{confident} score={hit['score']:5.2f}] "
            f"{hit['doc']['company']:11} | "
            f"area={hit['product_area']:30} | "
            f"{hit['doc']['title']}"
        )


if __name__ == '__main__':
    _cli()