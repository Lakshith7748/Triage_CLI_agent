"""Build a single corpus.json from markdown files under data/.

Each markdown file becomes one document with stable id, company,
subcategory, title, body, and a markdown-stripped searchable_text
field used by the BM25 retriever.
"""

import os
import re
import json
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

COMPANY_ALIASES = {
    'hackerrank': 'HackerRank',
    'claude': 'Claude',
    'anthropic': 'Claude',
    'visa': 'Visa',
}

URL_KEYS = ('source_url', 'url', 'canonical_url', 'link')
PASSTHROUGH_KEYS = ('title_slug', 'article_id', 'last_updated_iso', 'breadcrumbs')
MIN_BODY_LENGTH = 40

FRONT_MATTER_RE = re.compile(r'^---\s*\n(.*?)\n---\s*\n?(.*)$', re.DOTALL)
LAST_UPDATED_LINE_RE = re.compile(r'^\s*_Last updated:.*?_\s*$', re.MULTILINE)


# --- Front-matter --------------------------------------------------------

def _strip_quotes(value):
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
        return value[1:-1]
    return value


def parse_front_matter(raw_text):
    """Parse a small subset of YAML: scalars and string lists. Returns (dict, body)."""
    match = FRONT_MATTER_RE.match(raw_text)
    if not match:
        return {}, raw_text

    fm_block, body = match.group(1), match.group(2)
    metadata = {}
    current_list_key = None

    for line in fm_block.splitlines():
        if not line.strip():
            current_list_key = None
            continue

        stripped = line.lstrip()
        if current_list_key and line[0] in (' ', '\t') and stripped.startswith('-'):
            metadata[current_list_key].append(_strip_quotes(stripped[1:].strip()))
            continue

        current_list_key = None
        if stripped.startswith('#') or ':' not in stripped:
            continue

        key, _, value = stripped.partition(':')
        key, value = key.strip().lower(), value.strip()

        if value == '':
            metadata[key] = []
            current_list_key = key
        else:
            metadata[key] = _strip_quotes(value)

    return metadata, body.lstrip('\n')


# --- Body cleanup --------------------------------------------------------

def clean_body(body, title):
    """Strip duplicated H1 and Help-Center 'Last updated' line."""
    lines = body.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    if lines and lines[0].strip().startswith('# '):
        first_heading = lines[0].strip()[2:].strip()
        if title and first_heading.lower() == title.strip().lower():
            lines = lines[1:]

    text = LAST_UPDATED_LINE_RE.sub('', '\n'.join(lines))
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_first_heading(text):
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('# '):
            return stripped[2:].strip()
    return None


# --- Markdown stripping for BM25 ----------------------------------------

_MD_CODE_FENCE_RE = re.compile(r'```.*?```', re.DOTALL)
_MD_INLINE_CODE_RE = re.compile(r'`([^`]+)`')
_MD_IMAGE_RE = re.compile(r'!\[([^\]]*)\]\([^)]+\)')
_MD_LINK_RE = re.compile(r'\[([^\]]+)\]\([^)]+\)')
_MD_HEADING_RE = re.compile(r'^\s{0,3}#{1,6}\s+', re.MULTILINE)
_MD_BOLD_RE = re.compile(r'\*\*([^*]+)\*\*|__([^_]+)__')
_MD_ITALIC_RE = re.compile(r'(?<!\*)\*([^*\n]+)\*(?!\*)|(?<!_)_([^_\n]+)_(?!_)')
_MD_LIST_BULLET_RE = re.compile(r'^\s*[-*+]\s+', re.MULTILINE)
_MD_NUMBERED_LIST_RE = re.compile(r'^\s*\d+\.\s+', re.MULTILINE)
_MD_BLOCKQUOTE_RE = re.compile(r'^\s*>\s?', re.MULTILINE)
_WHITESPACE_RE = re.compile(r'\s+')


def strip_markdown(text):
    """Drop markdown syntax, keep the words. For BM25 indexing."""
    text = _MD_CODE_FENCE_RE.sub(lambda m: m.group(0).strip('`'), text)
    text = _MD_INLINE_CODE_RE.sub(r'\1', text)
    text = _MD_IMAGE_RE.sub(r'\1', text)
    text = _MD_LINK_RE.sub(r'\1', text)
    text = _MD_HEADING_RE.sub('', text)
    text = _MD_BOLD_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_ITALIC_RE.sub(lambda m: m.group(1) or m.group(2), text)
    text = _MD_LIST_BULLET_RE.sub('', text)
    text = _MD_NUMBERED_LIST_RE.sub('', text)
    text = _MD_BLOCKQUOTE_RE.sub('', text)
    text = _WHITESPACE_RE.sub(' ', text)
    return text.strip()


# --- Document assembly ---------------------------------------------------

def resolve_company(folder_name, override=None):
    if override:
        return override
    return COMPANY_ALIASES.get(folder_name.lower(), folder_name.replace('-', ' ').title())


def pick_url(metadata):
    for key in URL_KEYS:
        if metadata.get(key):
            return metadata[key]
    return ''


def load_markdown_file(path, company, subcategory=''):
    with open(path, 'r', encoding='utf-8') as f:
        raw = f.read()

    metadata, body = parse_front_matter(raw)

    title = (
        metadata.get('title')
        or extract_first_heading(body)
        or os.path.splitext(os.path.basename(path))[0].replace('-', ' ').title()
    )

    cleaned = clean_body(body, title)
    if len(cleaned) < MIN_BODY_LENGTH:
        return None

    url = pick_url(metadata)
    company_name = resolve_company(company, metadata.get('company'))
    id_seed = url or os.path.relpath(path, DATA_DIR)
    doc_id = hashlib.md5(id_seed.encode()).hexdigest()[:12]

    doc = {
        'id': doc_id,
        'company': company_name,
        'url': url,
        'title': title,
        'text': cleaned,
        'searchable_text': strip_markdown(f'{title}\n\n{cleaned}'),
    }
    if subcategory:
        doc['subcategory'] = subcategory
    for key in PASSTHROUGH_KEYS:
        if key in metadata:
            doc[key] = metadata[key]

    return doc


# --- Filesystem walk -----------------------------------------------------

def discover_markdown_files(data_dir):
    """Yield (file_path, top_level_folder, subcategory) for every .md under data_dir.

    The top-level folder always determines company, regardless of nesting depth.
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    for entry in sorted(os.listdir(data_dir)):
        company_root = os.path.join(data_dir, entry)
        if not os.path.isdir(company_root):
            continue
        for root, dirs, files in os.walk(company_root):
            dirs.sort()
            for name in sorted(files):
                if not name.lower().endswith('.md'):
                    continue
                file_path = os.path.join(root, name)
                rel = os.path.relpath(root, company_root)
                subcategory = '' if rel == '.' else rel.replace(os.sep, '/')
                yield file_path, entry, subcategory


# --- Entry point ---------------------------------------------------------

def build_corpus():
    docs = []
    seen_ids = set()
    skipped_empty = skipped_error = 0

    for path, folder, subcategory in discover_markdown_files(DATA_DIR):
        try:
            doc = load_markdown_file(path, folder, subcategory)
        except Exception as exc:
            print(f"[Corpus] Skipping {path}: {exc}")
            skipped_error += 1
            continue

        if doc is None:
            skipped_empty += 1
            continue

        if doc['id'] in seen_ids:
            doc['id'] = hashlib.md5((doc['id'] + path).encode()).hexdigest()[:12]
        seen_ids.add(doc['id'])

        docs.append(doc)

    docs.sort(key=lambda d: (d['company'], d.get('subcategory', ''), d['title']))

    out_path = os.path.join(DATA_DIR, 'corpus.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    _print_coverage(docs, skipped_empty, skipped_error, out_path)
    return docs


def _print_coverage(docs, skipped_empty, skipped_error, out_path):
    print(f"[Corpus] Built {len(docs)} documents.")
    if skipped_empty:
        print(f"[Corpus] Skipped {skipped_empty} files with empty/short bodies.")
    if skipped_error:
        print(f"[Corpus] Skipped {skipped_error} files that failed to parse.")

    by_company = {}
    for d in docs:
        by_company.setdefault(d['company'], []).append(d)

    for company in sorted(by_company):
        company_docs = by_company[company]
        print(f"\n  {company}: {len(company_docs)} docs")
        sub_counts = {}
        for d in company_docs:
            key = d.get('subcategory', '(root)')
            sub_counts[key] = sub_counts.get(key, 0) + 1
        for sub, count in sorted(sub_counts.items()):
            print(f"    {sub}: {count}")

    print(f"\n  Saved to: {out_path}")


if __name__ == '__main__':
    build_corpus()