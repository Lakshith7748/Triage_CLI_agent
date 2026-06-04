"""CSV runner for the triage triage_agent.

Reads support_tickets/support_tickets.csv, runs every row through the
triage_agent, writes support_tickets/output.csv with all 5 output fields.

Failure policy: any exception during processing -> escalate. Never
return an LLM-generated answer the triage_agent isn't confident in.

CLI:
    python main.py
    python main.py --limit 5       # smoke test on first N rows
    python main.py --dry-run       # routing only, no API calls
    python main.py --input X --output Y
"""

import os
import sys
import csv
import json
import time
import argparse

try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    load_dotenv(_env_path)
except ImportError:
    pass

from triage_agent import TriageAgent


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(BASE_DIR, '..', 'support_tickets', 'support_tickets.csv')
DEFAULT_OUTPUT = os.path.join(BASE_DIR, '..', 'support_tickets', 'output.csv')

OUTPUT_COLUMNS = ['issue', 'subject', 'company',
                  'response', 'product_area', 'status', 'request_type', 'justification']

NULL_COMPANY_VALUES = frozenset({'', 'none', 'null', 'nan', 'na', 'n/a'})


# --- Input ---------------------------------------------------------------

def normalize_company(raw):
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if cleaned.lower() in NULL_COMPANY_VALUES:
        return None
    return cleaned


def read_tickets(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Input CSV not found: {path}\n"
            f"Pass --input to specify a different path, or place the file at {DEFAULT_INPUT}"
        )
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"Input CSV is empty: {path}")

        normalized_field_map = {name.lower().strip(): name for name in reader.fieldnames}
        if 'issue' not in normalized_field_map:
            raise ValueError(
                f"Input CSV is missing required 'issue' column. "
                f"Found columns: {list(reader.fieldnames)}"
            )

        rows = []
        for raw_row in reader:
            rows.append({
                'issue': (raw_row.get(normalized_field_map.get('issue', 'issue')) or '').strip(),
                'subject': (raw_row.get(normalized_field_map.get('subject', 'subject')) or '').strip(),
                'company': normalize_company(raw_row.get(normalized_field_map.get('company', 'company'))),
            })
        return rows


# --- Failure policy ------------------------------------------------------

def make_error_escalation(reason):
    return {
        'status': 'escalated',
        'product_area': 'unknown',
        'response': 'Escalated to a human triage_agent.',
        'justification': f'Agent error during processing: {reason}. Escalated to avoid an ungrounded reply.',
        'request_type': 'bug',
    }


def safe_handle(triage_agent, ticket):
    try:
        return triage_agent.handle(
            issue=ticket['issue'],
            subject=ticket['subject'],
            company=ticket['company'],
        )
    except Exception as exc:
        return make_error_escalation(f'{type(exc).__name__}: {exc}')


# --- Progress ------------------------------------------------------------

def render_progress_bar(done, total, width=24):
    if total == 0:
        return '[no rows]'
    filled = int(width * done / total)
    return f"[{'#' * filled + '.' * (width - filled)}] {done}/{total}"


def print_progress(idx, total, ticket, result, start_time):
    elapsed = time.time() - start_time
    bar = render_progress_bar(idx, total)
    company = ticket['company'] or '-'
    area = result['product_area'] if result['product_area'] is not None else '-'
    issue_preview = ticket['issue'][:60].replace('\n', ' ')
    if len(ticket['issue']) > 60:
        issue_preview += '...'
    print(
        f"{bar} {elapsed:5.1f}s  "
        f"{result['status']:9} {result['request_type']:14} "
        f"area={area:20} "
        f"co={company:10} | {issue_preview}",
        flush=True,
    )


# --- Output --------------------------------------------------------------

def write_output(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: (row.get(col) if row.get(col) is not None else '')
                             for col in OUTPUT_COLUMNS})


def print_summary(results, elapsed):
    total = len(results)
    replied = sum(1 for r in results if r['status'] == 'replied')
    escalated = sum(1 for r in results if r['status'] == 'escalated')
    errors = sum(1 for r in results if 'Agent error' in (r.get('justification') or ''))

    by_request_type = {}
    by_product_area = {}
    for r in results:
        by_request_type[r['request_type']] = by_request_type.get(r['request_type'], 0) + 1
        area = r['product_area'] if r['product_area'] is not None else '(none)'
        by_product_area[area] = by_product_area.get(area, 0) + 1

    print()
    print('=' * 60)
    print(f"Processed {total} tickets in {elapsed:.1f}s")
    print(f"  Replied:   {replied}")
    print(f"  Escalated: {escalated}")
    if errors:
        print(f"  (of which triage_agent errors: {errors})")
    print()
    print("  By request_type:")
    for k, v in sorted(by_request_type.items(), key=lambda kv: -kv[1]):
        print(f"    {k:18} {v}")
    print()
    print("  By product_area:")
    for k, v in sorted(by_product_area.items(), key=lambda kv: -kv[1]):
        print(f"    {k:25} {v}")
    print('=' * 60)


# --- Dry run -------------------------------------------------------------

class DryRunAgent:
    """Stand-in for TriageAgent that mirrors routing without calling the LLM."""

    def __init__(self, real_agent):
        self.real_agent = real_agent

    def handle(self, issue, subject='', company=None):
        from triage_agent import classify_chitchat, _looks_like_off_topic_question, assess_risk

        query = f'{subject}\n\n{issue}' if subject else issue
        risk = assess_risk(query)

        if risk == 'high':
            hits = self.real_agent.retriever.search(query, top_k=3, company=company)
            top = hits[0] if hits else None
            return self.real_agent._escalate_high_risk(top, risk)

        chitchat_category = classify_chitchat(issue)
        if chitchat_category is not None:
            return self.real_agent._chitchat_reply(chitchat_category)

        hits = self.real_agent.retriever.search(query, top_k=3, company=company)
        top = hits[0] if hits else None

        if top and top['confident']:
            return {
                'status': 'replied',
                'product_area': top['product_area'],
                'response': f'[DRY RUN] Would call LLM with top doc: {top["doc"]["title"]}',
                'justification': f'[DRY RUN] Risk: {risk}. BM25 score={top["score"]:.2f}; would generate grounded reply.',
                'request_type': 'product_issue',
            }
        if not _looks_like_off_topic_question(issue):
            return self.real_agent._escalate(top, risk)
        return self.real_agent._out_of_scope_reply(top)


# --- Entry point ---------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='Run the triage triage_agent over a CSV of support tickets.')
    parser.add_argument('--input', default=DEFAULT_INPUT, help='Input CSV path')
    parser.add_argument('--output', default=DEFAULT_OUTPUT, help='Output CSV path')
    parser.add_argument('--limit', type=int, default=None, help='Process only the first N rows')
    parser.add_argument('--dry-run', action='store_true', help='Skip LLM calls; for retrieval sanity checks')
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Reading tickets from {args.input}")
    tickets = read_tickets(args.input)
    if args.limit:
        tickets = tickets[:args.limit]
    print(f"Loaded {len(tickets)} ticket(s).")

    print("Initializing triage triage_agent...")
    real_agent = TriageAgent()
    triage_agent = DryRunAgent(real_agent) if args.dry_run else real_agent
    if args.dry_run:
        print("  (DRY RUN — no LLM calls will be made.)")
    print()

    rows = []
    start_time = time.time()
    for i, ticket in enumerate(tickets, start=1):
        result = safe_handle(triage_agent, ticket)
        merged = {**ticket, **result}
        rows.append(merged)
        print_progress(i, len(tickets), ticket, result, start_time)

    elapsed = time.time() - start_time

    write_output(args.output, rows)
    print(f"\nWrote {len(rows)} rows to {args.output}")
    print_summary(rows, elapsed)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)