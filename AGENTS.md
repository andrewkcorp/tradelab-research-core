# TradeLab public-core agent instructions

Research only: no broker integration, credentials, live order routing or paid
market-data downloads. Do not publish, push, merge, change repository visibility,
submit applications or incur API charges without explicit human authorization.

- Read methodology before changing feature/entry/execution/accounting contracts.
- Preserve `known_at` causality, next-eligible-open entry, stop-first OHLC
  ambiguity policy and complete higher-timeframe aggregation.
- Add synthetic regression and adverse/boundary tests for every rule change.
- Keep experiments/selection criteria explicit. Never optimize against a holdout
  while describing it as untouched out-of-sample validation.
- No real dataset, credentials, account log or user-home path in tracked files.
- Do not silently change tests to match incorrect behaviour or fabricate evidence.
- Run `python -m unittest discover -s tests -t . -q` and
  `python tools/check_public_tree.py --tree .`; disclose failures and skips.
- Changes to a public module make its original SOURCE_MANIFEST hash historical
  provenance, not a hash of the updated module. Explain modifications in CHANGELOG.
- No new runtime dependency without maintainer agreement.
- AI-assisted changes require human review; agents cannot approve their own work.
