# Data and redistribution policy

No vendor OHLCV, broker statements, account identifiers, paid news feeds,
private strategy exports or private research runs are distributed here.
The MIT code license does not license third-party datasets or exchange content.

The quickstart uses programmatically generated, hand-specified synthetic bars.
The dashboard demo uses a fixed-seed random generator. Both are intentionally
labelled; neither is evidence of actual trading performance.

For a data-related issue, submit a minimal synthetic reproduction. If a public
dataset is essential, document its source, license, redistribution permissions,
coverage, time zone, provenance, modifications and checksum before proposing it.
Do not assume that buying data gives permission to republish it, or that a
derived file automatically loses restrictions on its source data.

Keep your own market data and generated results outside the repository. The
gitignore is defence in depth only: ignored files can still be force-added, and
Git history can retain a file after deletion. Review the actual staged files.

Never publish credentials or account data in issues, screenshots, caches, logs,
run manifests or example commands. Review absolute paths in generated manifests
before sharing. Data loaders are local-file tools, not authorized vendor clients.
