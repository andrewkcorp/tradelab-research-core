# Changelog

## 0.1.0a1 — public-core release candidate

- Extracted the generic research engine and synthetic regression suite from a
  pinned private development baseline, with allowed-path history and without private research data.
- Added a real-engine synthetic example distinct from the fabricated dashboard
  demonstration, and documented causality/fill/accounting limitations.
- Prepared contributor, security, data, governance and AI-review policies.
- Added source provenance, publication checks, packaging and a CI definition.
- Adapted the research-period policy's resource path for installed packages;
  bundled dates are explicitly illustrative. The no-generated-data test also
  checks source archives without requiring or inspecting a parent Git checkout.
  These adaptations do not change strategy or simulation rules; the manifest
  records original and exported hashes.

This is a release candidate until the human maintainer approves licensing and
publishes it. CI configuration does not imply a successful hosted CI run. Public
adoption, performance profitability and broker execution are not claimed.
