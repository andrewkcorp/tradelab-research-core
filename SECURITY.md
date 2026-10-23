# Security policy

This is an early-stage research tool, not a hardened service or live trading
system. Only the current public branch is maintained; there is no support SLA.

## Report privately

Use this repository's **Security → Report a vulnerability** when private
vulnerability reporting is enabled. If that control is unavailable, open an issue
that only asks the maintainer to enable a private reporting channel; do not include
the vulnerability, secrets, account data or exploit details in the public issue.

Include an affected version, a minimal synthetic reproduction, impact, and a
suggested mitigation. Do not probe systems or repositories you do not control.

## Local operation

- Keep the dashboard bound to `127.0.0.1`; it has no authentication or TLS.
- Load only trusted local research artifacts. In particular, never unpickle an
  untrusted cache or treat a downloaded database as safe executable input.
- Do not point the dashboard at directories containing account or vendor data
  that you are not willing to expose to the local client.
- Keep secrets outside the repository and rotate any credential that was exposed.
  Deleting a line in the current tree does not remove it from Git history.
- The bundled publication checker is heuristic. It is not a penetration test,
  comprehensive secret detector or certification.

Maintainers must enable GitHub private reporting and available secret scanning
before opening the repository publicly. Security improvements must not claim to
protect broker positions; there is no live broker adapter in this repository.
