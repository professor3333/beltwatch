# Security Policy

BeltWatch is a portfolio project under active development. It is not
intended for production use, and its threat model is limited to the
deployment described in [`docs/design.md`](docs/design.md) §21.

## Reporting a vulnerability

Please report suspected vulnerabilities privately through
[GitHub private vulnerability reporting](https://github.com/professor3333/beltwatch/security/advisories/new)
rather than in a public issue. Include steps to reproduce and the affected
component (upload handling, job processing, API, or deployment).

Issues of particular interest include:

- unsafe handling of untrusted image uploads, such as decompression bombs or
  file-type spoofing
- path traversal in artifact retrieval
- unauthenticated access to audit data or `/metrics`
- leakage of uploaded images through logs
