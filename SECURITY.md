# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |

## Security Model

Autolab tools are **local CLI utilities** that read TSV files and git state. They do not:

- Make network requests
- Execute arbitrary code from input files
- Write outside the current working directory (except git operations in autoevolve)
- Require elevated privileges

### autoevolve Git Operations

`autoevolve` creates and switches git branches. Malicious `results.tsv` descriptions could theoretically inject git arguments if not properly escaped. All git arguments are passed as explicit positional parameters, never interpolated into shell strings.

## Reporting a Vulnerability

**Please DO NOT open public issues for security vulnerabilities.**

To report a vulnerability, [create a private security advisory](https://github.com/dean0x/autolab/security/advisories/new) on this repository.

### What to Include

1. **Description** of the vulnerability
2. **Steps to reproduce** (minimal example)
3. **Impact assessment** (what can an attacker do?)
4. **Suggested fix** (if you have one)

### Response Timeline

- **Initial response**: Within 48 hours
- **Triage and assessment**: Within 1 week
- **Fix timeline**: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next release cycle

### Disclosure Policy

- We will acknowledge your report within 48 hours
- We will provide regular updates on our progress
- We will notify you when the vulnerability is fixed
- We will credit you in the security advisory (unless you prefer anonymity)
- We follow **coordinated disclosure**: we will not disclose the vulnerability until a fix is available

## Scope

### In Scope

- Command injection via malformed TSV input
- Path traversal in file arguments
- Git argument injection in autoevolve
- Denial of service via crafted input files

### Out of Scope

- Issues in dependencies (report to upstream)
- Social engineering
- Vulnerabilities requiring local code execution (you already have shell access)

## Contact

For non-security issues, please use:
- GitHub Issues: https://github.com/dean0x/autolab/issues

For security issues, use the private reporting method above.

---

**Security is a priority.** We take all reports seriously and appreciate responsible disclosure.
