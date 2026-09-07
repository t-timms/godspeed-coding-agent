"""Secret detection and redaction — regex + entropy analysis.

4-layer secret protection:
1. File access deny rules (handled by permission engine)
2. Context cleaning before LLM sees content
3. Output filtering on LLM responses
4. Audit log redaction (handled by audit.redactor)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Compiled secret detection patterns: (regex, type_name)
SECRET_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # API keys with known prefixes (more specific patterns first)
    (re.compile(r"sk-ant-[a-zA-Z0-9\-]{20,}"), "anthropic_api_key"),
    (
        re.compile(r"sk\s*-\s*ant\s*-\s*[a-zA-Z0-9\-\s]{20,}", re.IGNORECASE),
        "anthropic_api_key_obfuscated",
    ),
    (re.compile(r"sk-[a-zA-Z0-9\-]{20,}"), "openai_api_key"),
    (re.compile(r"sk\s*-\s*[a-zA-Z0-9\-\s]{20,}", re.IGNORECASE), "openai_api_key_obfuscated"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_access_key"),
    (re.compile(r"ghp_[a-zA-Z0-9]{36}"), "github_pat"),
    (re.compile(r"gho_[a-zA-Z0-9]{36}"), "github_oauth"),
    (re.compile(r"github_pat_[a-zA-Z0-9_]{22,}"), "github_fine_grained"),
    (re.compile(r"glpat-[a-zA-Z0-9\-]{20,}"), "gitlab_pat"),
    (re.compile(r"xoxb-[0-9]{10,}-[a-zA-Z0-9]+"), "slack_bot_token"),
    (re.compile(r"xoxp-[0-9]{10,}-[a-zA-Z0-9]+"), "slack_user_token"),
    (re.compile(r"SG\.[a-zA-Z0-9_\-.]{20,}\.[a-zA-Z0-9_\-.]{20,}"), "sendgrid_api_key"),
    (re.compile(r"sq0[a-z]{3}-[a-zA-Z0-9\-_]{22,}"), "square_api_key"),
    (re.compile(r"SK[0-9a-fA-F]{32}"), "twilio_api_key"),
    (
        re.compile(
            r"HRKU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
            r"-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
        ),
        "heroku_api_key",
    ),
    (re.compile(r"key-[0-9a-fA-F]{32}"), "mailgun_api_key"),
    (re.compile(r"NRAK-[A-Z0-9]{27}"), "new_relic_api_key"),
    (
        re.compile(
            r"https://hooks\.slack\.com/services/"
            r"T[a-zA-Z0-9_]+/B[a-zA-Z0-9_]+/[a-zA-Z0-9_]+",
            re.IGNORECASE,
        ),
        "slack_webhook_url",
    ),
    (re.compile(r"pypi-AgEIcH[a-zA-Z0-9_\-]{20,}"), "pypi_api_token"),
    (re.compile(r"npm_[a-zA-Z0-9]{36}"), "npm_token"),
    # Private keys
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA )?PRIVATE KEY-----"), "private_key"),
    (re.compile(r"-----BEGIN OPENSSH PRIVATE KEY-----"), "openssh_private_key"),
    # Firebase service account (JSON with private_key field)
    (re.compile(r'"private_key"\s*:\s*"-----BEGIN PRIVATE KEY-----"'), "firebase_service_account"),
    # Generic patterns
    (
        re.compile(r"""(?:password|passwd|pwd)\s*[:=]\s*['"][^'"]{8,}['"]""", re.IGNORECASE),
        "password_assignment",
    ),
    (
        re.compile(r"""(?:api_key|apikey|api-key)\s*[:=]\s*['"][^'"]{10,}['"]""", re.IGNORECASE),
        "api_key_assignment",
    ),
    (
        re.compile(
            r"""(?:api_key|apikey|api-key|anthropic_key|openai_api_key)\s*[:=]\s*\S{10,}""",
            re.IGNORECASE,
        ),
        "api_key_assignment_unquoted",
    ),
    (
        re.compile(r"""(?:secret|token)\s*[:=]\s*['"][^'"]{10,}['"]""", re.IGNORECASE),
        "secret_assignment",
    ),
    # Bearer tokens — require minimum 20 char token to avoid false positives
    (re.compile(r"Bearer\s+[a-zA-Z0-9\-._~+/]{20,}=*", re.IGNORECASE), "bearer_token"),
    # Connection strings
    (
        re.compile(r"(?:postgres|mysql|mongodb)://[^\s]+:[^\s]+@[^\s]+"),
        "database_connection_string",
    ),
    # Hugging Face
    (re.compile(r"hf_[a-zA-Z0-9]{20,}"), "huggingface_token"),
    # JWT tokens
    (re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), "jwt_token"),
    # Azure
    (re.compile(r"DefaultEndpointsProtocol=https;[^\s]{20,}"), "azure_connection_string"),
    # Azure SAS tokens
    (re.compile(r"sig=[a-zA-Z0-9%]{20,}"), "azure_sas_signature"),
    # AWS secret key (generic assignment)
    (
        re.compile(
            r"""aws_secret_access_key\s*[:=]\s*['"]?[A-Za-z0-9/+=]{30,}['"]?""",
            re.IGNORECASE,
        ),
        "aws_secret_key",
    ),
    # Google Cloud
    (re.compile(r"AIza[0-9A-Za-z_-]{35}"), "google_api_key"),
    # Stripe
    (re.compile(r"(?:sk|pk)_(?:test|live)_[a-zA-Z0-9]{20,}"), "stripe_key"),
    # PKCS8 private key
    (re.compile(r"-----BEGIN (?:ENCRYPTED )?PRIVATE KEY-----"), "pkcs8_private_key"),
    # Discord
    (re.compile(r"[MN][A-Za-z\d]{23,}\.[\w-]{6}\.[\w-]{27,}"), "discord_bot_token"),
    # Datadog
    (
        re.compile(
            r"(?:datadog|dd)_(?:api|app)_key\s*[:=]\s*['\"]?"
            r"[a-zA-Z0-9]{32,40}['\"]?",
            re.IGNORECASE,
        ),
        "datadog_key",
    ),
    # Unquoted password assignments (common in configs)
    (
        re.compile(r"""(?:password|passwd|pwd)\s*[:=]\s*\S{8,}""", re.IGNORECASE),
        "password_unquoted",
    ),
    # SSH private key file paths referenced in text
    (
        re.compile(r"~?/\.ssh/id_(?:rsa|ed25519|ecdsa|dsa)\b"),
        "ssh_key_path",
    ),
    # PEM/P12/PFX certificate file paths
    (
        re.compile(r"\S+\.(?:pem|p12|pfx|jks|keystore)\b", re.IGNORECASE),
        "certificate_file_path",
    ),
    # AWS SigV4 authorization headers
    (
        re.compile(r"AWS4-HMAC-SHA256\s+Credential=[A-Za-z0-9/+=]+", re.IGNORECASE),
        "aws_sigv4_header",
    ),
    # Secrets in URL query parameters
    (
        re.compile(
            r"(?:[?&])(?:key|api_key|apikey|access_token|token|sig|signature)"
            r"=[A-Za-z0-9_\-]{16,}",
            re.IGNORECASE,
        ),
        "url_query_secret",
    ),
]

# Minimum entropy threshold for high-entropy string detection
ENTROPY_THRESHOLD = 4.5
ENTROPY_MIN_LENGTH = 20

# Redaction placeholder
REDACTED = "[REDACTED]"


@dataclass(frozen=True, slots=True)
class SecretFinding:
    """A detected secret with its location and type."""

    secret_type: str
    match: str
    start: int
    end: int


def detect_secrets(text: str) -> list[SecretFinding]:
    """Detect potential secrets in text using regex patterns and entropy.

    Returns:
        List of SecretFinding objects with type, match, start, end.
    """
    findings: list[SecretFinding] = []
    seen_spans: set[tuple[int, int]] = set()

    # Pattern-based detection
    for pattern, secret_type in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if span not in seen_spans:
                seen_spans.add(span)
                findings.append(
                    SecretFinding(
                        secret_type=secret_type,
                        match=match.group(),
                        start=match.start(),
                        end=match.end(),
                    )
                )

    # Entropy-based detection for unmatched high-entropy strings
    for match in re.finditer(r"[a-zA-Z0-9+/\-_]{20,}", text):
        span = (match.start(), match.end())
        if span in seen_spans:
            continue
        token = match.group()
        if len(token) >= ENTROPY_MIN_LENGTH and _shannon_entropy(token) >= ENTROPY_THRESHOLD:
            seen_spans.add(span)
            findings.append(
                SecretFinding(
                    secret_type="high_entropy_string",
                    match=token,
                    start=match.start(),
                    end=match.end(),
                )
            )

    return findings


def redact_secrets(text: str) -> str:
    """Redact all detected secrets in text, replacing with [REDACTED]."""
    findings = detect_secrets(text)
    if not findings:
        return text

    # Build segments once — O(k + n) instead of O(k·n) string slicing
    sorted_findings = sorted(findings, key=lambda f: f.start)
    parts: list[str] = []
    last_end = 0
    for finding in sorted_findings:
        if finding.start < last_end:
            # Overlapping finding — skip (already redacted by previous)
            continue
        parts.append(text[last_end : finding.start])
        parts.append(REDACTED)
        last_end = finding.end
    parts.append(text[last_end:])
    return "".join(parts)


def _shannon_entropy(data: str) -> float:
    """Calculate Shannon entropy of a string in bits per character."""
    if not data:
        return 0.0

    freq: dict[str, int] = {}
    for char in data:
        freq[char] = freq.get(char, 0) + 1

    length = len(data)
    entropy = 0.0
    for count in freq.values():
        probability = count / length
        if probability > 0:
            entropy -= probability * math.log2(probability)

    return entropy
