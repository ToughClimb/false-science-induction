# Raw Reviewer Response After ESOL

```json
{
  "claim_supported": "partial",
  "what_results_support": [
    "ESOL molecular solubility domain shows that targeted paired swaps induce a false scaffold-high solubility association.",
    "Both neural MLP and XGBoost closed loops allocate more experiments toward the low-solubility target scaffold under targeted swap.",
    "Selected target molecules remain genuinely low-solubility, confirming oracle contradiction.",
    "The effect generalises beyond the GFP/protein domain, supporting the cross-domain mechanism claim."
  ],
  "what_results_dont_support": [
    "Aggregate R2 degradation is visible in stronger ESOL configurations.",
    "A clear stealth-allocation tradeoff exists in ESOL.",
    "No formal ESOL audit-boundary table is provided yet.",
    "ESOL uses only one target scaffold and three seeds."
  ],
  "missing_evidence": [
    "ESOL audit-boundary analysis.",
    "Multiple molecular scaffold targets or second molecular binding axis.",
    "Longer-horizon ESOL closed-loop runs.",
    "Diagnostic guardrail such as scaffold-stratified residuals.",
    "More ESOL seeds."
  ],
  "confidence": "medium",
  "summary_verdict": "ESOL adds valuable cross-domain breadth and supports the core mechanism, but the broad claim remains partial because audit non-diagnosticity and robustness are unresolved."
}
```
