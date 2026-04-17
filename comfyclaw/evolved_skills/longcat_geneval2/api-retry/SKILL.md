---
name: api-retry
description: >-
  Handle transient API failures with retry logic and error classification to distinguish infrastructure failures from generation failures
license: MIT
metadata:
  cluster: "api_overload_errors"
  origin: "self-evolve"
---

# API Retry & Error Resilience

## When to Use
Trigger this skill when:
- Verifier returns HTTP 503, 429, 502, 504 errors
- Error messages contain "overload", "rate limit", "timeout", "service unavailable"
- External API calls fail with network errors
- Empty feedback with zero score (often indicates API failure, not bad generation)

## Core Strategy
**Classify errors first**: Distinguish between generation failures (bad prompt, wrong parameters) and infrastructure failures (overloaded services, network issues, timeouts).

## Implementation

### 1. Error Classification
Before attempting workflow fixes, check if the error is infrastructure-related:
```python
if error_message in ['', None] and score == 0.0:
    # Likely API timeout or overload, not generation failure
    classify_as = 'infrastructure_failure'
elif any(keyword in str(error).lower() for keyword in 
         ['overload', 'rate limit', '503', '502', '504', '429', 'timeout', 'unavailable']):
    classify_as = 'infrastructure_failure'
else:
    classify_as = 'generation_failure'
```

### 2. Retry Logic
For infrastructure failures:
- Wait with exponential backoff: 2s, 4s, 8s
- Retry up to 3 times
- If all retries fail, mark as 'skipped_transient_error' rather than 'failed'
- Log separately from generation failures for accurate metrics

### 3. Graceful Degradation
- Use cached verifier results if available
- Fall back to simpler verification (file existence, dimension check)
- Continue benchmark run rather than halt on transient errors

### 4. Reporting
When infrastructure failures occur:
- Tag with `error_type: 'transient_api_failure'`
- Exclude from generation quality metrics
- Report separately in benchmark summary
- Recommend: "API service overloaded - retry later" rather than workflow changes

## What NOT to Do
- Don't modify prompts or workflow parameters for API errors
- Don't count API failures as generation failures in skill performance metrics
- Don't trigger other skills (prompt-artist, controlnet-control, etc.) for infrastructure errors

## Integration
This skill should run **before** any generation-focused skills analyze failures. It acts as a filter to prevent misdiagnosis of transient errors as quality issues.