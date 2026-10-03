# Testing

## Automated tests

```bash
python -m pytest -q
```

107 tests run against a scripted fake model, so they check the harness itself
(verification, phase gates, memory, grounding, escalation, consent) without an API
key. `tests/test_engine.py` also replays every bug found during live testing.

## Test identities

| Caller | DOB | ID last 4 | Claims |
|---|---|---|---|
| Margaret Chen | 1985-03-15 | 4472 | 4, including **CL-2048**: healthcare, denied, January 2026 |
| Ava Lopez | 1990-08-21 | 9180 | none |
| Ma Tian | 1964-09-10 | 6688 | CL-3001: healthcare, denied |
| Ya Wen Li (alias Yaven Li) | 1989-12-03 | 5317 | none |
| David Chen | representative for Margaret Chen | | Margaret's, with her consent |

## Scenarios to try

| Try | Expected |
|---|---|
| **Margaret (demo case)** button | Verified in one turn; CL-2048 confirmed from the remembered hint; denial explained; deadline passed, so a human is offered; email summary at the end |
| Name only, then **Frustrated caller** button | Acknowledges, explains why verification matters, no claim details |
| "What is RL?" three times | Declined twice, transferred on the 3rd |
| Verify as Ava, then "I want to file a new claim" | No claims on file; a human is offered for the new claim |
| **Representative** button | Consent request; answer it as Margaret with the pop-up in the inspector |
