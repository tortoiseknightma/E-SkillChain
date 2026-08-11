# Single-shard failure recovery fixes

## Root cause

The failed shard was a transient provider pre-response failure. The matrix
launcher then returned before finalizing successful siblings, and the monitor
restarted the whole ready stage. Once the failed query reached its persisted
attempt limit, those restarts made no progress and consumed the restart budget.

## Implemented safeguards

- Finalize clean parallel siblings before returning a stage failure; the next
  invocation therefore schedules only the failed shard.
- Stop the monitor when a restart produces no new completed shard or settled
  provider call.
- Validate runtime-lock and parallel-profile source hashes in matrix, direct
  shard, and monitor entry points before any provider reservation.
- Preserve a safe provider exception class in pre-response attempt receipts.
- Use a read-only Windows kernel liveness query in the SQLite aggregate gate;
  do not call `os.kill(pid, 0)` from worker cleanup.

The existing v22 root is intentionally not resumed: its v26 runtime lock does
not match the current runner/shard sources. A newly locked runtime/profile is
required for a recovery run.

## Reproduction of the remaining `dm-089` failure

The failure was reproduced twice with fresh ledgers and a direct, single-shard
worker, independent of matrix supervision:

- `runs/portfolio/portfolio-dev-mini-200x5-execution-v28-recovery`
- `runs/portfolio/portfolio-dev-mini-200x5-execution-v29-recovery`

Both runs stopped at Full shard 16, `dm-089`, during `assistant_action` after a
budget reservation. The provider response was HTTP 400 with the structured
error code `Arrearage` and the message that access was denied because the
DashScope account was not in good standing. The captured response is retained
in `runs/portfolio/provider-error-diagnostic.json`.

This is not a Full-only request-construction defect: the Full request's
`wire_request_sha256` (`4600fcd0e489b5d8cf62caca14001f64d1fd9d23534864ce5f68b463e9accdab`) is identical to the already successful
S1+S2 `dm-089` action request. The root cause is therefore the provider account
billing state, not the image, route, treatment bank, or algorithm. The current
fail-closed behavior correctly avoids writing a score, but leaves the reserved
call unresolved because a non-retryable provider error has no settlement
receipt. Do not retry this phase until the provider account is restored (or an
authorized alternate provider/account is configured); otherwise every retry
would be expected to fail with the same external condition.
