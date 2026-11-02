# One-shot restoration review

Same independent Sol critic as the r2 launch review; no second review pipeline.
Candidate: `restore_r2_once.py`, SHA-256
`509810a8bbec9cafee7841823ccb43e580ed884b9af52c445b07c493bb41a946`.

Initial exact-transfer review rejected a hash mismatch before dispatch.
Read-only byte inspection showed a PowerShell-appended CRLF after the existing
LF. Removing those two extra bytes from the assistant-owned staged copy restored
the exact reviewed SHA. No restoration command had run before this correction.

Final verdict: APPROVE. The distinct fixed idempotency key, exclusive receipt,
nonblocking existing host/CAS locks, exact stopped ownership, original production
identity, admission/idle/disk checks and inherited restore sequence were reviewed.

Dispatched once as `qwen-context-20260908-r2-restore.service`, invocation
`ece613196a92450bb0652e6936d177e5`, Restart=no, RuntimeMaxSec=2700.
This recovery never dispatches a benchmark, removes data, or changes a model.
Completion is established only by the separate terminal restoration receipt.
