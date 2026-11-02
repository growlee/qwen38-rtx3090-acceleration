# External backup interrupted the repaired-profile capture

The KV4350 unit stopped at 2026-09-10 00:37:09 UTC with
`root_storage_reserve_exhausted`, after 21 complete responses and the start of
`context-version-end-t32768`. This is an infrastructure interruption, not a
new model OOM or quality failure. The complete 21 responses remain immutable.
The inherited controller incorrectly labelled the whole mode unattempted
after its exception; raw case envelopes are the authoritative dispatch proof.

Root free space fell from 167340507136 bytes to 153615032320 at the abort,
then 145152282624 on a fresh read; owned experiment allocation was only
643072 bytes. It recovered to 167339081728 bytes without any cleanup by this
task. Candidate e5aec0877b5d576bdf4395cedbab12c157c537c213d26f9bf54bfb2040857a97
was stopped; client 5708e673f43d5fc29ef5cc11ca0b29b9f1eaa19179ce65bead9faca200fb12fd
was terminated by the guard. Qwen remained stopped/admission closed because
the same storage fence also prevented the ordinary restoration command.

Read-only attribution from the Homelab task
`01a02331-4785-7dd3-9d40-1e755db8e7dd`, reported 2026-09-10 00:41 UTC:

- Existing Server B `homelab-backup.timer` triggered at 00:35:11 UTC,
  invocation `8a414f16aa9648cd9570ddadc86fb77c`.
- Server A sudo log confirms `/usr/local/sbin/homelab-backup-export server-a`
  at 00:35:11.922245. It writes an uncompressed temporary AI archive on root,
  including the 21474836480-byte `unsloth_studio.ext4` image. The exact peak
  temporary archive size was not measured.
- Forgejo's 00:35:14 stop and 00:35:28 start were part of that backup.
- At 00:37:56 the AI export failed because
  `/srv/ai-server/state/unsloth_studio.ext4` changed during export. The backup's
  own existing cleanup removed its incomplete temporary archive; the service
  ended failed at 00:38:31. This was not a successful full AI backup.
- Next live timer trigger was 2026-09-11 01:04:32 UTC. The daily schedule is
  00:20 UTC plus up to 45 minutes randomized delay, with persistent catch-up.

No backup timer, backup data, service policy, or unrelated application was
changed. Preserved responses overlapping 00:35:11–00:38:31 UTC must retain
their quality evidence but must not be presented as isolated timing samples.
Restoration uses the existing exact-identity helper through a separately
reviewed one-shot adapter. Only the 63 missing terminal responses may continue.
