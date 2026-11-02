# FreeToken 0.1.3 / Qwen3.8-27B NVFP4

**PASS: model loaded; loopback endpoint verified; 5/5 streaming cases passed.**

TTFT fresh → continuation: 8k **12.424 → 1.903 s** (8023 cached tokens); 28k **27.883 → 0.214 s** (28041 cached tokens). Peak VRAM 23394 MiB; no OOM.

FreeToken is installed and stopped after the test. Original Qwen remains stopped exactly as before; gateway healthy, admission public. No permanent routing/autostart change.

See [REPORT.md](REPORT.md) for actual model/revision, JIT caveats, raw evidence, restoration and the exact guarded relaunch command. Previous RAM-blocked report is retained in before-resume-report/.
