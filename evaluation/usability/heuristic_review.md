# SignBridge live demo — heuristic review (author review, not user testing)

**Status: draft by the developer, to be checked by Manya before it is cited.** This is an expert-inspection
method (Nielsen's 10 usability heuristics) applied by a single evaluator who is also the developer, so it is
biased towards the developer's own mental model and finds fewer problems than 3–5 independent evaluators would.
It is **not** evidence about Deaf or hearing users and does not replace the study in `protocol.md`.

Method: each of the five scripted tasks in `protocol.md` (U1 greeting, U2 fingerspell a name, U3 calibrate,
U4 clinic check-in, U5 urgent message) was walked through in `live_demo.html`, and each problem was rated on
Nielsen's severity scale: 0 none · 1 cosmetic · 2 minor · 3 major · 4 catastrophe. Findings from the end-to-end
evaluation and the automated WCAG audit (`artifacts/eval/accessibility_audit.json`) are cross-referenced.

| # | Heuristic | Finding | Task | Severity |
|---|-----------|---------|------|----------|
| 1 | Visibility of system status | Good: status line, confidence bar, sign strip with countdown, recording countdown. Missing: when a letter is recognised but below the typing threshold, nothing says *why* it is not being typed. | U2, U3 | 2 |
| 2 | Match with the real world | Controls use developer terms — “Phrase sensitivity 0.22”, “Personalisation 60%”, “#MANYA” fingerspelling notation — that a first-time Deaf or hearing user would not understand. | U1–U3 | 2 |
| 3 | User control and freedom | Letters are typed automatically after a 1 s hold, so hesitating mid-word types unintended letters (end to end only ~61% of letters were typed correctly); there is no way to cancel a recording countdown once started. | U2, U3 | 3 |
| 4 | Consistency and standards | Tabs and reference tiles look like buttons but are `<div>`s that the keyboard cannot use; errors appear as browser `alert()` pop-ups while other feedback is in-page. | all | 2 |
| 5 | Error prevention | Recognised phrases are spoken immediately with no confirm step, including urgent ones — risky given the smoother invents sentences from garbled letters and Whisper heard “Call an ambulance now” as “Cool and ambulance now”. | U5 | 3 |
| 6 | Recognition rather than recall | Good: the reference grid and the phrase list show each handshape and how to sign it. Built-in sign instructions are hidden in the Phrases tab while signing happens in the centre panel. | U1 | 1 |
| 7 | Flexibility and efficiency | Good: signer profiles, calibration, avatar speed and TTS toggles. No keyboard shortcuts for Space / Backspace while the hands are busy signing. | U2 | 1 |
| 8 | Aesthetic and minimalist design | One dense screen: six controls in the signer bar plus three columns; heavy for a first call, and the most important output (typed message) sits below the video. | U1, U4 | 2 |
| 9 | Help users recognise and recover from errors | When recognition fails the language model produces a fluent wrong sentence (“The letter H stands for Hello”) instead of asking the signer to repeat; dialog messages such as “Not enough hand frames” do not say what to change. | U2, U5 | 3 |
| 10 | Help and documentation | Explanations live in a note at the bottom of the page; there is no first-run guidance on camera position, lighting or calibration. | U3 | 2 |

**Summary:** 10 issues — 3 major (severity 3: automatic typing without control, no confirmation for urgent
messages, invented sentences instead of a repeat request), 5 minor, 2 cosmetic; strengths in system status,
recognition-over-recall and flexibility.

**Design changes these point to** (not yet made): show “hold steady…” when a letter is below threshold; a
confirm-before-speak step for urgent phrases; let the smoother say “please repeat” instead of guessing; plain-
language labels with the numeric controls moved to an advanced panel; real buttons with ARIA labels and an
aria-live region for typed output.
