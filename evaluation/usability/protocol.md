# SignBridge usability study — protocol (ready to run)

Status: **protocol and scoring ready; not yet run with participants.** No usability results may be
reported until this study has been run under ethics approval.

## Participants (representative users)
- 5 Deaf / hard-of-hearing ASL users (primary users). Recruit through a Deaf community organisation; pay an
  hourly rate; a qualified ASL interpreter is present for consent and debriefing.
- 5 hearing non-signers (secondary users: family members, receptionists, clinicians).
- Record, with consent: hearing status, preferred language, dominant signing hand, ASL fluency, age band,
  skin-tone band (Monk scale, self-reported, optional) so accuracy can be broken down fairly.

## Setup
Laptop webcam + built-in microphone, ordinary room lighting, `live_demo.html` served locally, multi-signer
model selected, one fresh signer profile per participant. Pairs: one Deaf + one hearing participant.

## Tasks (scripted scenarios, counterbalanced order)
| ID | Scenario | Deaf participant | Hearing participant | Success criterion |
|----|----------|------------------|---------------------|-------------------|
| U1 | Greeting | signs HELLO → HOW → YOU, then ME → GOOD | reads/hears the message, replies by voice | hearing partner repeats the meaning correctly |
| U2 | Fingerspell a name | fingerspells own first name | says the name back | name correct |
| U3 | Calibration | calibrates 5 letters that failed in U2, repeats U2 | — | name correct after calibration |
| U4 | Clinic check-in | reads captions + avatar of "What is your date of birth?" and answers | asks the question by voice | answer matches the question |
| U5 | Urgent message | signs/fingerspells HELP / "CALL A DOCTOR" | acts on it | correct action in < 60 s |

## Measures
- Task success (binary), time on task, number of repeats/corrections (from screen recording).
- Recognition accuracy per participant: letters typed vs letters intended (live logs).
- SUS (10 items, 1–5) after the session — pass mark 68. `score_usability.py` computes it.
- Comprehension (hearing participants): proportion of signed messages understood correctly — target ≥ 80%.
- Avatar acceptability (Deaf participants, 1–5): "the avatar's signing was understandable" and "I would want
  an avatar in a real call" — plus open comments. Participants may reject the avatar outright; that is a result.
- Semi-structured debrief (10 min): what felt slow, what felt wrong, what should be removed.

## Ethics
Informed consent in English and ASL; right to withdraw; video is analysed then deleted; only landmarks and
typed text are stored; participants are told the tool assists, never replaces, an interpreter.
