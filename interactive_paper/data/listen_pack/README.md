# Listening pack — "真正听一下 TTS 念出来是什么样"

12 wavs pulled from `gate-data/audio_pool` (the frozen 600-query pool, OpenAI
`tts-1` voice `alloy`), picked as the worst query↔transcript corruption among
the 132 unique escalated ids in `gated_traces_v2`. Full gold text + what the
talker transcribed for each id: `cases.json`.

How to listen: open the wav, follow along in `cases.json` (`query` = what TTS
was asked to read, `transcript` = what MiniCPM heard).

## What to listen for, by failure category

| id | pool | category — what happened |
|---|---|---|
| q0208 | hard-knowledge | **BROKEN RENDER: 49 s of digital silence (peak=0).** The talker's "audio is silent" transcript was correct. Only broken wav in all 601 (Modal audit `figures/wav_audit.json`). |
| q0552 | trap | **Rare entity**: "Mustafa Adebayo Balogun" heard as "Mustapha Arabo Balogun". Does alloy actually pronounce it clearly? |
| q0250 | hard-knowledge | Rare entity: "Taurek" → "Turek". |
| q0271 | hard-knowledge | **Operator lost**: "Estimate 999 − 103" heard as "nine hundred ninety nine hundred and three". Does the TTS voice the minus at all? |
| q0164 | hard-knowledge | LaTeX formula: sqrt terms dropped, option digits corrupted (10-digit decimals). |
| q0212 | hard-knowledge | LaTeX: unilateral Z-transform; all four options garbled. |
| q0256 | hard-knowledge | Raw LaTeX in source text (`2-k\Omega`, `1-\muF`) — what does TTS even say here? |
| q0233 | hard-knowledge | Ordinary mishear: option content garbled. (Earlier "mojibake" claim was wrong — the `��` was a GBK console display artifact; the actual chars are normal U+201C/201D curly quotes.) |
| q0213 | hard-knowledge | **Not-transcription**: talker output "D) Mongolia" — it answered instead of transcribing. Audio itself is likely fine. |
| q0237 | hard-knowledge | Not-transcription: 96 s question; talker output an answer, not a transcript. |
| q0163 | hard-knowledge | Number formatting: "$815.50" → "eight hundred and fifty dollars and fifty cents" (the 1 dropped). |
| q0169 | hard-knowledge | **Control — benign**: sim score low only because digits were spelled out; content fully intact. The sim metric overstates corruption on rows like this. |

## Takeaway candidates (verify by ear)

1. File-level TTS is fine: 1/601 broken.
2. The real questions: (a) does alloy enunciate rare entities and math
   operators clearly enough that a human would transcribe them right?
   (b) how much of the "corrupted transcript" 64% is actually the talker
   failing to transcribe (answering / refusing) rather than mishearing?
