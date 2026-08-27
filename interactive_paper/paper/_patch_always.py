# -*- coding: utf-8 -*-
import io

edits = [
('sections/live.tex',
 r'''every tier's gain significant (paired bootstrap;
Figure~\ref{fig:dualview}).''',
 r'''every tier's gain significant (paired bootstrap;
Figure~\ref{fig:dualview}). A measured always-escalate arm closes the
curve at 66.5\% (P50 4.2\,s): the last 43 points of escalation buy
3.2 points, because past the gated tiers the binding constraint is the
transcription-uplink channel, not selection --- the same sessions
re-scored with gold-text expert answers reach 92.2\%.'''),
('sections/transfer.tex',
 r'''\;\textbf{+ gate @50\%} & \textbf{86.0} & \textbf{73.2} & \textbf{94.8} & \textbf{79.5} & \textbf{77.2} & \textbf{82.1} & \textbf{+18.5} \\
always-expert (ceiling) & 96.8 & 86.4 & 92.8 & 93.0 & 87.1 & 91.2 & +27.6 \\''',
 r'''\;\textbf{+ gate @50\%} & \textbf{86.0} & \textbf{73.2} & \textbf{94.8} & \textbf{79.5} & \textbf{77.2} & \textbf{82.1} & \textbf{+18.5} \\
always-escalate (measured) & 88.0 & 80.8 & 93.6 & 88.5 & 83.7 & 86.9 & +23.3 \\
always-expert (gold-text ceiling) & 96.8 & 86.4 & 92.8 & 93.0 & 87.1 & 91.2 & +27.6 \\'''),
('sections/transfer.tex',
 r'''reported in the text: 3.99$\to$4.35, no gain over random.}''',
 r'''reported in the text: 3.99$\to$4.35 gated, 4.76 measured
always-escalate, 4.96 gold ceiling; no gain over random.}'''),
('sections/transfer.tex',
 r'''the expert's whole advantage lands where the probe sends it
(69.6\%$\to$88.8\%, $p{<}.0001$), and always-escalate buys nothing
measurable over selective;''',
 r'''the expert's whole advantage lands where the probe sends it
(69.6\%$\to$88.8\%, $p{<}.0001$), and a measured live always-escalate
arm lands \emph{below} selective (93.6 vs.\ 94.8 at half the calls);'''),
]
for path, old, new in edits:
    t = io.open(path, encoding='utf-8').read()
    assert t.count(old) == 1, (path, old[:50], t.count(old))
    io.open(path, 'w', encoding='utf-8', newline='\n').write(t.replace(old, new))
    print('ok', path)
