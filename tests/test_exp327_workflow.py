from pathlib import Path
W=Path(".github/workflows/exp327-iterative-minimal-subset.yml")
def text():return W.read_text()
def test_manual_no_inputs_and_authority_binding():
 t=text();assert "workflow_dispatch:" in t and "inputs:" not in t
 for x in ("35409935457","10573847041","92794024c87ff9fd5d1e7131c2a7404c6d843abc74b825bb96d68d6f952ebdc4","35345351869","10547681681","4aa03459b5266a3455bcfbc8cb070d9ceaeb0e7b8390944e7d483b0953e567c5","b5ccf5bb9e709c8dbc4da33e87a5bbdcaa1164f3408787f1ae423bae7a7003ed"):assert x in t
def test_no_scientific_user_controls_or_scale():
 t=text().lower()
 for x in ("--model-size","--learning-rate","--seed","30000000","100000000"):assert x not in t
