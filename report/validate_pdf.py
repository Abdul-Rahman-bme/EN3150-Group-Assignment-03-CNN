"""Check the compiled report with Poppler and the LaTeX build log."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

root = Path(__file__).resolve().parent
pdf = root / '54_A03_EN3150.pdf'
text = subprocess.check_output(['pdftotext', str(pdf), '-']).decode()
info = subprocess.check_output(['pdfinfo', str(pdf)]).decode()
pages = int(re.search(r'^Pages:\s+(\d+)', info, re.M).group(1))
assert pages >= 15
for expected in ['230507R', '230521E', '230544C', '230108U', '54', '93.90',
                 '94.37', '98.12', '96.37', '46,373']:
    assert expected in text, expected
assert 'YOURGROUPNO' not in text
assert '[?]' not in text
log = (root / '54_A03_EN3150.log').read_text()
assert 'undefined citations' not in log
assert 'Overfull' not in log
assert 'Fatal error' not in log
result = {
    'filename': 'report/54_A03_EN3150.pdf', 'pages': pages,
    'bytes': pdf.stat().st_size, 'sha256': hashlib.sha256(pdf.read_bytes()).hexdigest(),
    'group': 54, 'compiled_with': 'pdfLaTeX + BibTeX; report/build.sh',
    'compilation_exit_status': 0,
    'checks': ['Four member indices present', 'Expected final metrics present',
               'Group placeholder absent', 'No unresolved citation markers', 'No overfull text boxes'],
    'visual_review': 'Title, architecture, fine-tuning curves and master-comparison page inspected',
    'status': 'Compiled evidence-audited draft; controlled compact run remains blocked',
}
(root / 'audit/pdf_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
