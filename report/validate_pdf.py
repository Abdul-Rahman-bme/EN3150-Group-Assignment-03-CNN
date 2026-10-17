"""Check the compiled report with pypdf and the LaTeX build log."""
import hashlib
import json
from pathlib import Path

from pypdf import PdfReader

root = Path(__file__).resolve().parent
pdf = root / '54_A03_EN3150.pdf'
reader = PdfReader(pdf)
pages = len(reader.pages)
text = '\n'.join(page.extract_text() or '' for page in reader.pages)
assert pages >= 15
for expected in ['230507R', '230521E', '230544C', '230108U', '54', '93.90',
                 '94.37', '93.63', '98.12', '96.37', '46,373', '15,442',
                 'Model D final result verification']:
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
    'group': 54, 'compiled_with': 'pdfLaTeX + BibTeX (MiKTeX on Windows; report/build.sh-equivalent sequence)',
    'compilation_exit_status': 0,
    'checks': ['Four member indices present', 'Five-model verified comparison metrics present',
               'Group placeholder absent', 'No unresolved citation markers', 'No overfull text boxes'],
    'visual_review': 'Master comparison, Model D confusion/curves/per-class results, CompactEuroSAT evidence wording and audit pages inspected',
    'status': 'Compiled evidence-audited draft; controlled compact run remains blocked',
}
(root / 'audit/pdf_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
