# =============================================================================
# RENDER ALL THIRTEEN THESIS FIGURES AND SAVE THEM TO DRIVE
# One cell. Mounts Drive, pulls the latest figure code and the experiment
# results, renders, verifies all thirteen exist, then writes them back.
# Every value is read from a stored result file. Nothing is typed in.
# =============================================================================
import os, sys, shutil, pathlib, subprocess, time

# REMOVED for custody: mounted Google Drive. See accord/cloud_paths.py.
# REMOVED for custody: mounted Google Drive. See accord/cloud_paths.py.

DRIVE  = pathlib.Path(os.environ.get('ACCORD_FIG_OUT', os.path.expanduser('~/mimic_local_only/accord_figures')))
BUNDLE = DRIVE / '02_figures' / 'colab_bundle'
DEST   = DRIVE / '02_figures'
WORK   = pathlib.Path('/content/figures')
REPO   = pathlib.Path('/content/repo')

# ---- 1. fresh copy of the code and the data -------------------------------
for d in (WORK, REPO):
    shutil.rmtree(d, ignore_errors=True)
shutil.copytree(BUNDLE, WORK)
REPO.mkdir(parents=True, exist_ok=True)
shutil.copytree(WORK / 'results', REPO / 'results')

# The scripts were written to run from the repository, so they carry an
# absolute path to it. Point that at the copy we just made.
for p in WORK.glob('*.py'):
    p.write_text(p.read_text().replace('/Users/rickygole/provenance-encoder', str(REPO)))

os.chdir(WORK)
n_data = sum(1 for _ in (REPO / 'results').rglob('*') if _.is_file())
print(f"code   : {len(list(WORK.glob('*.py')))} scripts")
print(f"data   : {n_data} result files\n")

# ---- 2. render -------------------------------------------------------------
for f in WORK.glob('fig-*.pdf'):
    f.unlink()                       # start clean so nothing stale survives

SCRIPTS = ['make_figures.py', 'fig_batch2.py', 'fig_threshold_surface.py',
           'make_figures_extra.py', 'fig_spec_curve.py', 'fig_gap_trajectory.py']
t0 = time.time()
for s in SCRIPTS:
    if not (WORK / s).exists():
        continue
    r = subprocess.run([sys.executable, s], capture_output=True, text=True)
    print(f"  {'ok  ' if r.returncode == 0 else 'FAIL'} {s}")
    if r.returncode != 0:
        print('       ' + (r.stderr.strip().splitlines() or ['?'])[-1][:150])
print(f"\nrendered in {time.time()-t0:.0f}s\n")

# ---- 3. verify, then save --------------------------------------------------
USED = ['fig-pipeline-status', 'fig-intro-headline', 'fig-rule-precision',
        'fig-prevalence-response', 'fig-falsification', 'fig-power', 'fig-forest',
        'fig-disc-strata', 'fig-scaling', 'fig-threshold-surface',
        'fig-disc-actionability', 'fig-met-exchange', 'fig-exemplar']

missing = [n for n in USED if not (WORK / f'{n}.pdf').exists()]
for n in USED:
    print(f"  {'ok      ' if (WORK/f'{n}.pdf').exists() else 'MISSING '} {n}")

if missing:
    print(f"\n{len(missing)} of {len(USED)} missing. Nothing saved.")
else:
    for n in USED:
        shutil.copy(WORK / f'{n}.pdf', DEST / f'{n}.pdf')
    for extra in WORK.glob('fig-*.pdf'):          # the unplaced ones too
        shutil.copy(extra, DEST / extra.name)
    print(f"\nall {len(USED)} figures saved to {DEST}")

# ---- 4. look at them -------------------------------------------------------
subprocess.run(['apt-get', '-qq', 'install', 'poppler-utils'], capture_output=True)
from IPython.display import display, Image
for n in USED:
    png = WORK / f'{n}.png'
    subprocess.run(['pdftoppm', '-png', '-r', '120', '-singlefile',
                    str(WORK / f'{n}.pdf'), str(png)[:-4]], capture_output=True)
    if png.exists():
        print(f"\n{'='*70}\n{n}\n{'='*70}")
        display(Image(str(png)))
