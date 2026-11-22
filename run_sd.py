"""run_sd (22 Nov 2026): runs a slice of jobs.json through stable-diffusion.cpp (FLUX.2 klein 4B GGUF) on one CPU machine.
  python3 run_sd.py jobs.json refs|shots part n models_dir quant out_dir
refs: each ref is redrawn with the next seed (up to 4) while its face area is skin-coloured (PROMPT_RULES 12)."""
import json, os, subprocess, sys, time
from PIL import Image
J, stage, part, n, M, Q, O = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6], sys.argv[7]
B = os.path.dirname(os.path.abspath(J)); allj = json.load(open(J)); jobs = ([j for j in allj['refs'] + allj['shots'] if j['id'] in stage.split(':')[1].split(',')] if stage.startswith('only:') else allj['refs'] if stage == 'refs' else [j for j in allj['shots'] if stage == 'shots' or (stage == 'free' and (j['ref'] in (None, 'c_extra.png'))) or (stage == 'named' and j['ref'] not in (None, 'c_extra.png'))])[part::n]
os.makedirs(O, exist_ok=True)

def skin(p):
    im = Image.open(p).convert("RGB").resize((192, 336)); a = im.load(); hit = tot = 0
    for y in range(20, 140):
        for x in range(50, 142):
            r, g, b = a[x, y]; tot += 1; hit += (r > 150 and r > g + 18 and g > b + 8 and r - b > 40)
    return hit / tot

def run(j, seed, out):
    cmd = [os.path.abspath('bin/sd-cli'), '--diffusion-model', f'{M}/flux-2-klein-4b-{Q}.gguf', '--vae', f'{M}/flux2-vae.safetensors',
           '--llm', f'{M}/Qwen3-4B-Q4_K_M.gguf', '-p', j['prompt'], '--cfg-scale', '1.0', '--sampling-method', 'euler', '--steps', '4',
           '-W', str(j['w']), '-H', str(j['h']), '--seed', str(seed), '-t', str(os.cpu_count()), '-o', out]
    if j.get('ref'): cmd[1:1] = ['-r', os.path.join(B, j['ref'])]
    t = time.time(); r = subprocess.run(cmd, capture_output=True, text=True)
    print(j['id'], seed, r.returncode, round(time.time() - t), 's', (r.stderr[-300:] if r.returncode else ''), flush=True)
    return r.returncode == 0 and os.path.exists(out)

for j in jobs:
    out = f"{O}/{j['id']}.png"
    if stage == 'refs':
        for k in range(4):
            if run(j, j['seed'] + 97 * k, out) and skin(out) < 0.03: break
            print('  skin', round(skin(out), 3) if os.path.exists(out) else 'none', flush=True)
    else:
        run(j, j['seed'], out)
print('done', len(os.listdir(O)))
