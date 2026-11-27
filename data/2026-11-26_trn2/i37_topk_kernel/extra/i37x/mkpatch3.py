"""i37 control: released path (kernel off) with the padded-vocabulary masked_fill removed. One changed line."""
import os, shutil, subprocess
home = os.path.expanduser("~")
src = home + "/w/sglang_rl/patches/sglang_neuron_int_topk"
dst = home + "/w/i37/patch_nomask"
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
p = dst + "/sampling.py"
old = '    logits = logits.float().masked_fill(ids.unsqueeze(0) >= vocabulary_size, float("-inf"))\n'
new = '    logits = logits.float()  # i37 control: padded-vocabulary mask removed (was masked_fill ids >= vocabulary_size)\n'
t = open(p).read()
i = t.index("def _tempered_scores(")
j = t.index("def _pool_topk(")
body = t[i:j]
assert body.count(old) == 1, body.count(old)
open(p, "w").write(t[:i] + body.replace(old, new) + t[j:])
d = subprocess.run(["diff", "-ru", src, dst], capture_output=True, text=True).stdout
open(home + "/w/i37/patch_nomask.diff", "w").write(d)
print(d)
