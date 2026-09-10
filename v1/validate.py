"""Cheap go/no-go check before committing to the full run.

Only question: is the error barrier at spawn epoch 0 clearly larger than at a
spawn epoch past the claimed chaotic transient? If not, there is no effect to
demo and the configuration needs to change, not the plot.
"""
import sys, json
import run

BASE = dict(device="auto", momentum=0.9, batch_size=128, kr_reg=1e-2, seed=0,
            schedule="step", n_test=10_000, ckpt_every=2.0, n_children=2,
            spawn_epochs=[0.0, 6.0], ntk_probe=64, kr_train=128, kr_test=128, kr_every=100)

CFGS = {
    "A": dict(preset="valA", n_train=10_000, width=24, epochs=60, lr=0.1),
    "D": dict(preset="valD", n_train=50_000, width=32, epochs=40, lr=0.1),
}

for tag in sys.argv[1:] or ["A", "D"]:
    cfg = dict(BASE); cfg.update(CFGS[tag])
    print(f"\n{'='*60}\nVALIDATING config {tag}: {cfg['n_train']} imgs, width {cfg['width']}, "
          f"{cfg['epochs']} ep\n{'='*60}")
    run.main(cfg)
    r = json.load(open(f"results/results_{cfg['preset']}.json"))
    b = {x["spawn_epoch"]: x["mean_test_barrier"] for x in r["barriers"]}
    e, l = list(b.values())[0], list(b.values())[-1]
    print(f">>> config {tag}: barrier@0={e:.4f}  barrier@late={l:.4f}  "
          f"ratio={e/max(l,1e-4):.1f}x  test_acc={r['final_test_acc']:.3f}")
    print(f">>> VERDICT {tag}: {'EFFECT PRESENT' if e > 0.03 and e > 3*l else 'NO CLEAR EFFECT'}")
