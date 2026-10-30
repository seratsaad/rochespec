"""LAMOST LRS fits on one node: the payne-zero grid is loaded once and shared by
forked workers (copy-on-write), each fitting whole stars with rochespec.lrs.fit_star.
Usage: CB_MODEL=pz CB_OUTER=L2 python tools/run_lrs_pool.py <nworkers> <tag> [subset: every k-th star, offset] -> cbscale/res_lam/results_<tag>.csv
Resumable: stars already in any results_*.csv are skipped. Per-star wall and CPU seconds are recorded (secs, cpu_secs).
"""
import sys, os, time, glob
import multiprocessing as mp
import pandas as pd
assert os.environ.get("CB_OUTER", "L2") == "L2"
from rochespec import lrs as F
if os.environ.get("CB_TSTEP"):            # cost test of the finer model Teff rows (rochespec.fine_joint); separate output directory
    from rochespec import fine_joint
    fine_joint.finer_teff_grid(F, step=int(os.environ["CB_TSTEP"])); F.OUT = F.OUT.rstrip("/") + f"_t{os.environ['CB_TSTEP']}/"

G = None


def work(s):
    c0 = time.process_time(); t0 = time.time()
    try:
        r = F.fit_star(s, OBS[OBS.sid == s.sid], G)
    except Exception as e:
        return s.sid, None, repr(e)[:200], time.time() - t0, time.process_time() - c0
    return s.sid, r, None, time.time() - t0, time.process_time() - c0


if __name__ == "__main__":
    nw, tag = int(sys.argv[1]), sys.argv[2]
    k, off = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (1, 0)
    os.makedirs(F.OUT + "x2", exist_ok=True)
    fn = F.OUT + f"results_{tag}.csv"
    prev = [pd.read_csv(f) for f in glob.glob(F.OUT + "results_*.csv")]
    done = set(pd.concat(prev).sid) if prev else set()
    rows = pd.read_csv(fn).to_dict("records") if os.path.exists(fn) else []
    st, OBS = F.targets(); st = st.iloc[off::k]
    if os.environ.get("SID_FILE"):          # restrict to a fixed list of Gaia ids (follow-up jobs get disjoint lists)
        st = st[st.sid.isin(set(pd.read_csv(os.environ["SID_FILE"]).sid))]
    todo = [s for _, s in st.iterrows() if s.sid not in done]
    t0 = time.time(); G = F.load("opt", dv=F.DV, rebin=True)
    print(f"grid loaded in {time.time() - t0:.0f} s; {len(todo)} stars to fit with {nw} workers", flush=True)
    t1 = time.time(); n = 0
    with mp.get_context("fork").Pool(nw) as pool:
        for sid, r, err, wall, cpu in pool.imap_unordered(work, todo, chunksize=1):
            n += 1
            if err: print(f"{sid} ERROR {err}", flush=True); continue
            r = dict(sid=sid, n_spec=0) if r is None else r
            r["cpu_secs"] = cpu; r["wall_secs"] = wall; rows.append(r)
            if n % 20 == 0 or n == len(todo):
                pd.DataFrame(rows).to_csv(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
                print(f"{n}/{len(todo)} done, {time.time() - t1:.0f} s elapsed", flush=True)
    pd.DataFrame(rows).to_csv(fn + ".tmp", index=False); os.replace(fn + ".tmp", fn)
    print(f"finished {n} in {time.time() - t1:.0f} s (+ load {t1 - t0:.0f} s)", flush=True)
