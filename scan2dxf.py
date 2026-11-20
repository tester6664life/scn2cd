#!/usr/bin/env python3
"""scan2dxf - scan of laser-cut plywood templates -> clean DXF.

Pipeline: two reference crosses (100 mm apart, top of the sheet) give the scale and the sheet skew;
the burnt lines are extracted, gaps bridged, every closed part is traced; edges are fitted as straight
lines (XLINE-style, corners = intersections), curves as arcs/splines, near-axis edges are made exactly
orthogonal, symmetric parts are averaged, short jogs get their own fitted line; the result is written
as an R12 DXF (layers CUT, HOLES, TABS, REF; units mm) plus an overlay PNG (scan + red contour).

usage:  python3 scan2dxf.py scan.png [-o OUTDIR] [--name NAME] [--ruler-mm 100] [--kerf 0.0]
                            [--t-lo 45 --t-hi 85] [--no-overlay] [--debug]
needs:  numpy scipy opencv-python scikit-image
"""
import sys, os, math, pickle, json, argparse, shutil, tempfile
import numpy as np, cv2
from collections import deque
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.filters import apply_hysteresis_threshold

# ======================= stage A: calibration, line mask, skeleton, gap bridging, faces =======================
def load(path):
    g = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if g is None: raise SystemExit('cannot read image: %s' % path)
    # scanner brightness: bring the white of the reference paper (top strip) to 255. Ordinary scans (white = 255) are untouched.
    W = float(np.percentile(g[:550], 90))
    if 120 < W < 250:
        print('paper white %.0f -> 255 (brightness normalised)' % W)
        g = np.clip(g.astype(np.float32) * (255.0 / W), 0, 255).astype(np.uint8)
    return g

# ---------------- crosses ----------------
def refine_cross(g, cx, cy, half=110):
    x0, y0 = int(cx) - half, int(cy) - half
    reg = 255.0 - g[y0:y0 + 2 * half, x0:x0 + 2 * half].astype(np.float32)
    reg[reg < 12] = 0
    cols, rows = reg.sum(0), reg.sum(1)
    vx, hy = int(np.argmax(cols)), int(np.argmax(rows))
    xs, ys = [], []
    for r in range(reg.shape[0]):
        if abs(r - hy) > 15:
            row = reg[r, max(vx - 8, 0):vx + 9]
            if row.sum() > 20: xs.append((row * np.arange(max(vx - 8, 0), vx + 9)).sum() / row.sum())
    for c in range(reg.shape[1]):
        if abs(c - vx) > 15:
            col = reg[max(hy - 8, 0):hy + 9, c]
            if col.sum() > 20: ys.append((col * np.arange(max(hy - 8, 0), hy + 9)).sum() / col.sum())
    return np.array([np.mean(xs) + x0, np.mean(ys) + y0])

def find_crosses(g, strip=600):
    dk_ = 255 - g[:strip].astype(np.int16); thr_ = int(min(20, max(8, 0.3 * np.percentile(dk_, 99.99))))
    s = dk_ > thr_
    s = cv2.morphologyEx(s.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(s, connectivity=8)
    c = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if 80 <= w <= 200 and 80 <= h <= 200 and a / (w * h) < 0.3 and 0.7 < w / h < 1.4:
            c.append(ce[i])
    c.sort(key=lambda p: p[0])
    if len(c) < 2: raise SystemExit('cross marks not found: %d candidates' % len(c))
    if len(c) > 2: c = [c[0], c[-1]]
    return [refine_cross(g, *p) for p in c]

# ---------------- sheet area ----------------
def sheet_box(g):
    b = cv2.blur(g[::8, ::8], (7, 7))
    wood = ((b > 190) & (b < 243)).astype(np.uint8)
    wood = cv2.morphologyEx(wood, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    n, lab, st, ce = cv2.connectedComponentsWithStats(wood)
    i = 1 + np.argmax(st[1:, 4]); x, y, w, h, a = st[i]
    return x * 8, y * 8, (x + w) * 8, (y + h) * 8

# ---------------- skeleton ----------------
K8 = np.ones((3, 3), np.uint8)
def endpoints(s):
    """line ends, tolerant to tiny 3-pixel triangles at the tip (win = pixels in 9x9 window)"""
    s8 = s.astype(np.uint8)
    win = cv2.boxFilter(s8, cv2.CV_32F, (9, 9), normalize=False, borderType=cv2.BORDER_CONSTANT)
    cand = s & (win <= 6.5)
    n, lab, st, ce = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    out = np.zeros_like(s, bool)
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i); k = np.argmin(win[ys, xs]); out[ys[k], xs[k]] = True
    return out

def tips(s):
    k = cv2.filter2D(s.astype(np.uint8), -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT)
    return np.argwhere(s & (k == 2))


FILL = []


def auto_thresholds(d, box):
    """Hysteresis thresholds from the scan itself: the burnt line is the darkest ~0.1% of the sheet (relative to the local
    background), wood texture sets the noise floor. Standard-contrast scans keep the long-validated defaults 45/85."""
    x0, y0, x1, y1 = box; dd = d[y0 + 40:y1 - 40, x0 + 40:x1 - 40]
    line = float(np.percentile(dd, 99.9)); noise = float(np.percentile(dd, 90))
    hi = float(np.clip(0.465 * line, 35, 140)); lo = max(0.50 * hi, 1.25 * noise); lo = min(lo, 0.75 * hi)
    if abs(hi - 85) <= 0.12 * 85 and abs(lo - 45) <= 0.15 * 45: hi, lo = 85, 45     # ordinary scan: use the validated defaults
    print('line thresholds (auto): lo %d hi %d  [line contrast %.0f, texture %.0f]' % (round(lo), round(hi), line, noise))
    return int(round(lo)), int(round(hi))

def line_mask(g, box, SC, T_LO=None, T_HI=None):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    bg = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)
    d = bg.astype(np.int16) - g.astype(np.int16)
    if os.environ.get('T_LO') or os.environ.get('T_HI') or T_LO is not None or T_HI is not None:
        T_LO = int(os.environ.get('T_LO', 45)) if T_LO is None else T_LO; T_HI = int(os.environ.get('T_HI', 85)) if T_HI is None else T_HI
        print('line thresholds (manual): lo %d hi %d' % (T_LO, T_HI))
    else:
        T_LO, T_HI = auto_thresholds(d, box)
    m = apply_hysteresis_threshold(d, T_LO, T_HI)
    x0, y0, x1, y1 = box; mg = 25
    z = np.zeros_like(m); z[y0 + mg:y1 - mg, x0 + mg:x1 - mg] = True
    m &= z
    # close tab-sized gaps (<= ~1.8 mm) along the two sheet axes only: a gap in an axis-parallel line is a laser tab, not a break
    L_ = int(1.8 * SC) | 1; mu = m.astype(np.uint8)
    mh = cv2.morphologyEx(mu, cv2.MORPH_CLOSE, np.ones((1, L_), np.uint8)); mv = cv2.morphologyEx(mu, cv2.MORPH_CLOSE, np.ones((L_, 1), np.uint8))
    # keep only thin fills (a gap in a line ~ line thickness); a fat fill is the space between two close parallel lines (narrow slot), which must stay open
    TH = int(1.1 * SC); HH, WW = mu.shape
    def run_len(r, c, dr, dc):
        n_ = 0
        while 0 <= r < HH and 0 <= c < WW and mu[r, c] and n_ < 400: n_ += 1; r += dr; c += dc
        return n_
    def thin_fill(mc, horiz):
        fl = (mc > 0) & (mu == 0); nn_, ll_, ss_, _ = cv2.connectedComponentsWithStats(fl.astype(np.uint8), connectivity=8)
        ok = np.zeros(nn_, bool)
        for i in range(1, nn_):
            x_, y_, w_, h_, a_ = ss_[i]
            if horiz:
                if h_ > TH: continue
                r = y_ + h_ // 2; rl = run_len(r, x_ - 2, -1, 0) + run_len(r, x_ - 2, 1, 0); rr = run_len(r, x_ + w_ + 1, -1, 0) + run_len(r, x_ + w_ + 1, 1, 0)
                ok[i] = max(rl, rr) <= h_ + 10 + 0.4 * SC
                # a laser tab is a gap IN a line: the rows just above/below the strip are empty. Dark pixels beside the strip = a pocket of a tooth/notch -> do not fill it
                if ok[i]:
                    xa, xb = x_ + int(0.2 * w_), x_ + max(int(0.8 * w_), int(0.2 * w_) + 1)
                    fa = (mu[max(y_ - 4, 0):max(y_ - 1, 1), xa:xb] > 0).mean() if y_ > 1 else 0; fb = (mu[min(y_ + h_ + 1, HH - 1):min(y_ + h_ + 4, HH), xa:xb] > 0).mean()
                    if max(fa, fb) > 0.30: ok[i] = False
            else:
                if w_ > TH: continue
                c = x_ + w_ // 2; rl = run_len(y_ - 2, c, 0, -1) + run_len(y_ - 2, c, 0, 1); rr = run_len(y_ + h_ + 1, c, 0, -1) + run_len(y_ + h_ + 1, c, 0, 1)
                ok[i] = max(rl, rr) <= w_ + 10 + 0.4 * SC
                if ok[i]:
                    ya, yb = y_ + int(0.2 * h_), y_ + max(int(0.8 * h_), int(0.2 * h_) + 1)
                    fa = (mu[ya:yb, max(x_ - 4, 0):max(x_ - 1, 1)] > 0).mean() if x_ > 1 else 0; fb = (mu[ya:yb, min(x_ + w_ + 1, WW - 1):min(x_ + w_ + 4, WW)] > 0).mean()
                    if max(fa, fb) > 0.30: ok[i] = False
        return ok[ll_]
    m = (mu > 0) if __import__('os').environ.get('NOCLOSE') else ((mu > 0) | thin_fill(mh, True) | thin_fill(mv, False))
    m &= z
    # remember what the closing filled: those are the laser tabs (uncut bridges), they go to the TABS layer
    nF, labF, stF, _ = cv2.connectedComponentsWithStats((m & ~(mu > 0)).astype(np.uint8), connectivity=8)
    FILL.clear()
    for i in range(1, nF):
        x_, y_, w_, h_, a_ = stF[i]
        if max(w_, h_) / SC < 0.3: continue
        if w_ >= h_: FILL.append(((int(x_), int(y_ + h_ // 2)), (int(x_ + w_), int(y_ + h_ // 2)), w_ / SC))
        else: FILL.append(((int(x_ + w_ // 2), int(y_)), (int(x_ + w_ // 2), int(y_ + h_)), h_ / SC))
    m8 = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    inv = (m8 == 0).astype(np.uint8)
    n, lab, st, ce = cv2.connectedComponentsWithStats(inv, connectivity=4)
    small = (st[:, 4] < 3000); small[0] = False
    m8[small[lab]] = 1   # fill tiny holes (skeleton loops at line ends)
    return m8 > 0

def clean_components(sk, SC, minlen_mm=120, minbox_mm=20):
    n, lab, st, ce = cv2.connectedComponentsWithStats(sk.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool); info = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        L = a / SC; box = max(w, h) / SC
        if L >= minlen_mm or box >= minbox_mm * 2: keep[i] = True
        info.append((i, L, box, keep[i]))
    return keep[lab], info

def run(path, out, ruler_mm=100.0, dpi=None):
    g = load(path); H, W = g.shape
    a, b = find_crosses(g)
    d = float(np.hypot(*(b - a))); SC = d / ruler_mm
    ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0]))
    print('crosses', a.round(2), b.round(2), 'dist %.2f px -> %.4f px/mm (%.1f dpi) ruler angle %.3f' % (d, SC, SC * 25.4, ang))
    box = sheet_box(g); print('sheet box px', box)
    m = line_mask(g, box, SC)
    sk = skeletonize(m)
    sk_all = sk.copy()
    sk, info = clean_components(sk, SC)
    # recover short but strong line pieces lying next to the kept skeleton (e.g. a slot edge isolated by gaps)
    kk0 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    d0 = cv2.GaussianBlur((cv2.morphologyEx(g, cv2.MORPH_CLOSE, kk0).astype(np.int16) - g.astype(np.int16)).astype(np.float32), (0, 0), 1.0)
    dist_k = cv2.distanceTransform((~sk).astype(np.uint8), cv2.DIST_L2, 3)
    nC, labC, stC, _ = cv2.connectedComponentsWithStats((sk_all & ~sk).astype(np.uint8), connectivity=8)
    rec = 0
    for i in range(1, nC):
        if stC[i, 4] < 2.5 * SC: continue
        ys_, xs_ = np.nonzero(labC == i) if stC[i, 4] < 20000 else (None, None)
        if ys_ is None: continue
        md0 = float(d0[ys_, xs_].mean())
        near = dist_k[ys_, xs_].min() <= 3 * SC and max(stC[i, 2], stC[i, 3]) >= 3 * SC
        small_part = stC[i, 4] >= 60 * SC and md0 >= 65 and stC[i, 2] >= 8 * SC and stC[i, 3] >= 8 * SC   # a small standalone part outline
        if md0 >= 65 and (near or small_part):
            sk[ys_, xs_] = True; rec += 1
    print('recovered strong short pieces:', rec)
    print('skeleton px', int(sk.sum()), 'components kept', sum(1 for i in info if i[3]), 'dropped', sum(1 for i in info if not i[3]))
    ys = np.nonzero(sk)[0]; Y0 = max(int(ys.min()) - 20, 0); Y1 = min(int(ys.max()) + 20, H)
    sk = sk[Y0:Y1].copy(); Hh, Ww = sk.shape
    # --- spurs
    def nbrs(s, x, y):
        return [(x + dx, y + dy) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dx or dy) and 0 <= x + dx < Ww and 0 <= y + dy < Hh and s[y + dy, x + dx]]
    sk2 = sk.copy(); removed = 0
    for (y, x) in np.argwhere(endpoints(sk2)):
        path = [(x, y)]; seen = {(x, y)}; cur = (x, y); hit = False
        while len(path) < 30:
            nb = [q for q in nbrs(sk2, *cur) if q not in seen]
            if len(nbrs(sk2, *cur)) >= 3 and len(path) > 1: hit = True; break
            if not nb: break
            cur = nb[0]; seen.add(cur); path.append(cur)
        if hit:
            for (px, py) in path[:-1]: sk2[py, px] = False
            removed += 1
    print('spurs removed', removed)
    sk_orig = sk2.copy()
    # --- bridge gaps
    bridged = sk2; gaps = []; unresolved = []
    MAXG = int(7 * SC)  # up to 7 mm
    def bfs_own(e, limit=90):
        seen = {e: 0}; dq = deque([e])
        while dq:
            cur = dq.popleft(); dd = seen[cur]
            if dd >= limit: continue
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (cur[0] + dx, cur[1] + dy)
                    if (dx or dy) and q not in seen and 0 <= q[0] < Ww and 0 <= q[1] < Hh and sk2[q[1], q[0]]:
                        seen[q] = dd + 1; dq.append(q)
        return seen
    for rnd in range(4):
        deleted = False; unresolved = []
        ep = endpoints(sk2)
        # laser tab ends often curl slightly, which fools the window test: add free tips of the thinned skeleton
        thin = skeletonize(sk2)
        for (ty, tx) in tips(thin):
            if ep[max(ty - 6, 0):ty + 7, max(tx - 6, 0):tx + 7].any(): continue
            ow = bfs_own((int(tx), int(ty)), limit=40)
            if len(ow) >= 30 and not any(dd <= 25 and len(nbrs(thin, *q)) >= 3 for q, dd in ow.items()): ep[ty, tx] = True
        ep = np.argwhere(ep); print('open ends', len(ep))
        ys2, xs2 = np.nonzero(sk2); P = np.c_[xs2, ys2]; tree = cKDTree(P)

        for (y, x) in ep:
            own = bfs_own((x, y))
            if len(own) < 20: unresolved.append((x, y, 'stub')); continue
            jd0 = [dd for q, dd in own.items() if dd <= 14 and len(nbrs(sk2, *q)) >= 3]
            if jd0:   # tiny side spur sitting on a line: delete it, it is not a real line end
                dj0 = min(jd0)
                for q, dd in own.items():
                    if dd < dj0: bridged[q[1], q[0]] = False
                deleted = True; continue
            back = np.array([q for q, dd in own.items() if 10 <= dd <= 60], float)
            if len(back) < 5: unresolved.append((x, y, 'short')); continue
            dv = np.array([x, y], float) - back.mean(0); dv /= np.hypot(*dv)
            idx = tree.query_ball_point([x, y], MAXG); best = None
            for i in idx:
                q = (int(P[i][0]), int(P[i][1]))
                if q in own and own[q] < 90: continue
                w = np.array(q, float) - [x, y]; t = w @ dv; lat = abs(w[0] * dv[1] - w[1] * dv[0])
                if t <= 0 or t > MAXG or lat > 4 + 0.23 * t: continue
                if t > 2.5 * SC:   # long gap: the far side must continue the same line
                    nb_ = P[tree.query_ball_point([q[0], q[1]], 14)].astype(float)
                    if len(nb_) < 6: continue
                    uu, ss, vv = np.linalg.svd(nb_ - nb_.mean(0), full_matrices=False)
                    if abs(vv[0] @ dv) < 0.9 or lat > 3 + 0.06 * t: continue
                sc = t + 2.5 * lat
                if best is None or sc < best[0]: best = (sc, q, t, lat)
            if best is None:
                jd = [dd for q, dd in own.items() if dd <= 70 and len(nbrs(sk2, *q)) >= 3]
                if jd:
                    dj = min(jd)
                    br = [q for q, dd in own.items() if dd < dj]
                    axok = False
                    if len(br) >= 35:
                        pb = np.array(br, float); vb = np.linalg.svd(pb - pb.mean(0), full_matrices=False)[2][0]
                        angb = abs(math.degrees(math.atan2(vb[1], vb[0]))) % 90; axok = min(angb, 90 - angb) < 8   # cracks are oblique, cut edges are axis-parallel
                    if axok and float(np.mean([d0[q[1] + Y0, q[0]] for q in br])) >= 60:
                        unresolved.append((x, y, 'strong')); continue   # a real strong line end, not a spur
                    for q, dd in own.items():
                        if dd < dj: bridged[q[1], q[0]] = False; sk2[q[1], q[0]] = False
                    deleted = True; continue
                for (y2, x2) in ep:
                    if (x2, y2) == (x, y) or (x2, y2) in own: continue
                    dd = math.hypot(x2 - x, y2 - y)
                    if dd <= 2.5 * SC and (best is None or dd < best[0]): best = (dd, (x2, y2), dd, 0)
            if best is None:
                for i in idx:
                    q = (int(P[i][0]), int(P[i][1]))
                    if q in own: continue
                    dd = math.hypot(q[0] - x, q[1] - y)
                    if dd <= 2.5 * SC and (best is None or dd < best[0]): best = (dd, q, dd, 0)
            if best is None: unresolved.append((x, y, 'none')); continue
            cv2.line(bridged.view(np.uint8), (x, y), best[1], 1, 1)
            gaps.append(dict(p0=(x, y + Y0), p1=(best[1][0], best[1][1] + Y0), mm=best[2] / SC))

        if not deleted: break
    # long faint gaps: route a min-cost path along the darkness map between two aligned open ends
    from skimage.graph import route_through_array
    kk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    dmap = (cv2.morphologyEx(g[Y0:Y1], cv2.MORPH_CLOSE, kk).astype(np.int16) - g[Y0:Y1].astype(np.int16))
    dmap = cv2.GaussianBlur(dmap.astype(np.float32), (0, 0), 2.0)
    # weak long bridges (>3 mm over almost no dark line) are separators, not edges: erase them, trim hooks
    keep_g = []
    for gp in gaps:
        (ax_, ay_), (bx_, by_) = gp['p0'], gp['p1']; ay_ -= Y0; by_ -= Y0
        if gp['mm'] <= 3.0: keep_g.append(gp); continue
        nn = max(int(math.hypot(bx_ - ax_, by_ - ay_)), 2)
        xs_ = np.linspace(ax_, bx_, nn).astype(int); ys_ = np.linspace(ay_, by_, nn).astype(int)
        md_ = np.mean([dmap[max(yy - 4, 0):yy + 5, max(xx - 4, 0):xx + 5].max() for xx, yy in zip(xs_, ys_)])
        if md_ < 31:   # dmap is blurred, so thresholds are lower than raw darkness
            mk = np.zeros(bridged.shape, np.uint8); cv2.line(mk, (int(ax_), int(ay_)), (int(bx_), int(by_)), 1, 3)
            bridged &= ~((mk > 0) & ~sk_orig)
            print('weak bridge erased', gp['p0'], gp['p1'], 'mm %.1f mean d %.1f' % (gp['mm'], md_))
        else: keep_g.append(gp)
    gaps[:] = keep_g
    # trim short hooks: a tail that turns >60 deg within 3 mm of its end is cut back to the turn
    def trim_hooks():
        for (y, x) in np.argwhere(tips(bridged)):
            own = bfs_own((x, y), limit=100)
            pts = sorted(own.items(), key=lambda kv: kv[1])
            if len(pts) < 40: continue
            path = [q for q, dd in pts if True]
            # walk back along the single path
            seq = [(x, y)]; seen = {(x, y)}; cur = (x, y)
            while len(seq) < 90:
                nbq = [q for q in nbrs(bridged, *cur) if q not in seen]
                if len(nbq) != 1: break
                cur = nbq[0]; seen.add(cur); seq.append(cur)
            if len(seq) < 60: continue
            tail = np.array(seq[:8], float); far = np.array(seq[30:60], float)
            v1 = tail[0] - tail[-1]; v2 = far[0] - far[-1]
            if np.hypot(*v1) < 3 or np.hypot(*v2) < 3: continue
            cosang = (v1 @ v2) / (np.hypot(*v1) * np.hypot(*v2))
            if cosang > 0.5: continue   # not a hook
            # cut at the sharpest turn within first 70 px
            best_i, best_c = None, 1.0
            for i in range(8, 70):
                a1 = np.array(seq[i]) - np.array(seq[i - 8]); a2 = np.array(seq[min(i + 8, len(seq) - 1)]) - np.array(seq[i])
                if np.hypot(*a1) < 3 or np.hypot(*a2) < 3: continue
                c = (a1 @ a2) / (np.hypot(*a1) * np.hypot(*a2))
                if c < best_c: best_c, best_i = c, i
            if best_i is not None and best_c < 0.5:
                for (px_, py_) in seq[:best_i]: bridged[py_, px_] = False
                print('hook trimmed at', (x, y + Y0), 'px', best_i)
    trim_hooks()
    # Manhattan L-closure: two nearby open ends are joined by an axis-parallel L that follows more dark line
    def l_close():
        bsk = skeletonize(bridged)
        def bfs_t(e, limit):
            seen = {e: 0}; dq = deque([e])
            while dq:
                cur = dq.popleft(); dd = seen[cur]
                if dd >= limit: continue
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        q = (cur[0] + dx, cur[1] + dy)
                        if (dx or dy) and q not in seen and 0 <= q[0] < Ww and 0 <= q[1] < Hh and bsk[q[1], q[0]]:
                            seen[q] = dd + 1; dq.append(q)
            return seen
        def free_tip(e):
            ow = bfs_t(e, 70)
            if len(ow) < 60: return False
            return not any(dd <= 60 and len(nbrs(bsk, *q)) >= 3 for q, dd in ow.items())
        eps2 = [(int(x), int(y)) for (y, x) in tips(bsk)]
        eps2 = [e for e in eps2 if free_tip(e)]
        done = set()
        for e1 in eps2:
            if e1 in done: continue
            own1 = bfs_t(e1, 150); best = None
            if len(own1) < 60: continue
            for e2 in eps2:
                if e2 == e1 or e2 in done or e2 in own1: continue
                dx, dy = e2[0] - e1[0], e2[1] - e1[1]
                if max(abs(dx), abs(dy)) > 8 * SC or min(abs(dx), abs(dy)) < 3: continue
                dd = math.hypot(dx, dy)
                if best is None or dd < best[0]: best = (dd, e2)
            if best is None: continue
            e2 = best[1]
            optA = [e1, (e1[0], e2[1]), e2]; optB = [e1, (e2[0], e1[1]), e2]
            def score(o):
                m_ = np.zeros(bridged.shape, np.uint8); cv2.polylines(m_, [np.array(o, np.int32)], False, 1, 1)
                ys3, xs3 = np.nonzero(m_); return float(np.mean([dmap[y3, x3] for y3, x3 in zip(ys3, xs3)]))
            o = optA if score(optA) > score(optB) else optB
            cv2.polylines(bridged.view(np.uint8), [np.array(o, np.int32)], False, 1, 1)
            gaps.append(dict(p0=(e1[0], e1[1] + Y0), p1=(e2[0], e2[1] + Y0), mm=best[0] / SC)); done.add(e1); done.add(e2)
            print('L-closure', (e1[0], e1[1] + Y0), '->', (e2[0], e2[1] + Y0))
    l_close()
    # last resort for ends nobody could resolve: join to the nearest foreign skeleton pixel within 7 mm
    ys2, xs2 = np.nonzero(bridged); P = np.c_[xs2, ys2]; tree = cKDTree(P); still = []
    open_now = set((int(xx), int(yy)) for yy, xx in np.argwhere(endpoints(bridged)))
    for (x, y, why) in unresolved:
        if (x, y) not in open_now: continue
        own = bfs_own((x, y), limit=120); best = None
        for i in tree.query_ball_point([x, y], MAXG):
            q = (int(P[i][0]), int(P[i][1]))
            if q in own: continue
            dd = math.hypot(q[0] - x, q[1] - y)
            if best is None or dd < best[0]: best = (dd, q)
        if best is None: still.append((x, y, why)); continue
        cv2.line(bridged.view(np.uint8), (x, y), best[1], 1, 1)
        gaps.append(dict(p0=(x, y + Y0), p1=(best[1][0], best[1][1] + Y0), mm=best[0] / SC))
    unresolved = still
    eps_ = [(int(x), int(y)) for (y, x) in np.argwhere(endpoints(bridged))]
    info_ = {}
    for (x, y) in eps_:
        own = bfs_own((x, y), limit=150)
        back = np.array([q for q, dd in own.items() if 10 <= dd <= 70], float)
        if len(back) < 5: continue
        dv = np.array([x, y], float) - back.mean(0); dv /= np.hypot(*dv)
        info_[(x, y)] = (own, dv)
    used = set()
    for (x, y), (own, dv) in info_.items():
        if (x, y) in used: continue
        best = None
        for (x2, y2), (own2, dv2) in info_.items():
            if (x2, y2) == (x, y) or (x2, y2) in used or (x2, y2) in own: continue
            w = np.array([x2 - x, y2 - y], float); t = w @ dv; lat = abs(w[0] * dv[1] - w[1] * dv[0])
            if t < 2.5 * SC or t > 45 * SC or lat > 3 * SC + 0.15 * t: continue
            if (-dv2) @ dv < 0.85: continue   # the two ends must point at each other
            sc = t + 3 * lat
            if best is None or sc < best[0]: best = (sc, (x2, y2), t)
        if best is None: continue
        x2, y2 = best[1]
        pad = 60; xa, xb = max(min(x, x2) - pad, 0), min(max(x, x2) + pad, Ww); ya, yb = max(min(y, y2) - pad, 0), min(max(y, y2) + pad, Hh)
        sub = dmap[ya:yb, xa:xb]
        cost = 1.0 / (1.0 + np.clip(sub, 0, 80) / 8.0) ** 2
        path, _ = route_through_array(cost, (y - ya, x - xa), (y2 - ya, x2 - xa), fully_connected=True, geometric=True)
        pl = np.array(path); L = len(pl); md = float(sub[pl[:, 0], pl[:, 1]].mean())
        print('route gap', (x, y + Y0), '->', (x2, y2 + Y0), 'len %.1f mm straight %.1f mm mean d %.1f' % (L / SC, best[2] / SC, md))
        if L <= 1.3 * best[2] and md >= 14:
            for (py, px) in path: bridged[py + ya, px + xa] = True
            gaps.append(dict(p0=(x, y + Y0), p1=(x2, y2 + Y0), mm=best[2] / SC)); used.add((x, y)); used.add((x2, y2))
    # final sweep: ends that are still open after bridging (e.g. created by earlier bridges)
    for (y, x) in np.argwhere(endpoints(bridged)):
        own = bfs_own((x, y), limit=120)
        if len(own) < 20: continue
        back = np.array([q for q, dd in own.items() if 10 <= dd <= 60], float)
        if len(back) < 5: continue
        dv = np.array([x, y], float) - back.mean(0); dv /= np.hypot(*dv)
        ys2, xs2 = np.nonzero(bridged); P = np.c_[xs2, ys2]; tree = cKDTree(P); best = None
        for i in tree.query_ball_point([x, y], MAXG):
            q = (int(P[i][0]), int(P[i][1]))
            if q in own: continue
            w = np.array(q, float) - [x, y]; t = w @ dv; lat = abs(w[0] * dv[1] - w[1] * dv[0])
            if t <= 0 or lat > 5 + 0.35 * t: continue   # the bridge must continue the line, not turn sideways
            dd = math.hypot(q[0] - x, q[1] - y)
            if best is None or dd < best[0]: best = (dd, q)
        if best is None: continue
        cv2.line(bridged.view(np.uint8), (x, y), best[1], 1, 1)
        gaps.append(dict(p0=(x, y + Y0), p1=(best[1][0], best[1][1] + Y0), mm=best[0] / SC))
    print('bridged', len(gaps), 'gaps; mm:', np.round(sorted(g_['mm'] for g_ in gaps), 2).tolist())
    print('unresolved ends', len(unresolved), [(u[0], u[1] + Y0, u[2]) for u in unresolved[:20]])
    # --- separators: a weak (faint) bridge between two closed regions is not a cut line -> merge them
    inv0 = (~bridged).astype(np.uint8)
    n0, lab0 = cv2.connectedComponents(inv0, connectivity=4); out0 = lab0[3, 3]
    merged_any = False
    for gp in list(gaps):
        if gp['mm'] < 2.5: continue
        (ax_, ay_), (bx_, by_) = gp['p0'], gp['p1']; ay_ -= Y0; by_ -= Y0
        nn = max(int(math.hypot(bx_ - ax_, by_ - ay_)), 2)
        xs_ = np.linspace(ax_, bx_, nn).astype(int); ys_ = np.linspace(ay_, by_, nn).astype(int)
        md_ = np.mean([dmap[max(yy - 4, 0):yy + 5, max(xx - 4, 0):xx + 5].max() for xx, yy in zip(xs_, ys_)])
        if md_ >= 35: continue
        ux, uy = (bx_ - ax_), (by_ - ay_); L_ = math.hypot(ux, uy) or 1.0; nx, ny = -uy / L_, ux / L_
        mx, my = (ax_ + bx_) // 2, (ay_ + by_) // 2
        la = lab0[int(np.clip(my + 12 * ny, 0, Hh - 1)), int(np.clip(mx + 12 * nx, 0, Ww - 1))]
        lb = lab0[int(np.clip(my - 12 * ny, 0, Hh - 1)), int(np.clip(mx - 12 * nx, 0, Ww - 1))]
        if la != lb and la != out0 and lb != out0 and la > 0 and lb > 0:
            mk = np.zeros(bridged.shape, np.uint8); cv2.line(mk, (int(ax_), int(ay_)), (int(bx_), int(by_)), 1, 3)
            bridged &= ~((mk > 0) & ~sk_orig)
            print('separator between two regions removed', gp['p0'], gp['p1'], 'mean d %.1f' % md_); merged_any = True
    # --- faces
    inv = (~bridged).astype(np.uint8)
    n, lab = cv2.connectedComponents(inv, connectivity=4)
    areas = np.bincount(lab.ravel(), minlength=n); outside = lab[3, 3]
    faces = []
    for i in range(1, n):
        if i == outside or areas[i] / SC ** 2 < 20: continue
        yy, xx = np.nonzero(lab == i)
        if xx.min() <= 2 or yy.min() <= 2 or xx.max() >= Ww - 3 or yy.max() >= Hh - 3: continue
        faces.append(dict(id=i, area=float(areas[i] / SC ** 2), bbox=(int(xx.min()), int(yy.min() + Y0), int(xx.max()), int(yy.max() + Y0))))
    print('faces', len(faces))
    for f in faces: print('  face %d area %.0f mm2 bbox %s' % (f['id'], f['area'], f['bbox']))
    gaps = list(gaps) + [dict(p0=p0_, p1=p1_, mm=mm_) for (p0_, p1_, mm_) in FILL]
    pickle.dump(dict(SC=SC, a=a, b=b, ang=ang, Y0=Y0, bridged=bridged, sk_orig=sk_orig, lab=lab, faces=faces, gaps=gaps, shape=g.shape), open(out + '/A.pkl', 'wb'))
    # overlay
    f3 = 5; sm = cv2.cvtColor(cv2.resize(g[Y0:Y1], (Ww // f3, Hh // f3), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    rng = np.random.RandomState(1)
    for f in faces:
        mm = cv2.resize((lab == f['id']).astype(np.uint8), (Ww // f3, Hh // f3), interpolation=cv2.INTER_NEAREST) > 0
        sm[mm] = (0.6 * sm[mm] + 0.4 * rng.randint(60, 255, 3)).astype(np.uint8)
    bs = cv2.resize((bridged & ~sk_orig).astype(np.uint8) * 255, (Ww // f3, Hh // f3), interpolation=cv2.INTER_AREA) > 0
    sm[cv2.dilate(bs.astype(np.uint8), np.ones((3, 3))) > 0] = (0, 0, 255)
    for (x, y, why) in unresolved: cv2.circle(sm, (x // f3, y // f3), 8, (255, 0, 255), 2)
    cv2.imwrite(out + '/A_overlay.png', sm)



# ======================= stage B: edge fitting, corners, symmetry, tabs, holes, DXF =======================
HOLE_Q = float(os.environ.get('HOLE_Q', '0.2')); BAND_F = float(os.environ.get('BAND_F', '0.7'))
EPS = 0.30; CLTOL = 0.25; STEP = 0.05
rep = []
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); rep.append(s)

def wmed(a, w):
    o = np.argsort(a); c = np.cumsum(np.array(w)[o]); return np.array(a)[o][np.searchsorted(c, c[-1] / 2)]
def tls(P):
    c = P.mean(0); u, s, vt = np.linalg.svd(P - c, full_matrices=False); return c, vt[0]
def turn(d1, d2):
    return math.degrees(math.atan2(d1[0] * d2[1] - d1[1] * d2[0], d1 @ d2))
def pct(a, q): return float(np.percentile(a, q))

def main(img, out, prefix='template', KERF=0.0, SX=1.0, SY=1.0):
    A = pickle.load(open(out + '/A.pkl', 'rb'))
    SC = A['SC']; X0 = np.array(A['a']); lab = A['lab']; Y0 = A['Y0']; faces0 = A['faces']
    g = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # ---------- pixel contours ----------
    pxc = {}
    for f in faces0:
        m = (lab == f['id']).astype(np.uint8); cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        c = max(cs, key=len)[:, 0, :].astype(float); c[:, 1] += Y0; pxc[f['id']] = c
    # ---------- shear-corrected frame ----------
    Hh = []; Vv = []; a0 = A['ang']
    for fid, c in pxc.items():
        dp = cv2.approxPolyDP(c.astype(np.float32).reshape(-1, 1, 2), 4.0, True)[:, 0, :]
        for p, q in zip(dp, np.roll(dp, -1, axis=0)):
            v = q - p; L = np.hypot(*v)
            if L < 15 * SC: continue
            ang = math.degrees(math.atan2(v[1], v[0])) % 180
            if min(ang, 180 - ang) < 2.5: Hh.append((ang if ang < 90 else ang - 180, L))
            elif abs(ang - 90) < 2.5: Vv.append((ang - 90, L))
    ah = wmed([x[0] for x in Hh], [x[1] for x in Hh]) if Hh else a0
    av = wmed([x[0] for x in Vv], [x[1] for x in Vv]) if Vv else a0
    say('axis angles: H %.3f deg (%d segs), V %.3f deg (%d segs) -> shear %.3f deg' % (ah, len(Hh), av, len(Vv), ah - av))
    rh, rv = math.radians(ah), math.radians(av)
    M = np.array([[math.cos(rh), -math.sin(rv)], [math.sin(rh), math.cos(rv)]]); Minv = np.linalg.inv(M)
    def to_mm(P):
        q = (np.atleast_2d(P) - X0) @ Minv.T / SC; return np.c_[q[:, 0] * SX, -q[:, 1] * SY]
    def to_px(Mm):
        Mm = np.atleast_2d(Mm); return (np.c_[Mm[:, 0] / SX, -Mm[:, 1] / SY] * SC) @ M.T + X0
    OFF = 0.5 / SC - KERF   # outward shift of fitted lines (skeleton bias, minus optional inward kerf offset)
    def fit_edge(pts, sgn, d0, L, force=None, fdev=0.45):
        if L < 0.8: return None
        tr = max(1, min(int(0.25 / 0.045), len(pts) // 4)); P = pts[tr:len(pts) - tr] if len(pts) > 2 * tr + 3 else pts
        c, d = tls(P); nrm = np.array([-d[1], d[0]]); dev = pct(np.abs((P - c) @ nrm), 95)
        if d @ d0 < 0: d = -d
        th = math.degrees(math.atan2(d[1], d[0])); ax = None; AXTOL = 3.5 if L >= 3 else 7.0
        if abs(((th + 90) % 180) - 90) < AXTOL: ax = 'H'
        elif abs(((th) % 180) - 90) < AXTOL: ax = 'V'
        if ax and os.environ.get('DBGTILT'):
            tl = ((th + 90) % 180) - 90 if ax == 'H' else (th % 180) - 90
            print('TILT L=%.1f tilt=%.2f dev=%.2f ax=%s c=(%.1f,%.1f)' % (L, tl, dev, ax, c[0], c[1]))
        if ax:   # snapping is only legitimate if the points really sit on the axis-parallel line (not merely on a slightly tilted one)
            devax = pct(np.abs(P[:, 1] - np.median(P[:, 1])), 95) if ax == 'H' else pct(np.abs(P[:, 0] - np.median(P[:, 0])), 95)
            if devax > 0.30: ax = None
        if force:
            ax = force; devax = pct(np.abs(P[:, 1] - np.median(P[:, 1])), 95) if ax == 'H' else pct(np.abs(P[:, 0] - np.median(P[:, 0])), 95)
            if devax > fdev: return None
            dev = max(dev, 0.0) if dev < 0.45 else 0.44
        if not ((ax and dev < 0.45) or dev < 0.17): return None
        r = dict(fit=True, c=c, d=d, dev=dev, axis=ax); nout = np.array([d[1], -d[0]]) * sgn
        if ax == 'H': r['coord'] = float(np.median(P[:, 1])) + OFF * np.sign(nout[1]); r['d'] = np.array([np.sign(d[0]), 0.])
        elif ax == 'V': r['coord'] = float(np.median(P[:, 0])) + OFF * np.sign(nout[0]); r['d'] = np.array([0., np.sign(d[1])])
        else: r['c'] = c + OFF * nout
        return r
    # ---------- per-face edges ----------
    faces = []; KEPT_T = set()
    for f in [q for q in faces0 if q['area'] > 100]:
        C = to_mm(pxc[f['id']]); N = len(C)
        dp = cv2.approxPolyDP(C.astype(np.float32).reshape(-1, 1, 2), EPS, True)[:, 0, :]
        idx = np.sort(np.array([np.argmin(np.hypot(*(C - q).T)) for q in dp])); V = C[idx]; n = len(V)
        area = 0.5 * np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1]); sgn = 1 if area > 0 else -1
        edges = []
        for e in range(n):
            a = idx[e]; b = idx[(e + 1) % n]; pts = C[a:b + 1] if b > a else np.r_[C[a:], C[:b + 1]]
            chord = V[(e + 1) % n] - V[e]; L = float(np.hypot(*chord)); d0 = chord / max(L, 1e-9)
            ed = dict(v0=e, v1=(e + 1) % n, pts=pts, len=L, t='S', fit=False, axis=None)
            r_ = fit_edge(pts, sgn, d0, L)
            if r_: ed.update(r_)
            else: ed['t'] = 'C'
            edges.append(ed)
        for e in range(n):   # stair step: a short edge between two parallel axis edges must be axis-perpendicular too (right angles), even if the scan makes it slightly slanted
            ed = edges[e]; pe, ne = edges[e - 1], edges[(e + 1) % n]
            if ed['t'] == 'S' and not ed.get('axis') and ed['len'] <= 12 and pe['t'] == 'S' and ne['t'] == 'S' and pe.get('axis') and pe.get('axis') == ne.get('axis'):
                want = 'H' if pe['axis'] == 'V' else 'V'; vv = ed['pts'][-1] - ed['pts'][0]; th = math.degrees(math.atan2(vv[1], vv[0]))
                dv_ = abs(((th + 90) % 180) - 90) if want == 'H' else abs((th % 180) - 90)
                if dv_ < 14:
                    r2 = fit_edge(ed['pts'], sgn, ed['pts'][-1] - ed['pts'][0], ed['len'], force=want)
                    if r2: ed.update(r2); say('face %d: stair step %.1f mm snapped to a right angle (was %.0f deg off-axis)' % (f['id'], ed['len'], dv_))
        items = []; st = next((k for k in range(n) if edges[k]['t'] != 'C'), 0); order = [edges[(st + k) % n] for k in range(n)]; cur = None
        for ed in order:
            if ed['t'] == 'C':
                if cur is not None and cur['t'] == 'C': cur['pts'] = np.r_[cur['pts'], ed['pts'][1:]]; cur['v1'] = ed['v1']
                else: cur = dict(ed); items.append(cur)
            else: items.append(ed); cur = ed
        nunion = 0; ch = True
        while ch and len(items) > 3:
            ch = False
            for k in range(len(items)):
                a_, b_ = items[k], items[(k + 1) % len(items)]
                if a_ is b_ or (a_.get('axis') and a_.get('axis') == b_.get('axis')): continue
                U = np.r_[a_['pts'], b_['pts'][1:]]; Lu = float(np.hypot(*(U[-1] - U[0])))
                if Lu < 4 or Lu > 80: continue
                r_ = fit_edge(U, sgn, (U[-1] - U[0]) / Lu, Lu)
                if r_ and r_['axis'] and r_['dev'] < 0.25 and min(a_['len'], b_['len']) >= 2.0 and np.max(np.abs((U - r_['c'] if False else U - U.mean(0)) @ np.array([-r_['d'][1], r_['d'][0]]) - np.median((U - U.mean(0)) @ np.array([-r_['d'][1], r_['d'][0]])))) < 0.5:
                    a_['pts'] = U; a_['v1'] = b_['v1']; a_['len'] = Lu; a_['t'] = 'S'; a_.update(r_)
                    j_ = (k + 1) % len(items); items.pop(j_); nunion += 1; ch = True; break
        if nunion: say('face %d: %d broken pieces merged into straight axis-parallel edges' % (f['id'], nunion))
        def cand(i): return i['t'] == 'S' and i.get('axis') is None and 1.0 <= i['len'] <= 60
        def chordof(i):
            v = V[i['v1']] - V[i['v0']]; return v / max(np.hypot(*v), 1e-9)
        # a nearly-horizontal/vertical piece that continues tangentially between two curved pieces is part of the curve, not a straight edge
        nconv = 0; chg = True
        while chg:
            chg = False; mc = len(items)
            for k in range(mc):
                c_, p_, n_ = items[k], items[k - 1], items[(k + 1) % mc]
                if c_['t'] == 'S' and c_.get('fit') and 1.0 <= c_['len'] <= 40 and not p_.get('axis') and not n_.get('axis') and p_ is not n_:
                    if abs(turn(chordof(p_), chordof(c_))) < 25 and abs(turn(chordof(c_), chordof(n_))) < 25:
                        c_.update(t='C', fit=False, axis=None); nconv += 1; chg = True
        if nconv: say('face %d: %d almost-straight pieces inside smooth curves returned to the curve' % (f['id'], nconv))
        m0 = len(items); st = next((k for k in range(m0) if not cand(items[k])), None)
        if st is not None:
            seq = [items[(st + k) % m0] for k in range(m0)]; outl = []; k = 0
            while k < m0:
                if cand(seq[k]):
                    j = k; sg = None
                    while j + 1 < m0 and cand(seq[j + 1]):
                        t_ = turn(chordof(seq[j]), chordof(seq[j + 1]))
                        if not (1.0 <= abs(t_) <= 35): break
                        if abs(t_) > 12 and sg is not None and np.sign(t_) != sg: break
                        if abs(t_) > 12: sg = np.sign(t_)
                        elif sg is None: sg = np.sign(t_)
                        j += 1
                    if j - k + 1 >= 3:
                        pts = seq[k]['pts']
                        for q in range(k + 1, j + 1): pts = np.r_[pts, seq[q]['pts'][1:]]
                        outl.append(dict(t='C', v0=seq[k]['v0'], v1=seq[j]['v1'], pts=pts, len=0, fit=False, axis=None)); k = j + 1; continue
                outl.append(seq[k]); k += 1
            mer = []
            for it_ in outl:
                if mer and mer[-1]['t'] == 'C' and it_['t'] == 'C': mer[-1]['pts'] = np.r_[mer[-1]['pts'], it_['pts'][1:]]; mer[-1]['v1'] = it_['v1']
                else: mer.append(it_)
            items = mer
        # fake staircase / tooth on a slanted line: DP + axis snapping turns a gently slanted straight line (4-20 deg) into H/V/H pieces, and a small stop-tooth sitting on it into a flat
        # step. If the raw points of such a run stay within 1 mm of ONE straight chord, trace them faithfully (polyline, eps 0.1 mm) instead of inventing axis steps.
        def unstair(items, nV):
            out = []; k = 0; mS = len(items); nun = 0; newV = []
            def small(x): return x['t'] == 'S' and not x.get('ledge') and not x.get('arc') and float(np.sum(np.hypot(*np.diff(x['pts'], axis=0).T))) <= 6
            while k < mS:
                j = k
                if small(items[k]):
                    while j + 1 < mS and small(items[j + 1]): j += 1
                if j - k + 1 >= 3:
                    pts = items[k]['pts']
                    for q in range(k + 1, j + 1): pts = np.r_[pts, items[q]['pts'][1:]]
                    ch = pts[-1] - pts[0]; Lr = float(np.hypot(*ch)); thr = abs(math.degrees(math.atan2(ch[1], ch[0])))
                    tilt = min(abs(((thr + 90) % 180) - 90), abs((thr % 180) - 90)); c_, d_ = tls(pts); n_ = np.array([-d_[1], d_[0]]); dv_ = float(np.abs((pts - c_) @ n_).max())
                    if 4.0 <= tilt <= 20 and Lr >= 3.0 and dv_ < 1.0:
                        ap = cv2.approxPolyDP(pts.astype(np.float32).reshape(-1, 1, 2), 0.10, False).reshape(-1, 2).astype(float); ap[0] = pts[0]; ap[-1] = pts[-1]
                        if len(ap) == 2 and dv_ < 0.15:
                            r_ = fit_edge(pts, sgn, ch / Lr, Lr); it_ = dict(t='S', v0=items[k]['v0'], v1=items[j]['v1'], pts=pts, len=Lr, fit=False, axis=None)
                            if r_: it_.update(r_)
                            else: it_['t'] = 'C'
                            out.append(it_)
                        else:
                            ids = [items[k]['v0']] + [nV + len(newV) + q for q in range(len(ap) - 2)] + [items[j]['v1']]; newV.extend(ap[1:-1])
                            for q in range(len(ap) - 1):
                                p0_, p1_ = ap[q], ap[q + 1]; Ls_ = float(np.hypot(*(p1_ - p0_))); u_ = (p1_ - p0_) / max(Ls_, 1e-9)
                                out.append(dict(t='S', v0=ids[q], v1=ids[q + 1], pts=np.array([p0_, p1_]), len=Ls_, fit=True, axis=None, c=p0_.copy(), d=u_, dev=0.05, raw=True))
                        nun += 1; k = j + 1; continue
                out.extend(items[k:j + 1]); k = j + 1
            return out, nun, newV
        items, nun_, newV_ = unstair(items, len(V))
        if newV_: V = np.vstack([np.asarray(V), np.array(newV_)])
        if nun_: say('face %d: %d slanted runs (fake staircase / small tooth) traced faithfully instead of axis steps' % (f['id'], nun_))
        # stop-teeth inside smooth curve chains (v9 logic): the spline fit would flatten a sharp ~0.5-1 mm tooth sitting on a curve/slanted run. Find sharp apexes of the raw DP polygon of the
        # chain (sag > 0.35 mm from the chord of the neighbours, short legs) and trace +-2.5 mm around them faithfully (raw pieces, polyline eps 0.10); the rest stays a smooth curve.
        def tooth_split(items, nV):
            out = []; newV = []; ntk = 0
            for itx in items:
                P = itx['pts']
                if itx['t'] != 'C' or len(P) < 40 or itx.get('raw'): out.append(itx); continue
                W = cv2.approxPolyDP(P.astype(np.float32).reshape(-1, 1, 2), 0.30, False).reshape(-1, 2).astype(float)
                cum = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; tr = cKDTree(P); wins = []
                for j in range(1, len(W) - 1):
                    b_ = W[j]; k = int(tr.query(b_)[1])
                    if cum[k] < 3.0 or cum[-1] - cum[k] < 3.0: continue
                    a_ = P[int(np.searchsorted(cum, cum[k] - 3.0))]; c_ = P[min(int(np.searchsorted(cum, cum[k] + 3.0)), len(P) - 1)]
                    ch_ = c_ - a_; Lh_ = float(np.hypot(*ch_))
                    if Lh_ < 2.0: continue
                    sg_ = abs(ch_[0] * (b_ - a_)[1] - ch_[1] * (b_ - a_)[0]) / Lh_
                    if __import__('os').environ.get('DBGT') and sg_ > 0.15: say('DBGT sag %.2f at %s' % (sg_, np.round(b_, 1)))
                    if sg_ < 0.25: continue
                    wins.append((int(np.searchsorted(cum, cum[k] - 2.5)), int(np.searchsorted(cum, cum[k] + 2.5))))
                if not wins: out.append(itx); continue
                wins.sort(); mw = [list(wins[0])]
                for a_, b_ in wins[1:]:
                    if a_ <= mw[-1][1]: mw[-1][1] = max(mw[-1][1], b_)
                    else: mw.append([a_, b_])
                mw = [[max(a_, 0), min(b_, len(P) - 1)] for a_, b_ in mw]
                mw = [w_ for w_ in mw if w_[1] - w_[0] > 5]
                if not mw: out.append(itx); continue
                if mw[0][0] < 12: mw[0][0] = 0
                if len(P) - 1 - mw[-1][1] < 12: mw[-1][1] = len(P) - 1
                pos = 0; first_v = itx['v0']; pieces = []
                for a_, b_ in mw:
                    if a_ - pos >= 12: pieces.append(('C', pos, a_))
                    pieces.append(('R', a_ if a_ - pos >= 12 else pos, b_)); pos = b_
                if len(P) - 1 - pos >= 12: pieces.append(('C', pos, len(P) - 1))
                elif pieces: pieces[-1] = (pieces[-1][0], pieces[-1][1], len(P) - 1)
                bidx = [first_v]
                for q in range(len(pieces) - 1): newV.append(P[pieces[q][2]]); bidx.append(nV + len(newV) - 1)
                bidx.append(itx['v1'])
                for q, (kind, a_, b_) in enumerate(pieces):
                    seg = P[a_:b_ + 1]
                    if kind == 'C':
                        d_ = dict(itx); d_.pop('cxy', None); d_.update(pts=seg, v0=bidx[q], v1=bidx[q + 1], len=float(np.hypot(*(seg[-1] - seg[0])))); out.append(d_)
                    else:
                        ap = cv2.approxPolyDP(seg.astype(np.float32).reshape(-1, 1, 2), 0.10, False).reshape(-1, 2).astype(float); ap[0] = seg[0]; ap[-1] = seg[-1]
                        ids = [bidx[q]]
                        for z in ap[1:-1]: newV.append(z); ids.append(nV + len(newV) - 1)
                        ids.append(bidx[q + 1])
                        tseg = cKDTree(seg); ix = [int(tseg.query(z)[1]) for z in ap]
                        for z in range(len(ap) - 1):
                            p0_, p1_ = ap[z], ap[z + 1]; Ls_ = float(np.hypot(*(p1_ - p0_))); u_ = (p1_ - p0_) / max(Ls_, 1e-9)
                            sub = seg[min(ix[z], ix[z + 1]):max(ix[z], ix[z + 1]) + 1]; c_ = p0_.copy(); d_ = u_
                            nz_ = len(ap) - 1; trim = 8
                            if nz_ >= 2 and (z == 0 or z == nz_ - 1):      # flank pieces: fit the line over the long straight stretch next to the tooth (apex is rounded by blur), like the v9 edge fits
                                for span_ in (6.0, 4.0, 2.5):
                                    if z == 0: lo_, hi_ = int(np.searchsorted(cum, cum[a_] - span_)), a_ + ix[1] - trim
                                    else: lo_, hi_ = a_ + ix[z] + trim, min(int(np.searchsorted(cum, cum[b_] + span_)), len(P) - 1)
                                    if hi_ - lo_ < 15: continue
                                    ext = P[lo_:hi_ + 1]; ce_, de_ = tls(ext); ne_ = np.array([-de_[1], de_[0]])
                                    if float(np.abs((ext - ce_) @ ne_).max()) < 0.12:
                                        c_, d_ = ce_, (de_ if de_ @ u_ > 0 else -de_); sub = None; break
                            if sub is not None and len(sub) >= 6:      # fit the line through the raw points: adjacent pieces then meet in a SHARP apex (intersection), like the v9 edge fits
                                c_, d_ = tls(sub)
                                if d_ @ u_ < 0: d_ = -d_
                            out.append(dict(t='S', v0=ids[z], v1=ids[z + 1], pts=(seg[min(ix[z], ix[z + 1]):max(ix[z], ix[z + 1]) + 1] if sub is None else (sub if len(sub) >= 2 else np.array([p0_, p1_]))), len=Ls_, fit=True, axis=None, c=c_, d=d_, dev=0.05, raw=True))
                ntk += len(mw)
            return out, ntk, newV
        if __import__('os').environ.get('DBGT') and f['id'] == 24:
            for q_, x_ in enumerate(items):
                for tgt in ((65.9, -144.7), (65.2, -252.4)):
                    dd_ = np.hypot(*(x_['pts'] - tgt).T).min()
                    if dd_ < 1.0: say('DBGT item %d %s n=%d raw=%s dist %.2f to %s' % (q_, x_['t'], len(x_['pts']), x_.get('raw'), dd_, tgt))
        items, ntk_, newV2_ = tooth_split(items, len(V))
        if newV2_: V = np.vstack([np.asarray(V), np.array(newV2_)])
        if ntk_: say('face %d: %d stop-teeth on curves kept as faithful raw pieces' % (f['id'], ntk_))
        # axis rescue: a short curve/slanted piece squeezed between two axis-aligned neighbours is almost surely a straight axis edge that the scan distorted
        # (gap in the laser line, shadow, slightly tilted tread) -> force it onto the axis (perpendicular step, collinear bridge or corner leg)
        def axis_rescue(items):
            mR = len(items); nres = 0
            if mR < 3: return 0
            for i_ in range(mR):
                it_ = items[i_]; pe_, ne_ = items[i_ - 1], items[(i_ + 1) % mR]
                if it_.get('axis') or it_.get('ledge') or it_.get('arc') or not (pe_.get('axis') and ne_.get('axis')): continue
                P_ = it_['pts']; pl_ = float(np.sum(np.hypot(*np.diff(P_, axis=0).T))); ch_ = P_[-1] - P_[0]; Lc_ = float(np.hypot(*ch_))
                if __import__('os').environ.get('DBGR') and abs(P_[0][0] - 74.4) < 0.6: say('DBGR %s %s %s %s pl %.1f ch %s' % (pe_.get('axis'), ne_.get('axis'), pe_['t'], ne_['t'], pl_, np.round(ch_, 2)))
                if pl_ > 12 or Lc_ < 0.8: continue
                aH = min(abs(((math.degrees(math.atan2(ch_[1], ch_[0])) + 90) % 180) - 90), 90); aV = abs(90 - aH)
                opts = []
                if pe_['axis'] == ne_['axis']:
                    A_ = pe_['axis']; perp_ = 'H' if A_ == 'V' else 'V'
                    opts.append((perp_, (aH if perp_ == 'H' else aV), 22 if pl_ <= 6 else 8))
                    if abs(pe_['coord'] - ne_['coord']) < 0.5: opts.append((A_, (aH if A_ == 'H' else aV), 8))
                else: lim_ = 22 if pl_ <= 4 else 8; opts.append(('H', aH, lim_)); opts.append(('V', aV, lim_))   # short leg: a nearly-axis continuation of the neighbour, not a chamfer
                opts = [o for o in opts if o[1] <= o[2]]
                if not opts: continue
                if os.environ.get('DBGKEEP'): print('RESC (%.1f,%.1f) Lc %.1f pl %.1f aH %.1f aV %.1f n %d opts %s' % (P_[0][0], P_[0][1], Lc_, pl_, aH, aV, len(P_), opts))
                if not os.environ.get('NOKEEPTILT') and Lc_ >= 1.5 and 4.0 <= min(aH, aV) <= 22.0 and len(P_) >= 8:   # a clean straight piece with a real tilt (not a rounded corner / noise): keep it slanted
                    cs_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P_, axis=0).T))]; Pm_ = P_[(cs_ > 0.6) & (cs_ < cs_[-1] - 0.6)]
                    if len(Pm_) >= 8 and cs_[-1] >= 2.6:
                        c2_, d2_ = tls(Pm_); dv2_ = pct(np.abs((Pm_ - c2_) @ np.array([-d2_[1], d2_[0]])), 95)
                        tl2_ = abs(((math.degrees(math.atan2(d2_[1], d2_[0])) + 90) % 180) - 90); tl2_ = min(tl2_, 90 - tl2_)
                        if os.environ.get('DBGKEEP'): print('KEEPCHK (%.1f,%.1f) Lc %.1f pl %.1f tilt %.1f dv %.3f' % (P_[0][0], P_[0][1], Lc_, pl_, tl2_, dv2_))
                        if dv2_ < 0.06 and 5.0 <= tl2_ <= 14.0:
                            KEPT_T.add(f['id']); say('kept-slanted piece at (%.1f,%.1f) len %.1f kept slanted (tilt %.1f deg, straightness %.3f mm)' % (P_[0][0], P_[0][1], Lc_, tl2_, dv2_)); continue
                want_ = min(opts, key=lambda o: o[1])[0]
                r_ = fit_edge(P_, sgn, ch_ / Lc_, Lc_, force=want_, fdev=0.8)
                if r_:
                    it_.update(r_); it_['t'] = 'S'; it_['len'] = Lc_; nres += 1
                    nbs = [q_ for q_ in (pe_, ne_) if q_.get('axis') == it_['axis'] and abs(q_['coord'] - it_['coord']) < 0.5]   # continue the neighbour's line exactly (no 0.2 mm jog)
                    if nbs: it_['coord'] = max(nbs, key=lambda q_: q_.get('len', 0))['coord']
            return nres
        if os.environ.get('DBGBAR') and f['id'] in (10, 11, 12):
            for it_ in items: print('ITEM f%d t=%s axis=%s len=%.2f fit=%s p0=%s p1=%s n=%d' % (f['id'], it_['t'], it_.get('axis'), it_.get('len', 0), it_.get('fit'), np.round(it_['pts'][0], 1), np.round(it_['pts'][-1], 1), len(it_['pts'])))
        nres = axis_rescue(items)
        if nres: say('face %d: %d short slanted/curved pieces between axis edges forced onto the axis' % (f['id'], nres))
        # small circular notches / round ends (e.g. a 4.2 mm round cut into an edge): a run of short pieces that fits one circle
        def plen(p): return float(np.sum(np.hypot(*np.diff(p, axis=0).T))) if len(p) > 1 else 0.0
        def kasa(P_):
            A_ = np.c_[2 * P_, np.ones(len(P_))]; sol = np.linalg.lstsq(A_, (P_ ** 2).sum(1), rcond=None)[0]
            c_ = sol[:2]; r_ = math.sqrt(max(sol[2] + c_ @ c_, 0)); return c_, r_
        nI = len(items); small = [plen(it_['pts']) <= 3.5 for it_ in items]; narc = 0
        if nI > 4 and not all(small):
            s0 = next(k for k in range(nI) if not small[k]); order_ = [(s0 + k) % nI for k in range(nI)]; newit = []; k = 0
            while k < nI:
                if not small[order_[k]]: newit.append(items[order_[k]]); k += 1; continue
                j = k
                while j + 1 < nI and small[order_[j + 1]]: j += 1
                run = [items[order_[q]] for q in range(k, j + 1)]; head = []; tail = []; arc_it = None
                while len(run) >= 3:
                    P_ = np.concatenate([r_['pts'] if q == 0 else r_['pts'][1:] for q, r_ in enumerate(run)])
                    c_, r_ = kasa(P_); res = np.abs(np.hypot(*(P_ - c_).T) - r_)
                    if 0.8 <= r_ <= 3.2 and pct(res, 95) < 0.20:
                        ang_ = np.unwrap(np.arctan2(P_[:, 1] - c_[1], P_[:, 0] - c_[0])); span = abs(math.degrees(ang_[-1] - ang_[0]))
                        mono = bool(np.all(np.diff(ang_) * np.sign(ang_[-1] - ang_[0]) > -0.05))
                        if span >= 100 and mono:
                            r2 = round(r_ / 0.05) * 0.05; na = max(int(abs(ang_[-1] - ang_[0]) * r2 / 0.2), 4); aa = np.linspace(ang_[0], ang_[-1], na + 1)
                            arcp = np.c_[c_[0] + r2 * np.cos(aa), c_[1] + r2 * np.sin(aa)]
                            arc_it = dict(t='C', v0=run[0]['v0'], v1=run[-1]['v1'], pts=arcp, len=0, fit=False, axis=None, arc=True); narc += 1
                            say('face %d: circular notch R=%.2f mm, %.0f deg, centre (%.2f, %.2f), fit 95%% %.3f mm' % (f['id'], r2, span, c_[0], c_[1], pct(res, 95)))
                            break
                    e0 = float(np.max(np.abs(np.hypot(*(run[0]['pts'] - c_).T) - r_))); e1 = float(np.max(np.abs(np.hypot(*(run[-1]['pts'] - c_).T) - r_)))
                    if e0 >= e1: head.append(run.pop(0))
                    else: tail.append(run.pop())
                newit.extend(head)
                newit.extend([arc_it] if arc_it is not None else run)
                newit.extend(tail[::-1]); k = j + 1
            if narc: items = newit
        # rough-arc fix: a chain of >=2 consecutive non-axis pieces (curve fragments and short tilted 'straights') is one smooth curve unless it has a real corner
        Vx = [v for v in V]; mI = len(items)
        def isn(i): return (i['t'] == 'C' or (i['t'] == 'S' and i.get('axis') is None)) and not i.get('arc') and not i.get('raw')
        if mI > 3 and not all(isn(i) for i in items):
            s0 = next(k for k in range(mI) if not isn(items[k])); seq = [items[(s0 + k) % mI] for k in range(mI)]; res_ = []; k = 0; nrm_ = 0
            while k < mI:
                if not isn(seq[k]): res_.append(seq[k]); k += 1; continue
                j = k
                while j + 1 < mI and isn(seq[j + 1]): j += 1
                run = seq[k:j + 1]; k = j + 1
                if len(run) < 2: res_.extend(run); continue
                P_ = np.concatenate([r_['pts'] if q == 0 else r_['pts'][1:] for q, r_ in enumerate(run)])
                sl_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P_, axis=0).T))]
                if sl_[-1] < 3.0: res_.extend(run); continue
                tt_ = np.arange(0, sl_[-1], 0.2); Q_ = np.c_[np.interp(tt_, sl_, P_[:, 0]), np.interp(tt_, sl_, P_[:, 1])]
                Wn = 10; nq = len(Q_); cut = []
                if nq > 2 * Wn + 2:
                    tn = np.zeros(nq)
                    for q in range(Wn, nq - Wn):
                        v1 = Q_[q] - Q_[q - Wn]; v2 = Q_[q + Wn] - Q_[q]; tn[q] = abs(turn(v1 / max(np.hypot(*v1), 1e-9), v2 / max(np.hypot(*v2), 1e-9)))
                    for q in range(Wn, nq - Wn):
                        if tn[q] > 40 and tn[q] == tn[max(q - Wn, 0):q + Wn + 1].max() and (not cut or q - cut[-1] > Wn): cut.append(q)
                bounds = [0] + cut + [nq - 1]; subs = []
                for a_, b_ in zip(bounds[:-1], bounds[1:]):
                    sub = Q_[a_:b_ + 1]
                    if len(sub) >= 2: subs.append(sub)
                for si, sub in enumerate(subs):
                    pts_ = sub.copy(); pts_[0] = (P_[0] if si == 0 else pts_[0]); pts_[-1] = (P_[-1] if si == len(subs) - 1 else pts_[-1])
                    v0_ = run[0]['v0'] if si == 0 else len(Vx)
                    if si > 0: Vx.append(pts_[0].copy())
                    v1_ = run[-1]['v1'] if si == len(subs) - 1 else len(Vx)
                    Ls_ = float(np.hypot(*(pts_[-1] - pts_[0])))
                    r2_ = fit_edge(pts_, sgn, (pts_[-1] - pts_[0]) / max(Ls_, 1e-9), Ls_) if Ls_ >= 1.0 else None
                    if r2_ is not None and r2_['dev'] < 0.12 and not r2_['axis']:
                        it_ = dict(t='S', v0=v0_, v1=v1_, pts=pts_, len=Ls_, fit=True, axis=None); it_.update(r2_)
                    else: it_ = dict(t='C', v0=v0_, v1=v1_, pts=pts_, len=0, fit=False, axis=None)
                    res_.append(it_)
                nrm_ += 1
            if nrm_: items = res_; V = np.array(Vx); say('face %d: %d chains of curve fragments re-fitted as smooth curves' % (f['id'], nrm_))
            if nrm_:
                nres2 = axis_rescue(items)
                if nres2: say('face %d: %d more pieces forced onto the axis after curve merge' % (f['id'], nres2))
        # stop ledges: a small (0.35-1.0 mm) lateral offset in an otherwise smooth wall is a real feature (a stop when the part is inserted into a slot), keep it as a crisp step
        def find_ledge(P):
            sl_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; Lc_ = sl_[-1]
            if Lc_ < 7.5: return None
            tq = np.arange(0, Lc_, 0.05); Q = np.c_[np.interp(tq, sl_, P[:, 0]), np.interp(tq, sl_, P[:, 1])]; Wd = 3.0; best_ = None
            for s0 in np.arange(Wd + 0.3, Lc_ - Wd - 0.3, 0.15):
                iL = (tq > s0 - Wd) & (tq < s0 - 0.3); iR = (tq > s0 + 0.3) & (tq < s0 + Wd)
                q0 = Q[np.searchsorted(tq, s0)]; tau = Q[np.searchsorted(tq, s0 + Wd)] - Q[np.searchsorted(tq, s0 - Wd)]; tau = tau / max(np.hypot(*tau), 1e-9); nu_ = np.array([-tau[1], tau[0]])
                a_ = (Q - q0) @ tau; l_ = (Q - q0) @ nu_
                cL = np.polyfit(a_[iL], l_[iL], 2); cR = np.polyfit(a_[iR], l_[iR], 2)
                rL = float(np.sqrt(np.mean((np.polyval(cL, a_[iL]) - l_[iL]) ** 2))); rR = float(np.sqrt(np.mean((np.polyval(cR, a_[iR]) - l_[iR]) ** 2)))
                dl = cR[2] - cL[2]
                if max(rL, rR) < (0.06 if abs(dl) < 0.3 else 0.08) and 0.20 <= abs(dl) <= 1.0 and abs(cR[1] - cL[1]) < 0.2:
                    sc_ = abs(dl) / (rL + rR + 0.03)
                    if best_ is None or sc_ > best_[0]: best_ = (sc_, s0, cL[2], cR[2], tau, nu_, q0)
            return best_
        # a short piece (<2.5 mm) between two walls that are offset sideways by 0.25-1 mm is the stop ledge itself: make it a square step (perpendicular to the wall)
        for k in range(len(items)):
            sI, A_, B_ = items[k], items[k - 1], items[(k + 1) % len(items)]
            if not (sI['t'] == 'S' and not sI.get('arc') and len(sI['pts']) >= 2 and float(np.sum(np.hypot(*np.diff(sI['pts'], axis=0).T))) <= 12.0): continue
            plen_ = float(np.sum(np.hypot(*np.diff(sI['pts'], axis=0).T)))
            if sI.get('axis') and sI.get('fit') and sI['len'] > 0.8 and (A_.get('axis') == sI['axis'] or B_.get('axis') == sI['axis']): continue
            walls = [x for x in (A_, B_) if x['t'] == 'S' and x.get('axis') and x.get('fit')]
            if not walls: continue
            X = walls[0]['axis']
            if len(walls) == 2 and walls[1]['axis'] != X: continue
            if len(walls) == 1 and (A_ if walls[0] is B_ else B_)['t'] != 'C': continue
            if A_ is B_: continue
            def lat(x_, end_):   # lateral coordinate (x for V walls, y for H walls) of an item end
                p_ = x_['pts'][-1] if end_ else x_['pts'][0]; return p_[0] if X == 'V' else p_[1]
            la = (walls[0]['coord'] if A_ is walls[0] else lat(A_, True)); lb = (walls[-1]['coord'] if (B_ is walls[-1]) else lat(B_, False))
            if len(walls) == 2: la, lb = A_['coord'], B_['coord']
            off_ = abs(lb - la)
            vv_ = sI['pts'][-1] - sI['pts'][0]; th_ = math.degrees(math.atan2(vv_[1], vv_[0])); tilt_ = abs(((th_ + 90) % 180) - 90) if X == 'V' else abs((th_ % 180) - 90)
            stair_ = (len(walls) == 2 and off_ > 1.0 and sI.get('axis') is None and tilt_ < 5)
            if not ((0.2 <= off_ <= 1.0 and plen_ <= 2.5) or stair_): continue
            want = 'H' if X == 'V' else 'V'; P_ = sI['pts']; mid = float(np.median(P_[:, 1] if want == 'H' else P_[:, 0]))
            sg_ = 1.0 if lb > la else -1.0; dd = np.array([sg_, 0.]) if want == 'H' else np.array([0., sg_]); nout = np.array([dd[1], -dd[0]]) * sgn
            cc = mid + OFF * np.sign(nout[1 if want == 'H' else 0])
            sI.update(t='S', fit=True, axis=want, coord=float(cc), d=dd, dev=0.1, len=float(off_), ledge=True)
            sI['c'] = np.array([0., cc]) if want == 'H' else np.array([cc, 0.]); nled0 = 1
            say('face %d: stop ledge %.2f mm (slanted piece) made a square step' % (f['id'], off_))
        nled = 0; ch_ = True; guard = 0
        while ch_ and guard < 12:
            ch_ = False; guard += 1
            for k in range(len(items)):
                it_ = items[k]
                if it_.get('arc') or it_['t'] not in ('S', 'C') or it_.get('ledge'): continue
                if it_['t'] == 'S' and not it_.get('fit'): continue
                lg = find_ledge(it_['pts'])
                if lg is None: continue
                sc_, s0, LA, LB, tau, nu_, q0 = lg; P = it_['pts']; sl_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
                tq = np.arange(0, sl_[-1], 0.05); Q = np.c_[np.interp(tq, sl_, P[:, 0]), np.interp(tq, sl_, P[:, 1])]
                Apts = Q[tq < s0 - 0.2]; Bpts = Q[tq > s0 + 0.2]
                pA = q0 + nu_ * LA; pB = q0 + nu_ * LB; iA = len(Vx); Vx.append(pA.copy()); iB = len(Vx); Vx.append(pB.copy())
                if it_['t'] == 'S':
                    LAl = float(np.hypot(*(Apts[-1] - Apts[0]))); LBl = float(np.hypot(*(Bpts[-1] - Bpts[0])))
                    rA = fit_edge(Apts, sgn, tau, LAl); rB = fit_edge(Bpts, sgn, tau, LBl)
                    if rA is None or rB is None: Vx.pop(); Vx.pop(); continue
                    IA = dict(t='S', v0=it_['v0'], v1=iA, pts=Apts, len=LAl, fit=True, axis=None, ledge=True); IA.update(rA)
                    IB = dict(t='S', v0=iB, v1=it_['v1'], pts=Bpts, len=LBl, fit=True, axis=None, ledge=True); IB.update(rB)
                else:
                    IA = dict(t='C', v0=it_['v0'], v1=iA, pts=Apts, len=0, fit=False, axis=None, ledge=True)
                    IB = dict(t='C', v0=iB, v1=it_['v1'], pts=Bpts, len=0, fit=False, axis=None, ledge=True)
                dstep = nu_ * (1 if LB > LA else -1); c0 = q0 + nu_ * (LA + LB) / 2; nout = np.array([dstep[1], -dstep[0]]) * sgn
                IS = dict(t='S', v0=iA, v1=iB, pts=np.array([pA, pB]), len=abs(LB - LA), fit=True, dev=0.0, axis=None, ledge=True)
                thS = math.degrees(math.atan2(dstep[1], dstep[0]))
                if abs(((thS + 90) % 180) - 90) < 3.5: IS.update(axis='H', coord=float(c0[1]) + OFF * np.sign(nout[1]), d=np.array([np.sign(dstep[0]), 0.]), c=c0)
                elif abs((thS % 180) - 90) < 3.5: IS.update(axis='V', coord=float(c0[0]) + OFF * np.sign(nout[0]), d=np.array([0., np.sign(dstep[1])]), c=c0)
                else: IS.update(c=c0 + OFF * nout, d=dstep)
                items[k:k + 1] = [IA, IS, IB]; nled += 1; ch_ = True
                say('face %d: stop ledge %.2f mm kept as a square step' % (f['id'], abs(LB - LA))); break
        # junction ledges: the offset sits exactly where a curve meets a wall (no separate short piece) -> look across the junction
        def tan_ok(A_, B_):
            def tg(P, end):
                sl_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
                if end: i0, i1 = np.searchsorted(sl_, max(sl_[-1] - 6.0, 0)), np.searchsorted(sl_, max(sl_[-1] - 2.5, 0.5))
                else: i0, i1 = np.searchsorted(sl_, 2.5), np.searchsorted(sl_, 6.0)
                i1 = min(i1, len(P) - 1); i0 = min(i0, i1 - 1)
                v = (P[i1] - P[i0]); return v / max(np.hypot(*v), 1e-9)
            return abs(turn(tg(A_['pts'], True), tg(B_['pts'], False))) < 15
        k = 0
        while k < len(items):
            A_, B_ = items[k], items[(k + 1) % len(items)]
            ok_ = (A_ is not B_ and not A_.get('arc') and not B_.get('arc') and not A_.get('ledge') and not B_.get('ledge') and A_['t'] in ('S', 'C') and B_['t'] in ('S', 'C') and (A_['t'] == 'C' or B_['t'] == 'C')
                   and len(A_['pts']) > 3 and len(B_['pts']) > 3 and tan_ok(A_, B_))
            if ok_:
                def tail(P, L_, end):
                    sl_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
                    return P[sl_ >= sl_[-1] - L_] if end else P[sl_ <= L_]
                TA, TB = tail(A_['pts'], 7.0, True), tail(B_['pts'], 7.0, False); lenA = float(np.sum(np.hypot(*np.diff(TA, axis=0).T)))
                lg = find_ledge(np.r_[TA, TB[1:]])
                if __import__('os').environ.get('DBGJ') and f['id'] == int(__import__('os').environ['DBGJ']): say('DBGJ junction %d %s/%s at %s lenA %.1f lg %s' % (k, A_['t'], B_['t'], np.round(A_['pts'][-1], 1), lenA, None if lg is None else (round(lg[1], 1), round(lg[2], 2), round(lg[3], 2))))
                if lg is not None and abs(lg[1] - lenA) < 2.5:
                    sc_, s0, LA, LB, tau, nu_, q0 = lg; sgm = 1 if LB > LA else -1
                    # cut the pieces back to the ledge zone
                    cutA = s0 - 0.2 - lenA   # >0 : ledge lies inside B
                    if cutA < 0:
                        P = A_['pts']; sl_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; A_['pts'] = P[sl_ <= sl_[-1] + cutA]
                    P = B_['pts']; sl_ = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; startB = max(s0 + 0.2 - lenA, 0.0); B_['pts'] = P[sl_ >= startB]
                    if len(A_['pts']) > 3 and len(B_['pts']) > 3:
                        pA = q0 + nu_ * LA; pB = q0 + nu_ * LB; iA = len(Vx); Vx.append(pA.copy()); iB = len(Vx); Vx.append(pB.copy())
                        dstep = nu_ * sgm; c0 = q0 + nu_ * (LA + LB) / 2; nout = np.array([dstep[1], -dstep[0]]) * sgn
                        IS = dict(t='S', v0=iA, v1=iB, pts=np.array([pA, pB]), len=abs(LB - LA), fit=True, dev=0.0, axis=None, ledge=True)
                        thS = math.degrees(math.atan2(dstep[1], dstep[0]))
                        if abs(((thS + 90) % 180) - 90) < 3.5: IS.update(axis='H', coord=float(c0[1]) + OFF * np.sign(nout[1]), d=np.array([np.sign(dstep[0]), 0.]), c=c0)
                        elif abs((thS % 180) - 90) < 3.5: IS.update(axis='V', coord=float(c0[0]) + OFF * np.sign(nout[0]), d=np.array([0., np.sign(dstep[1])]), c=c0)
                        else: IS.update(c=c0 + OFF * nout, d=dstep)
                        A_['v1'] = iA; B_['v0'] = iB
                        items.insert(k + 1, IS); nled += 1; say('face %d: junction stop ledge %.2f mm kept as a square step' % (f['id'], abs(LB - LA))); k += 1
            k += 1
        if nled: V = np.array(Vx)
        faces.append(dict(id=f['id'], V=V, C=C, items=items, sgn=sgn))
        say('face %d: DP vertices %d, items %d (S %d, C %d, axis-fitted %d)' % (f['id'], n, len(items), sum(i['t'] == 'S' for i in items), sum(i['t'] == 'C' for i in items), sum(i.get('axis') is not None for i in items)))
    # ---------- coordinate clusters ----------
    allw = []
    for fc in faces:
        for ax in 'HV':
            E = [i for i in fc['items'] if i['t'] == 'S' and i.get('axis') == ax]; E.sort(key=lambda i: i['coord']); cl = []
            for i in E:
                if cl and i['coord'] - cl[-1][-1]['coord'] <= CLTOL: cl[-1].append(i)
                else: cl.append([i])
            fc.setdefault('cl', {})[ax] = []
            for k, gq in enumerate(cl):
                w = np.array([min(i['len'], 60) for i in gq]); mval = float(np.average([i['coord'] for i in gq], weights=w))
                for i in gq: i['cid'] = k
                fc['cl'][ax].append(dict(m=mval, w=float(w.sum()), val=mval, n=len(gq)))
        it = fc['items']; m = len(it); fc['slots'] = []
        for k in range(m):
            A_, B_, C_ = it[k], it[(k + 1) % m], it[(k + 2) % m]
            if all(x['t'] == 'S' and x.get('axis') for x in (A_, B_, C_)) and A_['axis'] == C_['axis'] != B_['axis'] and A_['d'] @ C_['d'] < 0:
                ax = A_['axis']; w = abs(fc['cl'][ax][C_['cid']]['m'] - fc['cl'][ax][A_['cid']]['m'])
                if 3.3 <= w <= 4.4: fc['slots'].append((ax, A_['cid'], C_['cid'], w)); allw.append(w)
    allw = np.array(allw)
    if len(allw):
        best = max(allw, key=lambda c: np.sum(np.abs(allw - c) <= 0.1)); sel = allw[np.abs(allw - best) <= 0.1]; WSLOT = float(np.round(np.median(sel), 2))
    else: WSLOT = None
    say('slot-like widths: %d found; common slot width %s mm; values %s' % (len(allw), WSLOT, np.round(np.sort(allw), 2).tolist()))
    for fc in faces:
        for ax in 'HV':
            cls = fc['cl'][ax]
            if not cls: continue
            rows = []; rhs = []; K = len(cls)
            for k, c in enumerate(cls): r = np.zeros(K); r[k] = math.sqrt(c['w']); rows.append(r); rhs.append(math.sqrt(c['w']) * c['m'])
            if WSLOT:
                for (a_, i, j, w) in fc['slots']:
                    if a_ != ax or abs(w - WSLOT) > 0.12: continue
                    lo, hi = (i, j) if cls[i]['m'] < cls[j]['m'] else (j, i)
                    r = np.zeros(K); r[hi] = 100.; r[lo] = -100.; rows.append(r); rhs.append(100. * WSLOT)
            sol = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
            for k, c in enumerate(cls): c['val'] = float(sol[k])
        for i in fc['items']:
            if i['t'] == 'S' and i.get('axis'):
                i['coord'] = fc['cl'][i['axis']][i['cid']]['val']; i['c'] = np.array([0., i['coord']]) if i['axis'] == 'H' else np.array([i['coord'], 0.])
    # ---------- merge collinear ----------
    def same_line(a, b):
        if not (a['t'] == 'S' and b['t'] == 'S' and a['fit'] and b['fit']): return False
        if a['axis'] and a['axis'] == b['axis']: return a['cid'] == b['cid']
        if a['axis'] or b['axis']: return False
        ang = abs(turn(a['d'], b['d'])); nb = np.array([-b['d'][1], b['d'][0]])
        if not (ang < 5 and abs((a['c'] - b['c']) @ nb) < 0.35): return False
        P = np.r_[a['pts'], b['pts']]; c, d = tls(P); n_ = np.array([-d[1], d[0]]); return np.percentile(np.abs((P - c) @ n_), 95) < 0.15
    nchamf = 0
    for fc in faces:
        it = fc['items']; ch = True
        while ch and len(it) > 3:
            ch = False
            for k in range(len(it)):
                a, b = it[k], it[(k + 1) % len(it)]
                if same_line(a, b):
                    a['pts'] = np.r_[a['pts'], b['pts'][1:]]; a['v1'] = b['v1']; a['len'] += b['len']
                    if not a['axis']:
                        c, d = tls(a['pts'])
                        if d @ a['d'] < 0: d = -d
                        a['c'], a['d'] = c, d
                    it.pop((k + 1) % len(it)) if (k + 1) % len(it) > k else it.pop(0); ch = True; break
        # corner rule: a short piece between two straight edges that meet at ~90 deg is a burnt corner blob -> sharp corner
        ch = True
        def chord_len(i): return float(np.hypot(*(i['pts'][-1] - i['pts'][0])))
        while ch and len(it) > 4:
            ch = False
            for k in range(len(it)):
                p_, c_, n_ = it[k - 1], it[k], it[(k + 1) % len(it)]
                if not (p_['t'] == 'S' and n_['t'] == 'S' and p_['fit'] and n_['fit'] and p_ is not n_): continue
                if chord_len(c_) > 2.0 or c_['len'] > 3.0 and c_['t'] == 'S': continue
                if p_['len'] < 1.0 or n_['len'] < 1.0: continue
                tt = abs(turn(p_['d'], n_['d']))
                if 65 <= tt <= 115:
                    it.pop(k); nchamf += 1; ch = True; break
        # second pass (after merging): long straight pieces that continue tangentially between curves belong to the curve
        Vf = fc['V']; chg = True; n2 = 0
        def chv(i):
            v = Vf[i['v1']] - Vf[i['v0']]; return v / max(np.hypot(*v), 1e-9)
        while chg:
            chg = False; mc = len(it)
            for k in range(mc):
                c_, p_, n_ = it[k], it[k - 1], it[(k + 1) % mc]
                if c_['t'] == 'S' and c_.get('fit') and 1.0 <= c_['len'] <= 45 and not p_.get('axis') and not n_.get('axis') and p_ is not n_:
                    if abs(turn(chv(p_), chv(c_))) < 25 and abs(turn(chv(c_), chv(n_))) < 25:
                        c_.update(t='C', fit=False, axis=None); n2 += 1; chg = True
        if n2:
            ch = True
            while ch and len(it) > 2:
                ch = False
                for k in range(len(it)):
                    a, b = it[k], it[(k + 1) % len(it)]
                    if a['t'] == 'C' and b['t'] == 'C' and a is not b:
                        a['pts'] = np.r_[a['pts'], b['pts'][1:]]; a['v1'] = b['v1']; it.pop((k + 1) % len(it)) if (k + 1) % len(it) > k else it.pop(0); ch = True; break
            say('face %d: %d long near-straight pieces joined into the neighbouring smooth curve' % (fc['id'], n2))
        say('face %d: items after merge/corner rule: %d' % (fc['id'], len(it)))
    say('chamfers collapsed to sharp 90-degree corners: %d' % nchamf)
    # ---------- nodes / polygons ----------
    def inter(a, b):
        den = a['d'][0] * b['d'][1] - a['d'][1] * b['d'][0]
        if abs(den) < 0.2: return None
        t = ((b['c'] - a['c'])[0] * b['d'][1] - (b['c'] - a['c'])[1] * b['d'][0]) / den; return a['c'] + a['d'] * t
    def proj(i, p): return i['c'] + i['d'] * ((p - i['c']) @ i['d'])
    def make_curve(B_, sgn, d_in=None, d_out=None):
        P = B_['pts']; sl = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; Lc = sl[-1]
        tt = np.arange(0, Lc, 0.15); Q = np.c_[np.interp(tt, sl, P[:, 0]), np.interp(tt, sl, P[:, 1])]
        if len(Q) >= 8:
            if Lc > 20 and not B_.get('arc'):
                # tiny jogs (<~0.5 mm, laser start/stop marks) on a long smooth arc are not shape: drop points far from a stiff smoothing spline
                t0_, _ = splprep([Q[:, 0], Q[:, 1]], s=len(Q) * 0.12 ** 2, k=3); d0_ = np.array(splev(np.linspace(0, 1, 4000), t0_)).T
                rs_ = cKDTree(d0_).query(Q)[0]; kp = (rs_ < 0.2); kp[:7] = True; kp[-7:] = True
                if kp.sum() >= 8: Q = Q[kp]
            Qa = Q
            if d_in is not None: Qa = np.r_[Q[0] - d_in * 0.15 * np.arange(12, 0, -1)[:, None], Qa]
            if d_out is not None: Qa = np.r_[Qa, Q[-1] + d_out * 0.15 * np.arange(1, 13)[:, None]]
            tck, _ = splprep([Qa[:, 0], Qa[:, 1]], s=len(Qa) * 0.05 ** 2, k=3)
            u = np.linspace(0, 1, 4000); xy = np.array(splev(u, tck)).T
            i0 = int(np.argmin(np.hypot(*(xy - Q[0]).T))); i1 = int(np.argmin(np.hypot(*(xy - Q[-1]).T)))
            if i1 > i0 + 5: xy = xy[i0:i1 + 1]
            s2 = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
            n2 = max(int(round(s2[-1] / 0.5)), 2); q = np.interp(np.linspace(0, s2[-1], n2 + 1), s2, np.arange(len(s2))); xy = xy[np.round(q).astype(int)]
        else: xy = P[[0, -1]]
        if abs(OFF) > 1e-6 and len(xy) > 2:
            tg = np.gradient(xy, axis=0); tg /= np.maximum(np.hypot(*tg.T), 1e-9)[:, None]
            xy = xy + OFF * np.c_[tg[:, 1], -tg[:, 0]] * sgn
        return xy
    def end_line(xy, at_end):
        if len(xy) < 4: return None
        e, o_ = (xy[-1], xy[-4]) if at_end else (xy[0], xy[3]); tg = (e - o_) if at_end else (o_ - e); tg = tg / max(np.hypot(*tg), 1e-9)
        return dict(c=e, d=tg)
    for fc in faces:
        it = fc['items']; m = len(it); nodes = []
        def nbr_dir(S_, into):
            if not (S_['t'] == 'S' and S_.get('fit') and S_.get('axis')): return None
            v = S_['pts'][-1] - S_['pts'][0]; d = S_['d'] * (1 if S_['d'] @ v >= 0 else -1); return d
        def mk(k):
            c_ = it[k]; A_, N_ = it[k - 1], it[(k + 1) % len(it)]; di = nbr_dir(A_, True); do = nbr_dir(N_, False)
            Pp = c_['pts']; sl_ = np.r_[0, np.cumsum(np.hypot(*np.diff(Pp, axis=0).T))]
            if di is not None:
                v = Pp[min(np.searchsorted(sl_, 2.0), len(Pp) - 1)] - Pp[0]
                if abs(turn(di, v / max(np.hypot(*v), 1e-9))) > 25: di = None
            if do is not None:
                v = Pp[-1] - Pp[max(np.searchsorted(sl_, sl_[-1] - 2.0), 0)]
                if abs(turn(v / max(np.hypot(*v), 1e-9), do)) > 25: do = None
            return make_curve(c_, fc['sgn'], di, do)
        for k_ in range(m):
            if it[k_]['t'] == 'C': it[k_]['cxy'] = mk(k_)
        # micro-jogs: a short axis stub (<1.6 mm) between a curve and an axis edge is the curve running into that edge, not a feature
        chg_ = True
        while chg_ and len(it) > 4:
            chg_ = False; mm_ = len(it)
            for k_ in range(mm_):
                s_, A_, N_ = it[k_], it[k_ - 1], it[(k_ + 1) % mm_]
                if not (s_['t'] == 'S' and s_.get('axis') and float(np.sum(np.hypot(*np.diff(s_['pts'], axis=0).T))) < 1.6): continue
                if A_['t'] == 'C' and N_['t'] == 'S' and N_.get('fit') and N_.get('axis') and N_['axis'] != s_['axis'] and 'cxy' in A_:
                    L_ = end_line(A_['cxy'], True); q_ = inter(L_, N_) if L_ is not None else None
                    if q_ is not None and np.hypot(*(q_ - L_['c'])) < 2.2 and (q_ - L_['c']) @ L_['d'] > 0:
                        it.pop(k_); chg_ = True; say('face %d: micro-jog stub removed' % fc['id']); break
                if N_['t'] == 'C' and A_['t'] == 'S' and A_.get('fit') and A_.get('axis') and A_['axis'] != s_['axis'] and 'cxy' in N_:
                    L_ = end_line(N_['cxy'], False); q_ = inter(L_, A_) if L_ is not None else None
                    if q_ is not None and np.hypot(*(q_ - L_['c'])) < 2.2 and (q_ - L_['c']) @ L_['d'] < 0:
                        it.pop(k_); chg_ = True; say('face %d: micro-jog stub removed' % fc['id']); break
        # laser start/stop jogs: curve end, 1-2 short pieces (<4.5 mm in all), then an axis edge lying within 0.9 mm of the curve end -> the curve simply joins that edge
        def plen2(p): return float(np.sum(np.hypot(*np.diff(p, axis=0).T)))
        chg_ = True
        while False and chg_ and len(it) > 5:
            chg_ = False; mm_ = len(it)
            for k_ in range(mm_):
                A_ = it[k_]
                if A_['t'] != 'C' or 'cxy' not in A_: continue
                for nsh in (1, 2):
                    if nsh + 2 > mm_: break
                    sh = [it[(k_ + 1 + q) % mm_] for q in range(nsh)]; N_ = it[(k_ + 1 + nsh) % mm_]
                    if not all(x['t'] == 'S' and plen2(x['pts']) < 3.0 for x in sh) or sum(plen2(x['pts']) for x in sh) >= 4.5: continue
                    if not (N_['t'] == 'S' and N_.get('fit') and N_.get('axis')): continue
                    e_ = A_['cxy'][-1]; dist = abs((e_ - N_['c']) @ np.array([-N_['d'][1], N_['d'][0]]))
                    if dist < 0.9 and sh[0].get('axis') == N_['axis']:
                        for x in sh: it.remove(x)
                        N_['jogfix'] = 'e'; chg_ = True; say('face %d: laser jog (%d pieces) merged into the arc end' % (fc['id'], nsh)); break
                if chg_: break
                # mirrored case: axis edge, jog, curve
            if chg_: continue
            for k_ in range(mm_):
                B_ = it[k_]
                if B_['t'] != 'C' or 'cxy' not in B_: continue
                for nsh in (1, 2):
                    if nsh + 2 > mm_: break
                    sh = [it[(k_ - 1 - q) % mm_] for q in range(nsh)]; N_ = it[(k_ - 1 - nsh) % mm_]
                    if not all(x['t'] == 'S' and plen2(x['pts']) < 3.0 for x in sh) or sum(plen2(x['pts']) for x in sh) >= 4.5: continue
                    if not (N_['t'] == 'S' and N_.get('fit') and N_.get('axis')): continue
                    e_ = B_['cxy'][0]; dist = abs((e_ - N_['c']) @ np.array([-N_['d'][1], N_['d'][0]]))
                    if dist < 0.9 and sh[0].get('axis') == N_['axis']:
                        for x in sh: it.remove(x)
                        B_['jogfix'] = 's'; chg_ = True; say('face %d: laser jog (%d pieces) merged into the arc start' % (fc['id'], nsh)); break
                if chg_: break
        m = len(it)
        for k in range(m):
            A_, B_ = it[k - 1], it[k]; raw = fc['V'][B_['v0']]
            fa = A_['t'] == 'S' and A_['fit']; fb = B_['t'] == 'S' and B_['fit']
            if fa and fb:
                p = inter(A_, B_); p = p if p is not None and np.hypot(*(p - raw)) < 2.5 else proj(B_, raw)
            elif fb:
                p = proj(B_, raw)
                if B_.get('jogfix') == 'e' and A_['t'] == 'C': p = proj(B_, A_['cxy'][-1])
                if A_['t'] == 'C':
                    L_ = end_line(A_['cxy'], True)
                    if L_ is not None and abs(L_['d'][0] * B_['d'][1] - L_['d'][1] * B_['d'][0]) > 0.26:
                        q_ = inter(L_, B_)
                        if q_ is not None and np.hypot(*(q_ - L_['c'])) < 1.8 and (q_ - L_['c']) @ L_['d'] > 0 and (q_ - p) @ B_['d'] < 0.05: p = q_; A_['ext_e'] = True   # only extend outward (sharpen), never pull a spike apex back
                    elif L_ is not None: A_['wl_e'] = 4.0   # tangent junction: spread the small lateral correction over 4 mm instead of a visible kink
            elif fa:
                p = proj(A_, raw)
                if B_.get('jogfix') == 's' and B_['t'] == 'C': p = proj(A_, B_['cxy'][0])
                if B_['t'] == 'C':
                    L_ = end_line(B_['cxy'], False)
                    if L_ is not None and abs(L_['d'][0] * A_['d'][1] - L_['d'][1] * A_['d'][0]) > 0.26:
                        q_ = inter(L_, A_)
                        if q_ is not None and np.hypot(*(q_ - L_['c'])) < 1.8 and (q_ - L_['c']) @ L_['d'] < 0 and (q_ - p) @ A_['d'] > -0.05: p = q_; B_['ext_s'] = True
                    elif L_ is not None: B_['wl_s'] = 4.0
            elif A_['t'] == 'C' and B_['t'] == 'C':
                p = (A_['cxy'][-1] + B_['cxy'][0]) / 2
                L1, L2 = end_line(A_['cxy'], True), end_line(B_['cxy'], False)
                if L1 is not None and L2 is not None and abs(L1['d'][0] * L2['d'][1] - L1['d'][1] * L2['d'][0]) > 0.26 and A_ is not B_:
                    q_ = inter(L1, L2)
                    if q_ is not None and np.hypot(*(q_ - L1['c'])) < 1.8 and np.hypot(*(q_ - L2['c'])) < 1.8 and (q_ - L1['c']) @ L1['d'] > 0 and (q_ - L2['c']) @ L2['d'] < 0:
                        p = q_; A_['ext_e'] = True; B_['ext_s'] = True
            else: p = raw.copy()
            nodes.append(p)
        poly = []; kp = []
        for k in range(m):
            a, b = nodes[k], nodes[(k + 1) % m]; B_ = it[k]
            if B_['t'] == 'S':
                poly.append(a); kp.append(bool(B_.get('raw')) or bool(it[k - 1].get('raw')))
            else:
                xy = B_['cxy']
                s3 = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]; w0 = np.exp(-s3 / B_.get('wl_s', 1.5))[:, None]; w1 = np.exp(-(s3[-1] - s3) / B_.get('wl_e', 1.5))[:, None]
                if B_.get('ext_s'): w0 = w0 * 0; poly.append(a); kp.append(False)
                if B_.get('ext_e'): w1 = w1 * 0
                xy = xy + (a - xy[0]) * w0 + (b - xy[-1]) * w1
                add_ = list(xy if B_.get('ext_e') else xy[:-1]); poly += add_; kp += [False] * len(add_)
        fc['poly'] = np.array(poly)
        # final cleanup: remove scan wobble / 0.1-0.3 mm jogs on straight and slanted runs; corners, real ledges (>0.35 mm) stay; traced teeth ('raw' pieces) are kept as they are
        SIMP = float(__import__('os').environ.get('SIMP', '0.15'))
        if SIMP > 0 and len(poly) > 8:
            P_ = np.array(poly); kp_ = np.array(kp); n0_ = len(P_)
            def dp_open(Q): return cv2.approxPolyDP(Q.astype(np.float32).reshape(-1, 1, 2), SIMP, False).reshape(-1, 2).astype(float)
            if kp_.any():
                idx = list(np.nonzero(kp_)[0]); out_ = []
                for q in range(len(idx)):
                    i0, i1 = idx[q], idx[(q + 1) % len(idx)]
                    seg = P_[i0:i1 + 1] if i1 > i0 else np.r_[P_[i0:], P_[:i1 + 1]]
                    out_ += list(dp_open(seg)[:-1]) if len(seg) > 2 and not (kp_[i0] and kp_[i1] and len(seg) <= 2) else list(seg[:-1])
                ap_ = np.array(out_)
            else: ap_ = cv2.approxPolyDP(P_.astype(np.float32).reshape(-1, 1, 2), SIMP, True).reshape(-1, 2).astype(float)
            if len(ap_) >= 4: fc['poly'] = ap_; say('face %d: contour simplified %d -> %d vertices (eps %.2f mm)' % (fc['id'], n0_, len(ap_), SIMP))
            # stop-teeth / micro-features: where the raw scan contour leaves the simplified outline by >TOOTH mm over a short stretch, trace the raw contour there (eps 0.10)
            TOOTH = float(__import__('os').environ.get('TOOTH', '0'))
            Craw = np.asarray(fc['C'], float) if TOOTH > 0 and fc.get('C') is not None else None
            if Craw is not None and len(Craw) > 20 and len(fc['poly']) >= 4:
                nt_ = 0; ban_ = np.zeros(len(Craw), bool)
                for _ in range(40):
                    A_ = fc['poly']; cl_ = np.r_[A_, A_[:1]]; sl_ = np.r_[0, np.cumsum(np.hypot(*np.diff(cl_, axis=0).T))]
                    dn_ = np.arange(0, sl_[-1], 0.05); Ad = np.c_[np.interp(dn_, sl_, cl_[:, 0]), np.interp(dn_, sl_, cl_[:, 1])]
                    dC_ = cKDTree(Ad).query(Craw)[0]; hit = np.nonzero((dC_ > TOOTH) & ~ban_)[0]
                    if not len(hit): break
                    # cluster contiguous (cyclic) indices, pick the worst
                    cls_ = []; 
                    for i_ in hit:
                        if cls_ and i_ - cls_[-1][-1] <= 15: cls_[-1].append(i_)
                        else: cls_.append([i_])
                    cls_ = [c_ for c_ in cls_ if len(c_) >= 3]
                    if not cls_: break
                    c_ = max(cls_, key=lambda c_: dC_[c_].max()); s_, e_ = c_[0], c_[-1]; N_ = len(Craw)
                    s_ -= 25; e_ += 25
                    seg = Craw[np.arange(s_ - 1, e_ + 2) % N_]
                    ban_[np.arange(c_[0] - 5, c_[-1] + 6) % N_] = True
                    if np.sum(np.hypot(*np.diff(seg, axis=0).T)) > 9.0: continue      # long deviation = not a tooth, leave it
                    dpp = cv2.approxPolyDP(seg.astype(np.float32).reshape(-1, 1, 2), 0.12, False).reshape(-1, 2).astype(float)
                    tA = cKDTree(A_); i_s = int(tA.query(seg[0])[1]); i_e = int(tA.query(seg[-1])[1]); n_ = len(A_)
                    Ar = np.roll(A_, -i_s, axis=0); ie = (i_e - i_s) % n_
                    if ie == 0: continue
                    new_ = np.r_[dpp, Ar[ie + 1:]]
                    if len(new_) < 4: continue
                    fc['poly'] = new_; nt_ += 1
                if nt_: say('face %d: %d small teeth/ledges traced from the raw contour' % (fc['id'], nt_))
    # ---------- mirror symmetry per part: keep the cleaner half and mirror it ----------
    from scipy.optimize import linear_sum_assignment
    def dense(P, step):
        cl = np.r_[P, P[:1]]; sl = np.r_[0, np.cumsum(np.hypot(*np.diff(cl, axis=0).T))]
        tt = np.arange(0, sl[-1], step); return np.c_[np.interp(tt, sl, cl[:, 0]), np.interp(tt, sl, cl[:, 1])]
    def find_axis(P, axis):
        T = cKDTree(dense(P, 0.1)); S = dense(P, 0.4); c0 = (P[:, axis].min() + P[:, axis].max()) / 2; best = None
        for c in np.arange(c0 - 5, c0 + 5.01, 0.05):
            Mi = S.copy(); Mi[:, axis] = 2 * c - Mi[:, axis]; d, _ = T.query(Mi); sc = float(np.mean(np.minimum(d, 0.5)))
            if best is None or sc < best[0]: best = (sc, c, np.mean(d < 1.0), np.percentile(d, 95), float(d.max()))
        if __import__('os').environ.get('DBGA'):
            for c in (-198.25, -198.5):
                Mi = S.copy(); Mi[:, axis] = 2 * c - Mi[:, axis]; d, _ = T.query(Mi); print('DBGA axis', axis, c, 'median %.3f mean %.3f p95 %.3f' % (np.median(d), d.mean(), np.percentile(d, 95)))
        return best
    def mirror_pt(Q, axis, c):
        Q = np.array(Q, float); Q[..., axis] = 2 * c - Q[..., axis]; return Q
    def mirror_half(P, axis, c):
        n = len(P); side = P[:, axis] >= c; seq = []
        for i in range(n):
            a_, b_ = P[i], P[(i + 1) % n]; seq.append((a_, bool(side[i]), False))
            if side[i] != side[(i + 1) % n]:
                x = a_ + (b_ - a_) * ((c - a_[axis]) / (b_[axis] - a_[axis])); x[axis] = c; seq.append((x, None, True))
        ci = [k for k, q in enumerate(seq) if q[2]]
        if len(ci) < 2 or len(ci) % 2: return None, 'crossings %d' % len(ci)
        seq = seq[ci[0]:] + seq[:ci[0]]; ci = [k for k, q in enumerate(seq) if q[2]]; runs = []
        for k in range(len(ci)):
            i0, i1 = ci[k], ci[(k + 1) % len(ci)]; seg = seq[i0:i1 + 1] if i1 > i0 else seq[i0:] + seq[:1]
            pts = np.array([q[0] for q in seg]); sd = [q[1] for q in seg if q[1] is not None]
            runs.append(dict(pts=pts, side=sd[0] if sd else True))
        o = 1 - axis; Rr = [r for r in runs if r['side']]; Ll = [r for r in runs if not r['side']]
        if len(Rr) != len(Ll): return None, 'unbalanced runs'
        cost = np.zeros((len(Rr), len(Ll)))
        for i, r in enumerate(Rr):
            for j, l in enumerate(Ll):
                r0, r1, l0, l1 = r['pts'][0][o], r['pts'][-1][o], l['pts'][0][o], l['pts'][-1][o]
                cost[i, j] = min(abs(l0 - r0) + abs(l1 - r1), abs(l0 - r1) + abs(l1 - r0))
        ri, ci_ = linear_sum_assignment(cost)
        if cost[ri, ci_].max() > 2.0: return None, 'halves do not pair (cost %.1f)' % cost[ri, ci_].max()
        partner = {}
        for i, j in zip(ri, ci_): partner[id(Rr[i])] = Ll[j]; partner[id(Ll[j])] = Rr[i]
        nR = sum(len(r['pts']) for r in Rr); nL = sum(len(l['pts']) for l in Ll); keep_side = nR <= nL
        out = []; tags = []
        for r in runs:
            if r['side'] == keep_side:
                pts = r['pts'].copy(); tg = True
                Mp = mirror_pt(partner[id(r)]['pts'], axis, c); Dm = dense(np.r_[Mp, Mp[::-1][1:-1]], 0.05); Tm = cKDTree(Dm)
                Tv = cKDTree(Mp); p0s = pts.copy()
                for i_ in range(1, len(pts) - 1):
                    l1 = np.hypot(*(pts[i_] - pts[i_ - 1])); l2 = np.hypot(*(pts[i_ + 1] - pts[i_]))
                    dv_, jv = Tv.query(p0s[i_])
                    if dv_ < 1.0: pts[i_] = (p0s[i_] + Mp[jv]) / 2      # corner/vertex pairs: true averaging of both halves (no pushing everything to one side)
                    elif l1 <= 0.8 and l2 <= 0.8:
                        dd_, jj = Tm.query(pts[i_])
                        if dd_ < 1.5: pts[i_] = (pts[i_] + Dm[jj]) / 2
            else:
                pts = mirror_pt(partner[id(r)]['pts'], axis, c); tg = False
                if np.hypot(*(pts[0] - r['pts'][0])) > np.hypot(*(pts[-1] - r['pts'][0])): pts = pts[::-1]
            for q in pts: out.append(q); tags.append(tg)
        # merge duplicated axis points, prefer the kept side
        res = []
        for q, tg in zip(out, tags):
            if res and abs(q[axis] - c) < 1e-9 and abs(res[-1][0][axis] - c) < 1e-9 and np.hypot(*(q - res[-1][0])) < 0.6:
                if tg and not res[-1][1]: res[-1] = (q, tg)
                continue
            if res and np.hypot(*(q - res[-1][0])) < 1e-6: continue
            res.append((q, tg))
        if len(res) > 1 and abs(res[0][0][axis] - c) < 1e-9 and abs(res[-1][0][axis] - c) < 1e-9 and np.hypot(*(res[0][0] - res[-1][0])) < 0.6: res.pop()
        return np.array([q for q, _ in res]), ('kept %s half' % ('upper/right' if keep_side else 'lower/left'))
    # ---------- xline clean: short jogs/bumps/kinks between two long slanted straight edges -> intersection of those edges ----------
    def _sd(p, a, b):
        ab = b - a; t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1); return np.hypot(*(p - (a + t * ab)))
    def _axis(v):
        ang = abs(math.degrees(math.atan2(v[1], v[0]))) % 90; return min(ang, 90 - ang) < 2.0
    def xline_clean(P):
        P = [np.array(q, float) for q in P]; nfix = 0
        changed = True
        while changed and len(P) > 6:
            changed = False; n = len(P)
            for k in (1, 2, 3, 4, 5, 6):
                for s0 in range(n):
                    ia, s, t, ib = (s0 - 1) % n, s0, (s0 + k) % n, (s0 + k + 1) % n
                    A0, A1, B0, B1 = P[ia], P[s], P[t], P[ib]
                    la, lb = np.hypot(*(A1 - A0)), np.hypot(*(B1 - B0))
                    run = [P[(s0 + j) % n] for j in range(k + 1)]
                    rl = [np.hypot(*(run[j + 1] - run[j])) for j in range(k)]
                    small = (not os.environ.get('XSMALL')) is False and k <= 3 and la >= 5.5 and lb >= 5.5 and max(rl) <= 5.0 and sum(rl) <= 11.0
                    big = la >= 25 and lb >= 25 and sum(rl) <= 30.0
                    if not (small or big): continue
                    if _axis(A1 - A0) or _axis(B1 - B0): continue
                    da, db = (A1 - A0) / la, (B1 - B0) / lb
                    cr = da[0] * db[1] - da[1] * db[0]
                    if abs(math.degrees(math.asin(np.clip(cr, -1, 1)))) > 45 or da @ db < 0: continue
                    lim = 1.6 if small else 1.2
                    if abs(cr) < 0.14 and not small: continue
                    if abs(cr) < 0.14:  # near-collinear: put both ends on one line, drop the jog
                        Q = np.array([A0, A1, B0, B1]); c = Q.mean(0); u_ = np.linalg.svd(Q - c)[2][0]
                        pr = lambda q: c + ((q - c) @ u_) * u_
                        newv = [pr(A1), pr(B0)]
                    else:
                        tt = ((B0 - A0)[0] * db[1] - (B0 - A0)[1] * db[0]) / cr; X = A0 + tt * da
                        u = (X - B0) @ db
                        ex = 6.0 if small else sum(rl) + 3.0
                        if tt < la * 0.5 or tt > la + ex or u < -ex or u > lb * 0.6: continue
                        newv = [X]
                    path = [A1] + newv + [B1 if len(newv) == 1 else B0]
                    dev = max(min(_sd(q, path[j], path[j + 1]) for j in range(len(path) - 1)) for q in run)
                    if dev > lim: continue
                    if len(newv) == 2 and k == 1 and max(np.hypot(*(newv[0] - A1)), np.hypot(*(newv[1] - B0))) < 0.03: continue
                    idx = [(s0 + j) % n for j in range(k + 1)]
                    for j in idx[1:]: P[j] = None
                    P[idx[0]] = newv[0]
                    if len(newv) == 2: P[idx[1] if False else idx[0]] = newv[0]; P[idx[-1]] = newv[1]
                    P = [q for q in P if q is not None]; nfix += 1; changed = True; break
                if changed: break
        return np.array(P), nfix
    # ---------- jog refine: a short connector between two long straight edges gets its own XLINE fitted to the measured contour ----------
    def jog_refine(P, C):
        P = [np.array(q, float) for q in P]; n = len(P); tC = cKDTree(C); nC = len(C); nfix = 0; out_ = []; i = 0
        skip = set(); res = {}
        for s0 in range(n):
            for k in (1, 2):
                ia, t, ib = (s0 - 1) % n, (s0 + k) % n, (s0 + k + 1) % n
                A0, A1, B0, B1 = P[ia], P[s0], P[t], P[ib]
                la, lb = np.hypot(*(A1 - A0)), np.hypot(*(B1 - B0))
                run = [P[(s0 + j) % n] for j in range(k + 1)]
                rl = sum(np.hypot(*(run[j + 1] - run[j])) for j in range(k))
                if la < 4.5 or lb < 4.5 or rl > 3.5 or rl < 0.3: continue
                if _axis(A1 - A0) or _axis(B1 - B0): continue
                da, db = (A1 - A0) / la, (B1 - B0) / lb
                ca, cb = tC.query(A1)[1], tC.query(B0)[1]
                fwd = (cb - ca) % nC; bwd = (ca - cb) % nC
                if min(fwd, bwd) > 400: continue
                ids = [(ca + j) % nC for j in range(-30, fwd + 31)] if fwd <= bwd else [(ca - j) % nC for j in range(-30, bwd + 31)]
                R = C[ids]
                dA = np.abs((R - A1) @ np.array([-da[1], da[0]])); dB = np.abs((R - B0) @ np.array([-db[1], db[0]]))
                m = (dA > 0.12) & (dB > 0.12)
                # keep the contiguous middle part
                if m.sum() < 6: continue
                Rm = R[m]
                c = Rm.mean(0); u_ = np.linalg.svd(Rm - c)[2][0]
                if np.ptp((Rm - c) @ u_) < 0.45: continue
                cr1 = da[0] * u_[1] - da[1] * u_[0]; cr2 = u_[0] * db[1] - u_[1] * db[0]
                if abs(cr1) < 0.2 or abs(cr2) < 0.2: continue
                def inter(p0, d0, p1, d1):
                    cr = d0[0] * d1[1] - d0[1] * d1[0]
                    return p0 + d0 * (((p1 - p0)[0] * d1[1] - (p1 - p0)[1] * d1[0]) / cr)
                X1 = inter(A0, da, c, u_); X2 = inter(c, u_, B0, db)
                if np.hypot(*(X1 - A1)) > 1.6 or np.hypot(*(X2 - B0)) > 1.6: continue
                if (X2 - X1) @ u_ < 0: continue
                res[s0] = (k, [X1, X2]); break
        used = set(); newP = []
        for i in range(n):
            if i in used: continue
            if i in res:
                k, nv = res[i]; newP.extend(nv); used.update((i + j) % n for j in range(k + 1)); nfix += 1
            else: newP.append(P[i])
        return np.array(newP), nfix
    for fc in faces:
        Q, nf = jog_refine(fc['poly'], fc['C'])
        if nf: fc['poly'] = Q; say('face %d: %d jog connector(s) refit as their own straight lines' % (fc['id'], nf))
    for fc in faces:
        Q, nf = xline_clean(fc['poly'])
        if nf: fc['poly'] = Q; say('face %d: %d short jogs/kinks replaced by intersection of the neighbouring straight edges' % (fc['id'], nf))
    for fc in faces:
        P = fc['poly'].copy(); done = []
        if fc['id'] in KEPT_T: say('face %d: has a genuinely slanted short edge, so it is not mirrored' % fc['id']); continue
        for axis in (0, 1):
            sc, c, frac, p95, dmx = find_axis(P, axis)
            if os.environ.get('DBGDM'): print('DM face %d axis %d frac %.2f p95 %.2f dmax %.2f' % (fc['id'], axis, frac, p95, dmx))
            if not (frac >= 0.85 and sc < 0.4 and p95 < 1.0): done.append(('no %s-symmetry (%.0f%% match, p95 %.2f)' % ('x' if axis == 0 else 'y', 100 * frac, p95))); continue
            c = round(c / STEP) * STEP; Q, why = mirror_half(P, axis, c)
            if Q is None: done.append('%s-axis %.2f: not applied (%s)' % ('x' if axis == 0 else 'y', c, why)); continue
            P = Q; done.append('%s-axis at %.2f mm (%.0f%% matched, 95%% dist %.2f mm): %s' % ('x' if axis == 0 else 'y', c, 100 * frac, p95, why)); fc.setdefault('axes', []).append((axis, c))
        fc['poly'] = P; say('face %d symmetry: %s' % (fc['id'], '; '.join(done)))
    # ---------- fidelity pass: where an edge strays >FID mm from the measured contour, re-trace that stretch ----------
    FID = float(os.environ.get('FID', 99))
    for fc in faces:
        if fc.get('axes'): continue
        P = fc['poly']; C = fc['C']; tC = cKDTree(C); newP = []; nfix = 0; n = len(P)
        for i in range(n):
            a, b = P[i], P[(i + 1) % n]; newP.append(a)
            L = np.hypot(*(b - a))
            if L < 0.8: continue
            S = np.array([a + (b - a) * t for t in np.linspace(0, 1, max(int(L / 0.2), 2))])
            d, ci = tC.query(S)
            if d[2:-2].max() <= FID if len(d) > 5 else True: continue
            ia, ib = tC.query(a)[1], tC.query(b)[1]
            nC = len(C); fw = (ib - ia) % nC
            seg = np.array([C[(ia + k) % nC] for k in range(fw + 1)]) if fw < nC / 2 else None
            if seg is None or len(seg) < 4: continue
            if np.hypot(*(seg[0] - a)) > 0.6 or np.hypot(*(seg[-1] - b)) > 0.6: continue
            ap = cv2.approxPolyDP(seg.astype(np.float32).reshape(-1, 1, 2), 0.12, False).reshape(-1, 2)
            if len(ap) <= 2: continue
            newP.extend(list(ap[1:-1])); nfix += 1
        if nfix:
            fc['poly'] = np.array(newP); say('face %d: %d edges re-traced from the measured contour (deviation > %.2f mm)' % (fc['id'], nfix, FID))
    # ---------- holes: smooth, texture-free regions inside parts (through windows) ----------
    gf = g.astype(np.float32); hp = gf - cv2.GaussianBlur(gf, (0, 0), 2.0); en = np.sqrt(cv2.blur(hp * hp, (21, 21)))
    hm = (en < 1.4).astype(np.uint8); hm = cv2.morphologyEx(hm, cv2.MORPH_OPEN, np.ones((11, 11), np.uint8))
    n, hl, hst, hce = cv2.connectedComponentsWithStats(hm, connectivity=8)
    holes = []
    for i in range(1, n):
        x, y, w, h, a = hst[i]
        if a < 6 * SC ** 2 or w > 1200 or h > 1200: continue
        cx, cy = hce[i]
        fid = next((fc['id'] for fc in faces if cv2.pointPolygonTest(pxc[fc['id']].astype(np.float32), (float(cx), float(cy)), False) > 0), None)
        if fid is None: continue
        # true hole edge = outer edge of the grey area INCLUDING its shadow band, found from brightness profiles across each side
        ed_ = {}
        def prof(side):
            o = np.arange(-int(1.6 * SC), int(2.6 * SC))
            if side in ('L', 'R'):
                r0, r1 = int(y + 0.25 * h), int(y + 0.75 * h); xs = (x + 6 - o) if side == 'L' else (x + w - 7 + o)
                return o, np.median(gf[r0:r1][:, np.clip(xs, 0, gf.shape[1] - 1)], axis=0)
            c0, c1 = int(x + 0.25 * w), int(x + 0.75 * w); ys_ = (y + 6 - o) if side == 'T' else (y + h - 7 + o)
            return o, np.median(gf[np.clip(ys_, 0, gf.shape[0] - 1)][:, c0:c1], axis=1)
        for side in 'LRTB':
            o, pf = prof(side); pf = cv2.blur(pf.reshape(-1, 1), (1, 5)).ravel()
            Iin = np.median(pf[(o >= -int(1.4 * SC)) & (o <= -int(0.8 * SC))]); Wd = np.median(pf[o >= int(1.9 * SC)])
            win = (o >= -int(1.5 * SC)) & (o <= int(1.9 * SC)); m_ = pf[win].min()
            if m_ < Wd - 60:   # shadow band (visible far wall of the hole): caliper edge lies inside it, at BAND_F of its half-depth thickness
                idx_ = np.where(win & (pf < (Iin + m_) / 2))[0]
                ed_[side] = int(round(o[idx_.min()] + BAND_F * (o[idx_.max()] - o[idx_.min()]))) if len(idx_) else 6
                continue
            elif m_ < Wd - 25: sel = np.where((o >= -int(1.5 * SC)) & (o <= int(0.8 * SC)) & (pf < Wd - 20))[0]
            elif Iin < Wd - 12: sel = np.where(win & (pf < (Iin + Wd) / 2))[0]
            else: sel = []
            ed_[side] = int(o[sel.max()] + 1) if len(sel) else 6
        px_ = {'L': x + 6 - ed_['L'], 'R': x + w - 7 + ed_['R'], 'T': y + 6 - ed_['T'], 'B': y + h - 7 + ed_['B']}
        cn = to_mm(np.array([[px_['L'], px_['T']], [px_['R'], px_['T']], [px_['R'], px_['B']], [px_['L'], px_['B']]], float))
        x0, y0 = cn.min(0); x1, y1 = cn.max(0)
        holes.append(np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])); say('hole in face %d: %.2f x %.2f mm at (%.2f, %.2f)' % (fid, x1 - x0, y1 - y0, (x0 + x1) / 2, (y0 + y1) / 2))
    # ---------- tabs: places where the burnt line is missing ----------
    k_ = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    d = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k_).astype(np.int16) - g.astype(np.int16)
    dm = cv2.dilate(d.astype(np.uint8).clip(0, 255), np.ones((19, 19), np.uint8))
    tabs = []; longgaps = []
    for fc in faces:
        P = fc['poly']; closed = np.r_[P, P[:1]]; sl = np.r_[0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
        tt = np.arange(0, sl[-1], 0.1); S = np.c_[np.interp(tt, sl, closed[:, 0]), np.interp(tt, sl, closed[:, 1])]
        px = to_px(S); xi = np.clip(np.round(px[:, 0]).astype(int), 0, g.shape[1] - 1); yi = np.clip(np.round(px[:, 1]).astype(int), 0, g.shape[0] - 1)
        miss = dm[yi, xi] < 35; k = 0; nn = len(miss)
        while k < nn:
            if miss[k]:
                j = k
                while j + 1 < nn and miss[j + 1]: j += 1
                ln = (j - k + 1) * 0.1
                if ln >= 0.35: (tabs if ln <= 2.6 else longgaps).append((fc['id'], S[k], S[j], ln, S[(k + j) // 2]))
                k = j + 1
            else: k += 1
    for gp in A['gaps']:
        if not (0.35 <= gp['mm'] <= 2.6): continue
        a_, b_ = to_mm(np.array([gp['p0'][0], gp['p0'][1]], float))[0], to_mm(np.array([gp['p1'][0], gp['p1'][1]], float))[0]; mid = (a_ + b_) / 2
        if any(np.hypot(*(mid - t[4])) < 2.0 for t in tabs): continue
        best = None
        for fc in faces:
            dd = np.hypot(*(fc['poly'] - mid).T).min()
            if best is None or dd < best[0]: best = (dd, fc['id'])
        if best is None or best[0] > 1.0: continue   # a tab must sit on a part contour
        tabs.append((best[1], a_, b_, float(gp['mm']), mid))
    # limit: a few tabs per part (enough to hold it, not a perforation): keep up to 4 per part, spread along the contour, prefer 0.8-1.6 mm
    keep = []
    for fid in sorted(set(t[0] for t in tabs)):
        ts = sorted([t for t in tabs if t[0] == fid], key=lambda t: abs(t[3] - 1.2))
        ch = []
        for t in ts:
            if all(np.hypot(*(t[4] - c[4])) > 12.0 for c in ch): ch.append(t)
            if len(ch) >= 4: break
        keep += ch
    tabs = keep
    say('tabs (uncut bridges 0.35-2.6 mm): %d, widths mm: %s' % (len(tabs), np.round(sorted(t[3] for t in tabs), 2).tolist()))
    say('longer unlit stretches (>2.6 mm, for review): %d %s' % (len(longgaps), [(t[0], np.round(t[4], 1).tolist(), round(t[3], 1)) for t in longgaps]))
    # ---------- DXF ----------
    R = lambda P: np.round(np.asarray(P) / STEP) * STEP
    def dedup(p):
        o = [p[0]]
        for q in p[1:]:
            if np.hypot(*(q - o[-1])) > 0.04: o.append(q)
        if np.hypot(*(o[0] - o[-1])) < 0.04: o.pop()
        return np.array(o)
    def dxf(path, layers, ents):
        o = ['0', 'SECTION', '2', 'HEADER', '9', '$ACADVER', '1', 'AC1009', '9', '$INSUNITS', '70', '4', '0', 'ENDSEC', '0', 'SECTION', '2', 'TABLES', '0', 'TABLE', '2', 'LTYPE', '70', '1', '0', 'LTYPE', '2', 'CONTINUOUS', '70', '0', '3', 'Solid line', '72', '65', '73', '0', '40', '0.0', '0', 'ENDTAB', '0', 'TABLE', '2', 'LAYER', '70', str(len(layers))]
        for nme, col in layers: o += ['0', 'LAYER', '2', nme, '70', '0', '62', str(col), '6', 'CONTINUOUS']
        o += ['0', 'ENDTAB', '0', 'ENDSEC', '0', 'SECTION', '2', 'ENTITIES']
        for layer, pts, closed in ents:
            o += ['0', 'POLYLINE', '8', layer, '66', '1', '10', '0.0', '20', '0.0', '30', '0.0', '70', '1' if closed else '0']
            for q in pts: o += ['0', 'VERTEX', '8', layer, '10', '%.3f' % q[0], '20', '%.3f' % q[1], '30', '0.0']
            o += ['0', 'SEQEND', '8', layer]
        o += ['0', 'ENDSEC', '0', 'EOF']; open(path, 'w').write('\n'.join(o) + '\n')
    ents = []
    for fc in faces: ents.append(('CUT', dedup(R(fc['poly'])), True))
    for h in holes: ents.append(('HOLES', R(h), True))
    for t in tabs:
        a, b = R(t[1]), R(t[2])
        if np.hypot(*(a - b)) < 0.05: b = a + [0.05, 0]
        ents.append(('TABS', np.array([a, b]), False))
    for cc in (to_mm(A['a'])[0], to_mm(A['b'])[0]):
        ents += [('REF', R(np.array([cc + [-2.5, 0], cc + [2.5, 0]])), False), ('REF', R(np.array([cc + [0, -2.5], cc + [0, 2.5]])), False)]
    dxf(out + '/' + prefix + '.dxf', [('CUT', 7), ('HOLES', 3), ('TABS', 1), ('REF', 5)], ents)
    say('DXF written: %d closed CUT contours, %d holes, %d tabs' % (len(faces), len(holes), len(tabs)))
    # ---------- verification vs. skeleton ----------
    sk = A['bridged']; ys, xs = np.nonzero(sk); tree = cKDTree(np.c_[xs, ys + Y0])
    for fc in faces:
        P = R(fc['poly']); closed = np.r_[P, P[:1]]; sl = np.r_[0, np.cumsum(np.hypot(*np.diff(closed, axis=0).T))]
        tt = np.arange(0, sl[-1], 0.2); S = np.c_[np.interp(tt, sl, closed[:, 0]), np.interp(tt, sl, closed[:, 1])]
        dd, _ = tree.query(to_px(S)); dd = dd / SC
        say('face %d: perimeter %.0f mm, deviation from scan line: median %.2f, 95%% %.2f, max %.2f mm' % (fc['id'], sl[-1], np.median(dd), pct(dd, 95), dd.max()))
    pickle.dump(dict(faces=faces, holes=holes, tabs=tabs, longgaps=longgaps, M=M, X0=X0, SC=SC), open(out + '/B.pkl', 'wb'))
    open(out + '/' + prefix + '_report.txt', 'w').write('\n'.join(rep))
    # ---------- overlay ----------
    f3 = 5; sm = cv2.cvtColor(cv2.resize(g, (g.shape[1] // f3, g.shape[0] // f3), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    big = cv2.resize(sm, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR); s = 2.0 / f3
    for fc in faces: cv2.polylines(big, [np.round(to_px(R(fc['poly'])) * s).astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
    for h in holes: cv2.polylines(big, [np.round(to_px(h) * s).astype(np.int32)], True, (255, 0, 0), 1, cv2.LINE_AA)
    for t in tabs: cv2.circle(big, tuple(np.round(to_px(t[4])[0] * s).astype(int)), 6, (0, 160, 0), 1, cv2.LINE_AA)
    for t in longgaps: cv2.circle(big, tuple(np.round(to_px(t[4])[0] * s).astype(int)), 9, (255, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out + '/' + prefix + '_overlay.png', big)
    for zi, z_ in enumerate([q for q in __import__('os').environ.get('ZOOMS', '').split(';') if q]):
        zx, zy = map(float, z_.split(',')); cx, cy = np.round(to_px(np.array([zx, zy]))[0]).astype(int); hw = int(float(__import__("os").environ.get("ZW", 4.5)) * SC)
        crop = cv2.cvtColor(g[max(cy - hw, 0):cy + hw, max(cx - hw, 0):cx + hw], cv2.COLOR_GRAY2BGR); crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        for fc in faces:
            pp = (to_px(R(fc['poly'])) - [max(cx - hw, 0), max(cy - hw, 0)]) * 3
            cv2.polylines(crop, [np.round(pp).astype(np.int32)], True, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.imwrite(out + '/zoom%d.png' % zi, crop)



def convert(scan, outdir, name=None, ruler_mm=100.0, kerf=0.0, t_lo=None, t_hi=None, overlay=True, debug=False):
    """scan image -> outdir/NAME.dxf (+ NAME_overlay.png). Returns the DXF path."""
    global rep
    rep = []
    if t_lo is not None: os.environ['T_LO'] = str(t_lo)
    if t_hi is not None: os.environ['T_HI'] = str(t_hi)
    name = name or os.path.splitext(os.path.basename(scan))[0]
    os.makedirs(outdir, exist_ok=True)
    work = os.path.join(outdir, '_work') if debug else tempfile.mkdtemp(prefix='s2d_')
    os.makedirs(work, exist_ok=True)
    try:
        run(scan, work, ruler_mm)
        main(scan, work, name, kerf)
        shutil.copy(os.path.join(work, name + '.dxf'), os.path.join(outdir, name + '.dxf'))
        if overlay: shutil.copy(os.path.join(work, name + '_overlay.png'), os.path.join(outdir, name + '_overlay.png'))
        shutil.copy(os.path.join(work, name + '_report.txt'), os.path.join(outdir, name + '_report.txt'))
    finally:
        if not debug: shutil.rmtree(work, ignore_errors=True)
    return os.path.join(outdir, name + '.dxf')

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='scan of laser-cut templates -> DXF')
    ap.add_argument('scan'); ap.add_argument('-o', '--outdir', default='.'); ap.add_argument('--name')
    ap.add_argument('--ruler-mm', type=float, default=100.0, help='distance between the two reference crosses, mm')
    ap.add_argument('--kerf', type=float, default=0.0, help='extra inward offset of the contour, mm')
    ap.add_argument('--t-lo', type=int); ap.add_argument('--t-hi', type=int)
    ap.add_argument('--no-overlay', action='store_true'); ap.add_argument('--debug', action='store_true')
    a = ap.parse_args()
    print('DXF:', convert(a.scan, a.outdir, a.name, a.ruler_mm, a.kerf, a.t_lo, a.t_hi, not a.no_overlay, a.debug))
