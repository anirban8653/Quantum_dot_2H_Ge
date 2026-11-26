import re, sys, numpy as np
src, out = sys.argv[1], sys.argv[2]
s = open(src).read()
TR = 'transform="matrix(0.998185, 0, 0, 1, 0, 0)"'
BLUE = '#2166AC'

def arrow(p, q, lw=1.1, head=5.0, w=2.2, both=True, col='black'):
    p, q = np.array(p, float), np.array(q, float)
    u = (q - p) / np.linalg.norm(q - p); n = np.array([-u[1], u[0]])
    def tri(t, d):
        b = t - d * head
        return f'M {t[0]:.3f} {t[1]:.3f} L {b[0]+n[0]*w:.3f} {b[1]+n[1]*w:.3f} L {b[0]-n[0]*w:.3f} {b[1]-n[1]*w:.3f} Z'
    a = p + u * head * 0.6 if both else p; b = q - u * head * 0.6
    o = f'<path fill="none" stroke="{col}" stroke-width="{lw}" stroke-linecap="round" d="M {a[0]:.3f} {a[1]:.3f} L {b[0]:.3f} {b[1]:.3f}"/>'
    o += f'<path fill="{col}" stroke="none" d="{tri(q, u)}"/>'
    if both: o += f'<path fill="{col}" stroke="none" d="{tri(p, -u)}"/>'
    return o

# ---------- defs: glow gradient + clip for (b) potential ----------
r = np.linspace(0, 1, 11); g = np.exp(-(r / 0.5) ** 2)
stops = ''.join(f'<stop offset="{ri:.1f}" stop-color="{BLUE}" stop-opacity="{0.9*gi:.3f}"/>' for ri, gi in zip(r, g))
defs = (f'<radialGradient id="glow" cx="0.5" cy="0.5" r="0.5">{stops}</radialGradient>'
        '<clipPath id="clipV" clipPathUnits="userSpaceOnUse"><rect x="300" y="0" width="200" height="254"/></clipPath>')
s = s.replace('<defs>', '<defs>' + defs, 1)

# ---------- (a) push right gate back along the wire to open the dot region ----------
DX, DY = 22.0, -22.0*27.93/54.81
SX = DX*0.998185
def shift_in(s, key, pts, sx):
    i = s.index(key); j = s.index('/>', i)
    el = s[i:j]
    for p in pts:
        x, y = map(float, p.split())
        el = el.replace(p, f'{x+sx:.6f} {y+DY:.6f}')
    return s[:i] + el + s[j:]
# back segment (raw coords inside transformed paths -> shift DX; untransformed -> SX)
s = shift_in(s, 'd="M 246.525569 100.945312 L 172.332314 138.75', ['172.332314 138.75'], DX)
s = shift_in(s, 'd="M 207.257812 138.75 L 281.320312', ['207.257812 138.75', '207.257812 103.445312'], SX)
s = shift_in(s, 'd="M 172.019531 103.445312 L 246.078125', ['172.019531 103.445312', '207.257812 103.445312'], SX)
s = shift_in(s, 'd="M 246.525569 65.640625 L 172.332314', ['172.332314 103.445312'], DX)
s = shift_in(s, 'd="M 281.831837 65.640625 L 207.634669', ['207.634669 103.445312'], DX)
s = shift_in(s, 'd="M 281.831837 100.945312 L 207.634669', ['207.634669 138.75'], DX)
# middle segment
s = shift_in(s, 'd="M 164.309941 142.835938 L 109.499523', ['164.309941 142.835938'], DX)
s = shift_in(s, 'd="M 164.309941 107.535156 L 109.499523', ['164.309941 107.535156'], DX)
s = shift_in(s, 'd="M 199.61621 107.535156 L 144.801878', ['199.61621 107.535156'], DX)
s = shift_in(s, 'd="M 199.61621 142.835938 L 144.801878', ['199.61621 142.835938'], DX)
s = shift_in(s, 'd="M 144.539062 170.765625 L 199.253906', ['199.253906 142.835938', '199.253906 107.535156'], SX)
s = shift_in(s, 'd="M 109.300781 135.460938 L 164.011719', ['164.011719 107.535156', '199.253906 107.535156'], SX)
# right gate (5 paths) -> translate group
a = s.rindex('<path', 0, s.index('d="M 204.910976 146.367188'))
b = s.index('/>', s.index('d="M 164.309941 107.535156 L 199.61621 107.535156')) + 2
s = s[:a] + f'<g transform="translate({SX:.4f},{DY:.4f})">' + s[a:b] + '</g>' + s[b:]

# ---------- (a) remove red arc, add hole cloud ----------
arc = re.search(r'<path fill="none" stroke-width="1.90418"[^>]*d="M 196.790769 133.898438[^>]*/>', s).group(0)
c0, c1 = np.array([181.96+DX, 125.19+DY]), np.array([127.15, 153.11])   # wire centre at right/left gate faces
mid = (c0 + c1) / 2; ax = (c0 - c1) / np.linalg.norm(c0 - c1)
ang = np.degrees(np.arctan2(ax[1], ax[0]))
L = 13.0          # 1/e half-width of |psi0|^2 in drawing units  (= l_dot)
RX, RY = 2.1 * L, 15
cloud = (f'<g {TR}><ellipse cx="0" cy="0" rx="{RX}" ry="{RY}" fill="url(#glow)" '
         f'transform="translate({mid[0]:.3f},{mid[1]:.3f}) rotate({ang:.3f})"/></g>')
s = s.replace(arc, cloud)

# ---------- (a) l_dot arrow: span the cloud (+-l_dot) ----------
old = re.search(r'<path fill="none" stroke-width="1.10242"[^>]*d="M 128.248396 121.839844[^>]*/>\s*'
                r'<path[^>]*d="M 133.17922 121.800781[^>]*/>\s*<path[^>]*d="M 175.932599 95.066406[^>]*/>', s).group(0)
up = np.array([0, -30.75])
pa, pb = mid + up - L * ax, mid + up + L * ax
guides = ''.join(f'<path fill="none" stroke="#555" stroke-width="0.5" stroke-dasharray="1.5,1.5" '
                 f'd="M {p[0]:.3f} {p[1]+2:.3f} L {p[0]:.3f} {p[1]-up[1]:.3f}"/>' for p in (pa, pb))
s = s.replace(old, f'<g {TR}>{guides}{arrow(pa, pb)}</g>')
ELL_X, DOT_DX = 134.6, 0.8
# '2' in front of l_dot, move label with the arrow
lab_keys = ['d="M 129.', 'd="M 134.', 'd="M 141.', 'd="M 147.']
idx = []
for m in re.finditer(r'<path fill-rule="nonzero" fill="rgb\(0%, 0%, 0%\)" fill-opacity="1" d="([^"]*)"/>', s):
    nums = list(map(float, re.findall(r'-?\d+\.?\d*', m.group(1))))
    xs, ys = nums[0::2], nums[1::2]
    if 129 < min(xs) and max(xs) < 152 and 87 < min(ys) and max(ys) < 103: idx.append((m.start(), m.end()))
assert len(idx) == 4, idx
a, b = idx[0][0], idx[-1][1]
LX, LY = (pa[0]+pb[0])/2 - 154.55 + 0, (pa[1]+pb[1])/2 - 108.4
lab = s[idx[1][0]:b]          # keep 'dot' glyphs, drop the italic 'l' glyph
s = s[:a] + (f'<g transform="translate({LX-6:.3f},{LY-1:.3f})">'
             f'<text x="{ELL_X}" y="99.2" font-family="DejaVu Serif" font-size="15" text-anchor="end">2\u2113</text>'
             f'<g transform="translate({DOT_DX},0)">' + lab + '</g></g>') + s[b:]

# ---------- (b) potential: dark grey, clipped at x-axis ----------
s = re.sub(r'(<path fill="none" stroke-width="1.80396"[^>]*?)stroke="rgb\(69.802856%, 9.411621%, 16.862488%\)"',
           r'\1stroke="rgb(20%, 20%, 20%)" clip-path="url(#clipV)"', s, count=1)

# ---------- (b) ground-state |psi0|^2 on the top level, axes ----------
xc, y0 = 390.14, 72.137
lb = (424.77 - 357.02) / 2 / 2      # half-width of level at mean height ~ turning point
lb = (423.41 - 358.35) / 2          # classical turning points of n=0 = +-l_dot
xs = np.linspace(xc - 1.55 * lb, xc + 1.55 * lb, 120); h = 24
ys = y0 - h * np.exp(-((xs - xc) / lb) ** 2)
dpsi = 'M ' + ' L '.join(f'{x:.3f} {y:.3f}' for x, y in zip(xs, ys))
wf = (f'<path fill="{BLUE}" fill-opacity="0.25" stroke="none" d="{dpsi} Z"/>'
      f'<path fill="none" stroke="{BLUE}" stroke-width="1.1" d="{dpsi}"/>')
axes = arrow((308, 256), (308, 44), lw=1.0, both=False) + arrow((304, 254), (462, 254), lw=1.0, both=False)
s = s.replace('</svg>', f'<g {TR}>{wf}{axes}</g>'
              '<text x="313" y="52" font-family="DejaVu Serif" font-style="italic" font-size="14">E</text>'
              '<text x="461" y="266" font-family="DejaVu Serif" font-style="italic" font-size="14">x</text></svg>')

# ---------- canvas ----------
s = s.replace('width="465" height="256" viewBox="0 0 465 256"', 'width="472" height="270" viewBox="0 0 472 270"', 1)
open(out, 'w').write(s)
