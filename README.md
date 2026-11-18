# GeoStreamFusion — project site

Static website for the GeoStreamFusion M.Tech project (IIT Tirupati, 2026).

| Path | What it is |
|---|---|
| `index.html` | Landing page |
| `architecture/` | Animated architecture walkthrough (single self-contained HTML file) |
| `viewer/` | Side-by-side 3D viewer: our reconstructions vs the native-4K reference, plus the input video |
| `poster/` | A1 poster (PDF and LaTeX source with figures) |

## Hosting with GitHub Pages

1. On GitHub, open **Settings → Pages**.
2. Under **Build and deployment**, set **Source** to *Deploy from a branch*, branch `main`, folder `/ (root)`, and save.
3. After a minute the site is live at `https://<user>.github.io/3D_reconstruction/`.

The site has no build step. `.nojekyll` makes GitHub serve every file as is.

## Notes

- The 3D models are reduced web copies (about 20–35 MB each). The full-resolution combined model (112 MB) is left out because GitHub rejects files over 100 MB.
- All timings were measured on one GPU shared with another training job.
- F-scores are measured against the native-4K reconstruction at 1% of scene size, in the same camera frame.

Venkatasai Eswar Nandu · Atul Tripathi · M Rajvardhan · Atri Sukul (CS25M206) · Supervisor: Dr. Chalavadi Vishnu

## Progressive showcase update — 17 November 2026

The landing page now contains the full project showcase and automatic progressive 3D loading. The previous landing page is preserved as `legacy-index.html`; `architecture/`, `viewer/` and `poster/` remain available. The revised poster linked by the new landing page is `assets/poster.pdf`, with a QR for this repository’s Pages address.

The campus model loads in 16 spatial pieces with four simultaneous requests and a shared preview/full-resolution texture. All 682,950 original web-mesh triangles and texture coordinates were verified unchanged. In a local simulated 8 Mbps / 60 ms connection, the first piece appeared at 6.3 s and loading completed at 27.3 s; production timing will vary. Failed transfers retry only missing pieces. The complete payload remains about 25 MB.

Expected URL after enabling Pages: https://cs25m213-gif.github.io/3D_reconstruction/

The primary showcase links `methods.html` for evaluation scopes. Historical comparison pages use their own protocols and are preserved separately.

## Architecture explanation and supplied walkthrough

The main showcase and `assets/poster.pdf` now show separate ADD, SWAP and KEEP gain equations with candidate/occupant definitions. Blue identifies camera geometry, maroon the learned selector, amber depth estimation, green surface fusion and purple mesh output. The same legend appears in `walkthrough/`, imported from the supplied GeoStreamFusion Walkthrough ZIP. The main architecture section embeds this animation and links to its full-page view. The earlier `architecture/` animation remains available.

SEVA Lab logo added to the poster and site header/footer. Source: https://sevalab-iitt.github.io/logo.png (matching the supplied logo).
