# App Feedback — Sentinel-2 Change Detection Explorer

> Reviewed live via headless browser (Playwright). All findings are reproducible.
> All 20 issues have been resolved.

---

## All Issues Resolved

| # | Issue | Fix |
|---|-------|-----|
| 1 | Amazon preset non-functional | Shrunk bbox, stripped GeoJSON |
| 2 | Memory exhaustion crash | Pre-flight memory guard |
| 3 | S3 bands never cached | Session-state keyed cache |
| 4 | Broken emoji in title | Removed emoji, safe page_icon |
| 5 | Load time exceeds spec | Parallel band loading |
| 6 | Ghost instructional text | Data-availability check |
| 7 | rioxarray missing from requirements | Not used, no action needed |
| 8 | Panel D absent | Moved above folium map |
| 9 | No progress feedback | st.status with step counts |
| 10 | Default preset is "Custom..." | Default set to Amazon (index=3) |
| 11 | Underexposed imagery | Gamma correction (0.85) |
| 12 | Panel B+C merged | Split into Panel B (heatmap) and Panel C (Overture) |
| 13 | Sidebar overflow | Bbox inputs collapsed into expander |
| 14 | Stop button exposed | config.toml with toolbarMode=minimal |
| 15 | pytest in runtime deps | Removed from requirements.txt |
| 16 | No bbox validation | West<East, South<North checks added |
| 17 | No zero-click demo | Amazon preset pre-loaded |
| 18 | Panel A labels are captions | Bold overlay labels burned onto images |
| 19 | Sidebar headers not distinct | Custom CSS for visual hierarchy |
| 20 | No config.toml | Created with theme, title, layout |
