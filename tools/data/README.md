# Data the figure generators read

`ne_110m_land.geojson` is the land polygon layer of Natural Earth at 1:110 million,
exactly as distributed: fetched on 2026-10-02 from the `master` branch of
<https://github.com/nvkelso/natural-earth-vector> (`geojson/ne_110m_land.geojson`),
whose `VERSION` file then read `5.2.0-pre`. Natural Earth is in the public domain:
<https://www.naturalearthdata.com/about/terms-of-use/>.

`tools/gen_proposal_figures.py` draws the coastlines of the proposal's
reachable-set figure from it. It is kept here, and not fetched, so that the
figure regenerates without a network.
