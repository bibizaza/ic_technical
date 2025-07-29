"""YTD performance utilities package.

This package provides functions to compute year‑to‑date (YTD) performance
series for various asset classes (equities, commodities, cryptocurrencies)
and helpers to insert charts and subtitles into PowerPoint presentations.

Each module implements a ``get_*_ytd_series`` function to compute the
percentage change from the start of the current year, a ``create_*_chart``
function to build a matplotlib figure with annotations, and an
``insert_*_chart`` function to insert the chart, subtitle and data source
footnote into a slide identified by placeholders.

Modules:

* ``equity_ytd`` – YTD performance for equity indices.
* ``commodity_ytd`` – YTD performance for commodities (to be implemented).
* ``crypto_ytd`` – YTD performance for cryptocurrencies (to be implemented).

"""