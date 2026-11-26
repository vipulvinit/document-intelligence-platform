"""Starter literature sources, created on first start when SEED_DEFAULT_SOURCES=True.

Each is a Europe PMC query (syntax: https://europepmc.org/searchsyntax). Add your own
candidates with POST /sources; disable these with PATCH /sources/{id} {"enabled": false}.

Tip: search titles and abstracts (TITLE_ABS:) rather than full text. A full-text search for
"malaria treatment" also matches articles that mention malaria once in passing (206 hits in a
week, versus 10 relevant ones with TITLE_ABS when this was written).
"""

DEFAULT_SOURCES = [
    {
        "name": "Clinical practice guidelines",
        "query": 'PUB_TYPE:"Practice Guideline" AND TITLE_ABS:(treatment OR management)',
        "poll_interval_hours": 24,
    },
    {
        "name": "Emerging infectious diseases",
        "query": 'TITLE_ABS:("emerging infectious disease" OR outbreak OR "novel virus" OR "novel pathogen") '
        "AND TITLE_ABS:(clinical OR treatment OR management)",
        "poll_interval_hours": 24,
    },
    {
        "name": "Outbreak preprints",
        "query": 'SRC:PPR AND TITLE_ABS:(outbreak OR "novel pathogen" OR "emerging virus")',
        "poll_interval_hours": 24,
    },
    {
        "name": "Tropical diseases treatment",
        "query": "TITLE_ABS:(dengue OR malaria OR chikungunya OR typhoid OR leptospirosis OR tuberculosis) "
        "AND TITLE_ABS:(treatment OR management) AND OPEN_ACCESS:y",
        "poll_interval_hours": 24,
    },
]
