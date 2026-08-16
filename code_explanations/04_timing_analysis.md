# `04_timing_analysis.py`

This descriptive script reconstructs episode-level intervals from the private
REDCap workbook:

- presentation to final recorded review;
- presentation to initial intervention (culture-collection proxy);
- presentation to first intravitreal treatment; and
- presentation to discharge.

Negative follow-up intervals and intervals longer than 10 years are excluded as
implausible, matching the preprocessing rule. Estimated culture-result timing is
calculated only where an intervention date exists; missing intervention dates
are not treated as same-day collection.

The episode-level timing table is derived patient data and is git-ignored.
