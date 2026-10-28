# 7. Take your data with you

{bdg-success}`What you get`

Type one of these in the project folder. Each one prints where it saved its files.

`macaw export`
: One table (CSV) with every measure, for every person, over time. Add `--rmd` for an R Markdown file that loads it.

`macaw export -f sequence`
: Who is talking at each second, per person, for sequence analysis.

`macaw export -f eaf`
: One ELAN file per recording with the machine codes, as a starting point for your own coding.

`macaw methods`
: A Methods section describing exactly the steps you ran, with its references.

Every run keeps its settings and the software versions it used, so anyone can rebuild the same table later.
