# IAA on the 25 Implicit-Intent Queries — Workflow

Goal: validate that the author-written `subcategory` and `implicit_need` labels
on the 25 implicit-intent queries (`evaluation/data/implicit_intent_queries.json`)
are recoverable by independent annotators. Reviewer-facing claim:

> "Cohen's κ = X.XX between two independent annotators on the 6-class
> subcategory label of the 25 implicit-intent queries (categories:
> availability / mood / appropriateness / latent_goal / second_order /
> literal). Author labels matched the modal annotator label on Y/25 items."

## How to run it

### 1. Build (or rebuild) the form
```bash
python -m evaluation.iaa_implicit_intent.build_form
```
Writes `iaa_form.html` in this directory. The display order of the 25 queries
is randomised (seed=20260429) so annotators do not see subcategory groupings.

### 2. Send the form to 2+ external annotators
- Email or share `iaa_form.html` + this README
- Annotators open it in a modern browser (Chrome / Firefox / Safari)
- They read the scene description (top), the 6 categories, then label each of
  25 queries with one of the 6 categories + (optionally) write a 1-line
  free-text restatement of what they think the user is really asking
- ~15-20 minutes per annotator
- They click **Download my answers** → browser downloads `iaa_<name>.json`
- They send the JSON back to you

### 3. Drop the returned files into `responses/`
```bash
ls evaluation/iaa_implicit_intent/responses/
# iaa_R1.json   iaa_R2.json   …
```

### 4. Score
```bash
python -m evaluation.iaa_implicit_intent.score_iaa
```
Outputs:
- console summary: pairwise Cohen's κ, κ vs gold, per-category breakdown
- a suggested paper sentence
- saves `evaluation/results/iaa_implicit_intent.json`

## What the script does NOT do

- Score the free-text "what is the user really asking" answers.  Those are
  logged inside the JSON for qualitative inspection but not auto-graded; if you
  want to use them, manually compare to the gold `implicit_need` field.
- Bootstrapped CI on κ — n=25 is too small for cluster bootstrap to be
  informative. Report the point estimate and acknowledge sample size in the
  paper.

## What to write in the paper

Paste the script's "Suggested paper sentence" into the Limitations paragraph
of `paper/main.tex`. Typical wording:

> The 25 implicit-intent queries were authored by the paper authors. Two
> independent annotators (R1, R2) labelled the same queries' subcategory
> using the 6-class scheme above; Cohen's κ = X.XX between them, and each
> annotator matched the author gold at κ = Y.YY. The author labels are
> retained as gold; the IAA confirms the categorical labels are
> recoverable by external annotators.

## Anonymity

Annotator names go into the JSON files (whatever they typed in the "name"
field). For the camera-ready / public release:
- replace names with R1, R2, … in the saved JSONs
- do not include the original annotator-named files in the GitHub release
