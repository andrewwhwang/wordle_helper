# Wordle Helper
A quick script that'll help you win at wordle

Given an input of guesses it'll output:
* a list of possible answer ordered by confidence
* a list of words ordered by how much it'll narrow down the possible answers
## Usage Example
Given the guesses:
```python
guesses = [
    "niter",
    "coals",
    "bongo",
    ]
patterns = [
    "🟨⬛⬛⬛⬛",
    "⬛🟩⬛⬛🟩",
    "🟩🟩🟩⬛⬛",
    ]

```
Output:
```
Answer Confidence (Total Entropy: 1.001 bits):
         bonds 50.01%
         bonus 49.98%
         bonks 0.01%
Expected Entropy Reduction:
         zouks 1.001 bits
         yukos 1.001 bits
         yukky 1.001 bits
         yukes 1.001 bits
         yuked 1.001 bits
```
