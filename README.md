# selenium-wordle

Plays the daily [New York Times Wordle](https://www.nytimes.com/games/wordle/index.html) in Chrome using Selenium, choosing each guess with a minimax search.


## How It Works

`minimaxwordle.py` opens the NYT Wordle page, clicks **Play**, gets past the pre-game ad ("Continue to Wordle") and the How To Play popup, then types its guesses and reads the colored tiles back.

It uses two word lists:

- `words.txt`: the 2,315 original Wordle answers (plus `soare`). Guesses are chosen against this list first.
- `allowed.txt`: all 14,855 words NYT accepts as guesses. NYT now sometimes picks answers that aren't in `words.txt` (52 so far in 2026), so the bot also keeps track of these.

How each guess is chosen:

1. The first guess is always `soare`.
2. **Minimax:** guess the `words.txt` word that leaves the fewest possible answers in the worst case. If several are equally good, take the one that best narrows down all NYT words that still fit.
3. **Probes:** up to twice, guess a word that can't be the answer when it narrows things down better than guessing one that can.
4. **Last 2–3 guesses:** if more NYT words fit than `words.txt` words, guess the NYT word that splits them into the most groups, but only if every remaining `words.txt` word can still be solved in time.
5. **Fallback:** if no `words.txt` word fits, switch to `allowed.txt` and guess the word that splits the remaining words into the most groups.

### Results

Measured by playing every answer offline with the bot's logic, up to puzzle #1981 (2026-11-21):

| Answers | Solved within 6 | Average guesses |
|---|---|---|
| `words.txt` (2,315) | 100% | 3.73 |
| Every NYT puzzle so far (1,982) | 100% | 3.75 |
| Other words NYT accepts (12,539) | 99.0% | 4.54 |


## How To Start

Make a virtual environment and install the packages in `requirements.txt`:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`batch_test.py` also needs PyTorch and NumPy:

```bash
pip install torch numpy
```


## minimaxwordle.py

Plays today's puzzle on https://www.nytimes.com/games/wordle/index.html. Run it from the repo folder, since the word lists are loaded from the current directory:

```bash
python minimaxwordle.py
```

A Chrome window opens and plays, and each guess is printed in the terminal. When it finishes, press Enter in the terminal to close the browser.

You don't need to install a driver: Selenium finds Chrome on its own, and downloads Chrome for Testing if Chrome isn't installed.


## batch_test.py

Plays every possible answer offline with the same logic as `minimaxwordle.py` and reports how many guesses each one took, in two groups:

- `words.txt` answers: should all be solved within 6 (prints `pass`)
- every other word NYT accepts: the cases where the answer isn't in `words.txt`

```bash
python batch_test.py
```

To test a different starting word, change `first_guess` at the top of the file (`minimaxwordle.py` always opens with `soare`).

It uses a CUDA GPU if one is available and the CPU otherwise (about 3–4 minutes on a 16-core CPU). It builds a 14,855 × 14,855 feedback table, about 440 MB.
