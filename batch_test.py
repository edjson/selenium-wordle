import sys
import os
import numpy as np
import torch
from tqdm import tqdm
from collections import Counter


# Plays every possible answer with the same logic as minimaxwordle.py and reports
# how many guesses each one takes:
#   - words.txt answers: should all be solved in 6
#   - every other word NYT accepts (allowed.txt): the case where today's answer
#     isn't in words.txt and the bot has to fall back
first_guess = "soare"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("CUDA not available, falling back to CPU")

def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)

with open(resource_path("words.txt"), "r") as file:
    answers = [line.strip().lower() for line in file if len(line.strip()) == 5]
with open(resource_path("allowed.txt"), "r") as file:
    raw = list(dict.fromkeys(line.strip().lower() for line in file if len(line.strip()) == 5))
raw += [w for w in answers if w not in set(raw)]

N = len(raw)
word_to_idx = {w: i for i, w in enumerate(raw)}
wordlist_np = np.array([[ord(c) - ord('a') for c in w] for w in raw], dtype=np.int8)
wordlist = torch.tensor(wordlist_np, dtype=torch.int8, device=DEVICE)

def make_table():
    POWERS = torch.tensor([243, 81, 27, 9, 3], dtype=torch.int32, device=DEVICE)
    CHUNK = 256
    table = torch.zeros((N, N), dtype=torch.int16, device=DEVICE)
    guesses = wordlist
    same   = guesses[:, :, None] == guesses[:, None, :]
    before = torch.tril(torch.ones(5, 5, dtype=torch.bool, device=DEVICE), diagonal=-1)
    for start in tqdm(range(0, N, CHUNK), desc="Building table"):
        end = min(start + CHUNK, N)
        secrets = wordlist[start:end]
        green  = (secrets[:, None, :] == guesses[None, :, :])
        s_exp  = secrets[:, None, :, None].int()
        g_exp  = guesses[None, :, None, :].int()
        match  = (s_exp == g_exp)
        # A repeated guess letter only turns yellow as many times as that letter is
        # left in the secret's non-green spots (left to right); extra copies are grey.
        # e.g. secret "aback", guess "bobby": only the first b is yellow.
        avail  = (match & ~green[:, :, :, None]).sum(dim=2)
        rank   = ((same & before)[None] & ~green[:, :, None, :]).sum(dim=-1)
        yellow = ~green & (rank < avail)
        colours = torch.where(green, torch.tensor(2, device=DEVICE),
                  torch.where(yellow, torch.tensor(1, device=DEVICE),
                                      torch.tensor(0, device=DEVICE)))
        table[start:end] = (colours.to(torch.int32) * POWERS).sum(dim=-1).to(torch.int16)
    return table

table = make_table()  # table[secret, guess] = feedback pattern

POWERS = torch.tensor([243, 81, 27, 9, 3], dtype=torch.int32, device=DEVICE)
WIN_KEY = int((torch.tensor([2,2,2,2,2], dtype=torch.int32, device=DEVICE) * POWERS).sum())
FIRST_GUESS_IDX = word_to_idx[first_guess]
ANSWERS = torch.tensor([word_to_idx[w] for w in answers], device=DEVICE)  # words.txt order
EVERYTHING = torch.arange(N, device=DEVICE)                               # allowed.txt order

def bucket_counts(cands, guesses):
    """counts[i, k] = how many of `cands` give feedback k when guessing guesses[i]."""
    sub = table[cands[None, :], guesses[:, None]].long()
    counts = torch.zeros(len(guesses), 729, dtype=torch.int32, device=DEVICE)
    counts.scatter_add_(1, sub, torch.ones_like(sub, dtype=torch.int32))
    return counts

def consistent(words, history):
    keep = torch.ones(len(words), dtype=torch.bool, device=DEVICE)
    for guess, key in history:
        keep &= table[words, guess] == key
    return words[keep]

# The choosers below match the functions of the same name in minimaxwordle.py.
# argmin returns the first minimum, so list order breaks ties the same way.

def break_tie(ties, nyt_words):
    if len(ties) == 1 or len(nyt_words) <= 1:
        return ties[0].item()
    worst = bucket_counts(nyt_words, ties).max(dim=1).values
    return ties[worst.argmin()].item()

def choose_minimax_guess(possible, nyt_words):
    worst = bucket_counts(possible, possible).max(dim=1).values
    ties = possible[worst == worst.min()]
    return break_tie(ties, nyt_words), worst.min().item()

def choose_probe_guess(possible, pool, nyt_words):
    worst = bucket_counts(possible, pool).max(dim=1).values
    score = worst * 2 + (~torch.isin(pool, possible)).int()
    ties = pool[score == score.min()]
    return break_tie(ties, nyt_words), (score.min() // 2).item()

separable = {}

def can_separate(words):
    """True if some guess gives every word in `words` a different feedback."""
    if len(words) <= 2:
        return True
    key = tuple(words.tolist())
    if key not in separable:
        groups = (bucket_counts(words, EVERYTHING) > 0).sum(dim=1)
        separable[key] = bool((groups == len(words)).any())
    return separable[key]

def is_safe(guess, possible, guesses_left):
    keys = table[possible, guess]
    if guesses_left == 2:
        return len(torch.unique(keys)) == len(possible)
    return all(can_separate(possible[keys == k]) for k in torch.unique(keys) if k.item() != WIN_KEY)

def choose_late_guess(possible, nyt_words, pool, guesses_left):
    counts = bucket_counts(nyt_words, pool)
    groups = (counts > 0).sum(dim=1).long()
    worst = counts.max(dim=1).values.long()
    not_possible = (~torch.isin(pool, possible)).long()
    not_nyt = (~torch.isin(pool, nyt_words)).long()
    # most NYT groups, then a words.txt word, then any NYT word, then smallest biggest group
    key = (729 - groups) * (4 * 16384) + not_possible * (2 * 16384) + not_nyt * 16384 + worst
    for i in torch.argsort(key, stable=True):
        if is_safe(pool[i].item(), possible, guesses_left):
            return pool[i].item()
    return None

def choose_fallback_guess(possible, pool, guesses_left):
    if len(possible) <= 2 or guesses_left == 1:
        return possible[0].item()
    counts = bucket_counts(possible, pool)
    groups = (counts > 0).sum(dim=1).long()
    worst = counts.max(dim=1).values.long()
    not_candidate = (~torch.isin(pool, possible)).long()
    # most groups, then a word that could be the answer, then smallest biggest group
    key = (729 - groups) * (2 * 16384) + not_candidate * 16384 + worst
    return pool[key.argmin()].item()

def next_guess(history, attempt, probes_used, used):
    if attempt == 1:
        return FIRST_GUESS_IDX, False
    possible = consistent(ANSWERS, history)
    possible = possible[~torch.isin(possible, used)]
    nyt_words = consistent(EVERYTHING, history)
    nyt_words = nyt_words[~torch.isin(nyt_words, used)]
    if len(possible) == 0:  # answer isn't in words.txt
        if len(nyt_words) == 0:
            return None, False
        pool = EVERYTHING[~torch.isin(EVERYTHING, used)]
        return choose_fallback_guess(nyt_words, pool, 7 - attempt), False
    guesses_left = 7 - attempt
    if guesses_left in (2, 3) and len(nyt_words) > len(possible):
        pool = EVERYTHING[~torch.isin(EVERYTHING, used)]
        late = choose_late_guess(possible, nyt_words, pool, guesses_left)
        if late is not None:
            return late, False
    guess, worst = choose_minimax_guess(possible, nyt_words)
    if len(possible) >= 3 and 6 - attempt >= 2 and probes_used < 2:
        pool = ANSWERS[~torch.isin(ANSWERS, used)]
        probe, probe_worst = choose_probe_guess(possible, pool, nyt_words)
        if probe_worst < worst:
            return probe, True
    return guess, False

# The next guess only depends on the guesses and feedback so far, so games that
# start the same way share their decisions.
decisions = {}

def play(secret_idx):
    history = ()
    used = []
    probes_used = 0
    for attempt in range(1, 7):
        if history not in decisions:
            used_t = torch.tensor(used, dtype=torch.long, device=DEVICE)
            decisions[history] = next_guess(history, attempt, probes_used, used_t)
        guess_idx, is_probe = decisions[history]
        if guess_idx is None:
            return None
        probes_used += is_probe
        key = table[secret_idx, guess_idx].item()
        if key == WIN_KEY:
            return attempt
        history += ((guess_idx, key),)
        used.append(guess_idx)
    return None

def report(name, secrets):
    guess_counts = Counter()
    failed = []
    for secret_idx in tqdm(secrets, desc=name):
        n = play(secret_idx)
        if n:
            guess_counts[n] += 1
        else:
            failed.append(raw[secret_idx])
    solved = sum(guess_counts.values())
    print(f"\n{name}: solved {solved}/{len(secrets)} ({100 * solved / len(secrets):.1f}%)")
    for n in sorted(guess_counts):
        print(f"  {n}: {guess_counts[n]}")
    if solved:
        print(f"  Average: {sum(n * c for n, c in guess_counts.items()) / solved:.3f}")
    print(f"  Fails: {len(failed)}")
    if failed:
        print(f"  Failed words: {failed}")
    return failed

answer_set = set(answers)
answer_fails = report("words.txt answers", [word_to_idx[w] for w in answers])
report("other NYT words", [i for i, w in enumerate(raw) if w not in answer_set])
print("\npass" if not answer_fails else "\nFAIL: some words.txt answers were not solved")
