from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    StaleElementReferenceException, TimeoutException, WebDriverException,
)
from collections import Counter
import time
import sys
import os

def resource_path(relative_path):
    base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
    return os.path.join(base_path, relative_path)

with open(resource_path("words.txt"), "r") as file:
    ALL_WORDS = [line.strip().lower() for line in file if len(line.strip()) == 5]

# Every word NYT accepts as a guess (matches the list in the live game). NYT has
# picked answers outside words.txt, so the solver falls back to this list when no
# word in words.txt fits.
with open(resource_path("allowed.txt"), "r") as file:
    ALLOWED_WORDS = list(dict.fromkeys(line.strip().lower() for line in file if len(line.strip()) == 5))

options = webdriver.ChromeOptions()
options.headless = False
# Selenium Manager (built into Selenium 4.6+) finds Chrome and a matching driver,
# and downloads Chrome for Testing if Chrome isn't installed (e.g. in WSL).
driver = webdriver.Chrome(options=options)

def sim_feedback(secret, guess):
    out = ["absent"] * 5
    remaining = Counter()
    for i in range(5):
        if guess[i] == secret[i]:
            out[i] = "correct"
        else:
            remaining[secret[i]] += 1
    for i in range(5):
        if out[i] == "absent":
            ch = guess[i]
            if remaining[ch] > 0:
                out[i] = "present"
                remaining[ch] -= 1
    return tuple(out)

def filter_possible(possible, guess, observed_states):
    obs = tuple(observed_states)
    return [w for w in possible if sim_feedback(w, guess) == obs]

def choose_minimax_guess(possible, banned):
    best = None
    best_score = float("inf")
    for g in possible:
        if g in banned:
            continue
        buckets = {}
        for s in possible:
            k = sim_feedback(s, g)
            buckets[k] = buckets.get(k, 0) + 1
        worst = max(buckets.values())
        if worst < best_score:
            best_score = worst
            best = g
    return best, best_score

def choose_probe_guess(possible, all_words, banned):
    # On a tie, prefer a word that could still be the answer: it splits the
    # remaining words just as well and might win outright. Without this, a tie
    # goes to whatever comes first alphabetically ("aback", "abase", ...).
    candidates = set(possible)
    best = None
    best_score = (float("inf"), True)
    for g in all_words:
        if g in banned:
            continue
        buckets = {}
        for s in possible:
            k = sim_feedback(s, g)
            buckets[k] = buckets.get(k, 0) + 1
        score = (max(buckets.values()), g not in candidates)
        if score < best_score:
            best_score = score
            best = g
    return best, best_score[0]

def type_guess(word):
    dismiss_popups()
    body = driver.find_element(By.TAG_NAME, "body")
    for ch in word:
        body.send_keys(ch)
        time.sleep(0.1)  
    body.send_keys(Keys.ENTER)

# Tiles of one board row. Matching every div[data-testid='tile'] on the page also
# picks up the 15 example tiles in the "How To Play" popup, which come first.
ROW_TILES = "div[role='group'][aria-label='Row {}'] div[data-testid='tile']"
AD_CONTINUE = "button[class*='Skip-module_skipButton']"  # "Continue to Wordle" on the pre-game ad
AD_OVERLAY = "div[class*='AdInterstitial']"
HELP_CLOSE = "button[class*='Modal-module_closeIcon']"

def wait_row_states(row):
    def evaluated(d):
        states = [t.get_attribute("data-state") for t in d.find_elements(By.CSS_SELECTOR, ROW_TILES.format(row))]
        if len(states) == 5 and all(s in {"correct", "present", "absent"} for s in states):
            return states
        return False
    return WebDriverWait(driver, 12, ignored_exceptions=[StaleElementReferenceException]).until(evaluated)

def visible(css):
    return [el for el in driver.find_elements(By.CSS_SELECTOR, css) if el.is_displayed()]

def dismiss_popups():
    """Click away the ad / How To Play popup if showing. Returns True if the board is usable."""
    for el in visible(AD_CONTINUE) + visible(HELP_CLOSE):
        try:
            el.click()
        except WebDriverException:
            pass  # still fading in, or already gone; try again next time
    blocked = visible(AD_OVERLAY) or visible(HELP_CLOSE)
    return not blocked and bool(driver.find_elements(By.CSS_SELECTOR, ROW_TILES.format(1)))

def start_game(timeout=90, settle=3):
    try:
        WebDriverWait(driver, 8).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='Play']"))
        ).click()
    except TimeoutException:
        pass
    # Without a subscription, Play is often followed by a full-screen video ad, and
    # after that the "How To Play" popup. Keystrokes go nowhere until both are gone,
    # and the board isn't rendered while the ad is up. The popup shows up a moment
    # after the ad closes, so wait until the board has stayed clear for `settle` s.
    deadline = time.time() + timeout
    clear_since = None
    while time.time() < deadline:
        if dismiss_popups():
            clear_since = clear_since or time.time()
            if time.time() - clear_since >= settle:
                return
        else:
            clear_since = None
        time.sleep(0.5)
    raise RuntimeError("Could not get past the ad / How To Play popup to the board.")

def solve():
    possible = ALL_WORDS[:]
    nyt_words = ALLOWED_WORDS[:]  # every NYT word that still fits the feedback
    probes_used = 0
    used_guesses = set()
    using_allowed = False

    for attempt in range(1, 7):
        if attempt == 1:
            guess = "soare"
        else:
            remaining_attempts = 6 - attempt
            guess, worst = choose_minimax_guess(possible, used_guesses)
            # Only spend a guess on a probe when it splits the remaining words
            # better than guessing one of them. With 1-2 words left, guessing one
            # of them is always at least as good.
            if len(possible) >= 3 and remaining_attempts >= 2 and probes_used < 2:
                probe, probe_worst = choose_probe_guess(possible, ALL_WORDS, used_guesses)
                if probe_worst < worst:
                    guess = probe
                    probes_used += 1
        if not guess:
            guess = next((w for w in possible if w not in used_guesses), None)
            if not guess:
                print("No words left to guess.")
                return

        used_guesses.add(guess)

        print(f"Guess {attempt}: {guess}")
        type_guess(guess)
        time.sleep(2)
        states = wait_row_states(attempt)
        if all(s == "correct" for s in states):
            print(f"Solved in {attempt}.")
            return
        possible = filter_possible(possible, guess, states)
        if guess in possible:
            possible.remove(guess)
        nyt_words = filter_possible(nyt_words, guess, states)
        if guess in nyt_words:
            nyt_words.remove(guess)

        if not possible and not using_allowed:
            print("Answer isn't in words.txt; switching to allowed.txt.")
            using_allowed = True
            possible = nyt_words[:]

        if not possible:
            print("No word matches the feedback.")
            return

        time.sleep(0.6)
    print("Out of guesses.")

try:
    driver.get("https://www.nytimes.com/games/wordle/index.html")
    start_game()
    time.sleep(1)
    solve()
finally:
    input("\nPress Enter to quit...")
    driver.quit()
