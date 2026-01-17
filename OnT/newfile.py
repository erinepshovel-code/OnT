"""
EDCM Mobile Lite v1.1
Optimized for Android Galaxy A16
Single-file, minimal dependencies, corrected fixation math, safer KV loading.
"""

import json
import math
import re
import time

from kivy.app import App
from kivy.core.clipboard import Clipboard
from kivy.lang import Builder
from kivy.uix.screenmanager import Screen

# -------------------------------------------------
# OPTIMIZED FOR MOBILE - SIMPLE BUT EFFECTIVE
# -------------------------------------------------

_TOKEN_RE = re.compile(r"\b\w+\b|[^\w\s]", re.UNICODE)

def tokenize(text: str):
    """Fast tokenizer for mobile."""
    if not text or not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())

def basic_entropy(tokens):
    """Fast entropy calculation."""
    if not tokens:
        return 0.0
    counter = {}
    for t in tokens:
        counter[t] = counter.get(t, 0) + 1
    total = len(tokens)
    return -sum((c/total) * math.log2(c/total) for c in counter.values())

def novelty_score(tokens_a, tokens_b):
    """Share of tokens in B not seen in A."""
    if not tokens_b:
        return 0.0
    set_a = set(tokens_a)
    return sum(1 for t in tokens_b if t not in set_a) / len(tokens_b)

def ttr(tokens):
    """Type-token ratio."""
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)

def repetition_ratio(tokens):
    """1 - TTR; higher means more repetition."""
    return 1.0 - ttr(tokens)

def count_patterns(text, patterns):
    """Count substring occurrences (fast)."""
    if not text:
        return 0
    tl = text.lower()
    return sum(tl.count(p) for p in patterns)

def mobile_analyze(prompt, resp_a, correction, resp_b):
    ta = tokenize(resp_a)
    tb = tokenize(resp_b)
    tc = tokenize(correction)

    len_a, len_b = len(ta), len(tb)
    ent_a = basic_entropy(ta)
    ent_b = basic_entropy(tb)

    # Short pattern lists (fast). Keep them literal substrings.
    refusal_pats = ["can't", "cannot", "sorry", "unable", "refuse", "won't"]
    hedge_pats = ["maybe", "perhaps", "might", "could", "i think", "i guess", "it seems"]

    # Density per 1k chars (cheap, stable)
    refusal_b = count_patterns(resp_b, refusal_pats) * 1000.0 / max(len(resp_b), 1)
    hedge_b = count_patterns(resp_b, hedge_pats) * 1000.0 / max(len(resp_b), 1)

    novelty = novelty_score(ta, tb)

    # Fixation should rise with repetition + low novelty (use Response B repetition)
    rep_b = repetition_ratio(tb)  # 0..1

    # Tiny "correction incorporation" proxy (optional)
    corr_overlap = 0.0
    if tc and tb:
        set_tc = set(tc)
        corr_overlap = sum(1 for t in tb if t in set_tc) / max(len(tb), 1)

    # Risk scores 0..1 (simple + interpretable)
    # Fixation: repetition-heavy + little novelty
    risk_fixation = min(0.55 * rep_b + 0.45 * (1.0 - novelty), 1.0)

    # Escalation/Shutdown: refusal density + low novelty + low correction incorporation
    risk_escalation = min(
        0.45 * min(refusal_b / 5.0, 1.0) +
        0.35 * (1.0 - novelty) +
        0.20 * (1.0 - corr_overlap),
        1.0
    )

    return {
        "tokens": f"A:{len_a} B:{len_b}",
        "entropy": f"{ent_a:.2f}→{ent_b:.2f}",
        "novelty": f"{novelty:.2f}",
        "refusal": f"{refusal_b:.1f}/1k chars",
        "hedging": f"{hedge_b:.1f}/1k chars",
        "risk_fix": f"{risk_fixation:.2f}",
        "risk_esc": f"{risk_escalation:.2f}",
        "timestamp": time.strftime("%H:%M:%S"),
        "full_json": json.dumps({
            "tokens_a": len_a,
            "tokens_b": len_b,
            "entropy_a": round(ent_a, 3),
            "entropy_b": round(ent_b, 3),
            "novelty": round(novelty, 3),
            "repetition_b": round(rep_b, 3),
            "correction_overlap_b": round(corr_overlap, 3),
            "refusal_density": round(refusal_b, 3),
            "hedging_density": round(hedge_b, 3),
            "risk_fixation": round(risk_fixation, 3),
            "risk_escalation": round(risk_escalation, 3),
        }, indent=2)
    }

def simple_bar(value, width=10):
    v = max(0.0, min(1.0, float(value)))
    filled = int(v * width)
    return "█" * filled + "░" * (width - filled)

# -------------------------------------------------
# MOBILE UI
# -------------------------------------------------

KV = r'''
<MobileScreen>:
    BoxLayout:
        orientation: 'vertical'
        padding: 10
        spacing: 6

        BoxLayout:
            size_hint_y: None
            height: 40
            Label:
                text: 'EDCM Mobile Lite'
                bold: True
                font_size: 18
                color: 0.4, 0.8, 1, 1

        Label:
            id: status
            size_hint_y: None
            height: 25
            text: 'Ready'
            color: 0.6, 0.9, 0.6, 1
            font_size: 14

        ScrollView:
            do_scroll_x: False
            BoxLayout:
                orientation: 'vertical'
                size_hint_y: None
                height: self.minimum_height
                spacing: 6

                Label:
                    text: 'Prompt:'
                    size_hint_y: None
                    height: 20
                    color: 0.8, 0.8, 0.6, 1

                TextInput:
                    id: prompt
                    hint_text: 'Original question'
                    size_hint_y: None
                    height: 60
                    multiline: True
                    font_size: 14

                Label:
                    text: 'Response A:'
                    size_hint_y: None
                    height: 20
                    color: 0.8, 0.8, 0.6, 1

                TextInput:
                    id: resp_a
                    hint_text: 'First response'
                    size_hint_y: None
                    height: 70
                    multiline: True
                    font_size: 14

                Label:
                    text: 'Correction:'
                    size_hint_y: None
                    height: 20
                    color: 0.8, 0.8, 0.6, 1

                TextInput:
                    id: correction
                    hint_text: 'Your correction/feedback'
                    size_hint_y: None
                    height: 70
                    multiline: True
                    font_size: 14

                Label:
                    text: 'Response B:'
                    size_hint_y: None
                    height: 20
                    color: 0.8, 0.8, 0.6, 1

                TextInput:
                    id: resp_b
                    hint_text: 'Follow-up response'
                    size_hint_y: None
                    height: 80
                    multiline: True
                    font_size: 14

        BoxLayout:
            size_hint_y: None
            height: 44
            spacing: 6

            Button:
                text: 'ANALYZE'
                font_size: 16
                bold: True
                background_color: 0.2, 0.6, 0.9, 1
                on_press: root.analyze()

            Button:
                text: 'COPY'
                font_size: 16
                background_color: 0.3, 0.5, 0.8, 1
                on_press: root.copy_results()

            Button:
                text: 'EXIT'
                font_size: 16
                background_color: 0.5, 0.2, 0.2, 1
                on_press: app.stop()

        ScrollView:
            TextInput:
                id: results
                readonly: True
                font_size: 13
                background_color: 0.1, 0.1, 0.15, 1
                foreground_color: 0.9, 0.95, 0.9, 1
                hint_text: 'Results will appear here...'
'''

class MobileScreen(Screen):
    def analyze(self):
        try:
            prompt = self.ids.prompt.text.strip()
            resp_a = self.ids.resp_a.text.strip()
            correction = self.ids.correction.text.strip()
            resp_b = self.ids.resp_b.text.strip()

            if not resp_a or not resp_b:
                self.ids.status.text = "Need Response A & B"
                self.ids.status.color = (0.9, 0.3, 0.3, 1)
                return

            results = mobile_analyze(prompt, resp_a, correction, resp_b)

            display_text = f"""
╔══════════════════════════════╗
║     EDCM MOBILE ANALYSIS     ║
╚══════════════════════════════╝

📊 BASIC METRICS:
  Tokens:     {results['tokens']}
  Entropy:    {results['entropy']}
  Novelty:    {results['novelty']} {simple_bar(float(results['novelty']))}

📝 PATTERN DENSITY:
  Refusal:    {results['refusal']}
  Hedging:    {results['hedging']}

⚠️  RISK SCORES:
  Fixation:   {results['risk_fix']} {simple_bar(float(results['risk_fix']))}
  Escalation: {results['risk_esc']} {simple_bar(float(results['risk_esc']))}

⏰ Analysis time: {results['timestamp']}

═══════════════════════════════
(Scroll down for JSON data)
═══════════════════════════════

{results['full_json']}
""".strip()

            self.ids.results.text = display_text
            self.ids.status.text = f"✓ Analyzed at {results['timestamp']}"
            self.ids.status.color = (0.4, 0.9, 0.4, 1)

        except Exception as e:
            self.ids.status.text = f"Error: {str(e)[:40]}..."
            self.ids.status.color = (0.9, 0.3, 0.3, 1)

    def copy_results(self):
        try:
            results_text = (self.ids.results.text or "").strip()
            if not results_text or "Results will appear" in results_text:
                self.ids.status.text = "Nothing to copy"
                self.ids.status.color = (0.9, 0.6, 0.2, 1)
                return

            Clipboard.copy(results_text)
            self.ids.status.text = "✓ Copied to clipboard!"
            self.ids.status.color = (0.4, 0.8, 0.9, 1)
        except Exception:
            self.ids.status.text = "Copy failed"
            self.ids.status.color = (0.9, 0.3, 0.3, 1)

class EDCMobile(App):
    def build(self):
        Builder.load_string(KV)
        scr = MobileScreen()
        scr.name = "main"
        return scr

if __name__ == "__main__":
    EDCMobile().run()
