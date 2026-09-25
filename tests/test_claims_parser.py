"""split_claims must return exactly the bullets of a bullet-list explanation (it feeds the --reward-mode claims reward)."""
import json, os
from nla.flow.claims import split_claims, format_claims
from nla.schema import extract_explanation

HERE = os.path.dirname(os.path.abspath(__file__))


def test_warmstart_targets_roundtrip():
    T = json.load(open(os.path.join(HERE, "ws_targets_200.json")))
    assert len(T) >= 150
    bad = [t for t in T if split_claims(extract_explanation(t["response"])) != [c.rstrip(";").strip() for c in t["claims"]]]
    assert not bad, f"{len(bad)} of {len(T)} targets do not round-trip, e.g. {bad[0]}"


def test_bullet_edge_cases():
    txt = "• Genre: historical fiction (G.A. Henty's \"The Tiger of Mysore\").\n• Content: art — history.\n• Final token: \"made\".\n- short\n* The price was $3.50; e.g. cheap"
    assert split_claims(txt) == ["Genre: historical fiction (G.A. Henty's \"The Tiger of Mysore\").", "Content: art — history.", "Final token: \"made\".", "short", "The price was $3.50; e.g. cheap"]
    assert split_claims("• first claim here\n  continues on the next line\n• second") == ["first claim here continues on the next line", "second"]
    assert split_claims(format_claims(["a b c", "d"])) == ["a b c", "d"]


def test_prose_unchanged():
    txt = "The text is a recipe for bread. It ends mid-sentence after the word flour."
    assert split_claims(txt) == ["The text is a recipe for bread.", "It ends mid-sentence after the word flour."]
    assert split_claims("Genre: news.") == []          # prose fragments under MIN_WORDS are still dropped
