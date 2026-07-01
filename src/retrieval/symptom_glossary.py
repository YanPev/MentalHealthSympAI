"""
Per-item symptom glossary: synonyms / lay phrasings / behavioural cues for each
PHQ-8 item, beyond the bare item text.

Single source of truth for two consumers that must stay aligned:
  * item-aware BM25 query expansion (item_aware_retrieval.py, hybrid_retrieval.py,
    build_itemaware_prod_hybrid.py) -- appended to the item text as the query.
  * keyword-hit-rate evidence-quality measurement (retrieval_window_analysis.py)
    -- tokenized to check whether retrieved windows mention the symptom.
"""

EXPAND = {
    "NoInterest": "interest pleasure enjoy enjoyment hobbies activities fun excited "
                  "bored unmotivated stopped doing things i used to like care about",
    "Depressed": "depressed down sad hopeless unhappy crying cry low mood blue empty "
                 "despair miserable feeling bad",
    "Sleep": "sleep asleep insomnia falling asleep staying asleep waking up early "
             "sleeping too much nights restless tossing turning in bed nap",
    "Tired": "tired fatigue no energy low energy exhausted lethargic worn out drained "
             "sluggish run down",
    "Appetite": "appetite eating eat food hungry hunger overeating poor appetite "
                "weight loss weight gain meals skipping meals eating too much snacking",
    "Failure": "failure failing failed guilt guilty worthless bad about myself "
               "let myself down let my family down disappointed useless ashamed",
    "Concentrating": "concentrate concentrating focus paying attention distracted "
                     "forgetful remembering forget reading watching tv mind wandering thoughts",
    "Moving": "moving slowly speaking slowly slowed down restless fidgety fidget "
              "agitated sluggish pacing can't sit still",
}
