import unittest
from src.scoring import clean_model_output, parse_with_regex, validate_and_standardize_results

class TestScoringParser(unittest.TestCase):
    def test_clean_model_output(self):
        # Markdown block cleaning
        raw_text = "```json\n[{\"facet\": \"Risktaking\", \"score\": 4}]\n```"
        self.assertEqual(clean_model_output(raw_text), "[{\"facet\": \"Risktaking\", \"score\": 4}]")
        
        # Leading/trailing text cleaning
        raw_text = "Here is the output:\n[{\"facet\": \"Risktaking\", \"score\": 4}]\nHope this helps!"
        self.assertEqual(clean_model_output(raw_text), "[{\"facet\": \"Risktaking\", \"score\": 4}]")

    def test_parse_with_regex(self):
        raw_text = "Some broken text {\"facet\": \"Risktaking\", \"status\": \"scored\", \"score\": 4} with other {\"facet\": \"Merriness\", \"status\": \"insufficient_evidence\", \"score\": null} stuff."
        parsed = parse_with_regex(raw_text)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0]['facet'], 'Risktaking')
        self.assertEqual(parsed[1]['facet'], 'Merriness')

    def test_validate_and_standardize_results(self):
        candidates = [
            {"normalized_facet": "Risktaking", "facet_type": "personality_trait"},
            {"normalized_facet": "Merriness", "facet_type": "personality_trait"},
            {"normalized_facet": "Brevity", "facet_type": "personality_trait"}
        ]
        
        # Test case: Valid list
        valid_input = [
            {"facet": "Risktaking", "status": "scored", "score": 4, "confidence": "high", "evidence": "He jumped."},
            {"facet": "Merriness", "status": "insufficient_evidence", "score": None, "confidence": "low", "evidence": "No jokes."}
        ]
        
        standardized = validate_and_standardize_results(valid_input, candidates)
        self.assertEqual(len(standardized), 3)
        
        # Check standard fields
        risk = next(s for s in standardized if s['facet'] == 'Risktaking')
        self.assertEqual(risk['status'], 'scored')
        self.assertEqual(risk['score'], 4)
        self.assertEqual(risk['confidence'], 'high')
        
        merriness = next(s for s in standardized if s['facet'] == 'Merriness')
        self.assertEqual(merriness['status'], 'insufficient_evidence')
        self.assertEqual(merriness['score'], None)
        
        # Brevity was missing, should have default fallback status
        brevity = next(s for s in standardized if s['facet'] == 'Brevity')
        self.assertEqual(brevity['status'], 'invalid_model_output')
        self.assertEqual(brevity['score'], None)
        self.assertEqual(brevity['confidence'], 'low')

    def test_validate_invalid_score_casts_to_abstain(self):
        candidates = [{"normalized_facet": "Risktaking", "facet_type": "personality_trait"}]
        
        # Scored with out-of-bounds score (e.g. 10)
        invalid_input = [{"facet": "Risktaking", "status": "scored", "score": 10, "confidence": "high", "evidence": "risk"}]
        standardized = validate_and_standardize_results(invalid_input, candidates)
        
        self.assertEqual(standardized[0]['status'], 'insufficient_evidence')
        self.assertEqual(standardized[0]['score'], None)

if __name__ == '__main__':
    unittest.main()
