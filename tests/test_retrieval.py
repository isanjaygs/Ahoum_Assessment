import unittest
from src.database import FacetDatabase

class TestFacetRetrieval(unittest.TestCase):
    def setUp(self):
        self.db = FacetDatabase()
        
    def test_route_facets(self):
        facet_list = ["Risktaking", "FSH level", "Nationality", "Sarcasm", "NonExistentFacet123"]
        observable, not_observable = self.db.route_facets(facet_list)
        
        obs_names = {o['normalized_facet'] for o in observable}
        not_obs_names = {n['normalized_facet'] for n in not_observable}
        
        self.assertIn("Risktaking", obs_names)
        self.assertIn("FSH level", not_obs_names)
        self.assertIn("Nationality", not_obs_names)
        self.assertIn("NonExistentFacet123", not_obs_names)
        
        # NonExistentFacet123 should have custom abstention reason
        missing_facet = next(n for n in not_observable if n['normalized_facet'] == "NonExistentFacet123")
        self.assertIn("not found in the catalogue", missing_facet['abstention_reason'])

    def test_hybrid_retrieval_keyword_trigger(self):
        # Conversation contains 'risk', which should trigger the keyword Risktaking
        candidates = self.db.retrieve_observable_facets("I like taking a huge risk in life.")
        candidate_names = {c['normalized_facet'].lower() for c in candidates}
        self.assertIn("risktaking", candidate_names)
        
        # Check that perfect keyword matches have perfect retrieval scores and correct method
        risk_cand = next(c for c in candidates if c['normalized_facet'].lower() == "risktaking")
        self.assertEqual(risk_cand['retrieval_method'], 'keyword_expansion')
        self.assertEqual(risk_cand['retrieval_score'], 1.0)

if __name__ == '__main__':
    unittest.main()
