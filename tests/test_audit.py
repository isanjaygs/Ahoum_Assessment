import unittest
from src.audit import normalize_facet_name, classify_facet

class TestFacetAudit(unittest.TestCase):
    def test_normalize_facet_name(self):
        self.assertEqual(normalize_facet_name("800. Sufi practice: Sufi retreat attendance count"), "Sufi retreat attendance count")
        self.assertEqual(normalize_facet_name("644. Spiritual virtue: Humility practice index"), "Humility practice index")
        self.assertEqual(normalize_facet_name("754. I Ching hexagram 36 resonance level"), "I Ching hexagram 36 resonance level")
        self.assertEqual(normalize_facet_name("Democratic Leadership:"), "Democratic Leadership")
        self.assertEqual(normalize_facet_name("  Assertiveness and control   "), "Assertiveness and control")
        
    def test_classify_facet_types(self):
        # 1. Malformed / Header
        ftype, obs_level, conf, _, _ = classify_facet("Democratic Leadership:", "Democratic Leadership")
        self.assertEqual(ftype, "malformed_header")
        self.assertEqual(obs_level, "not_observable")
        
        # 2. Medical / Physiological
        ftype, obs_level, conf, _, _ = classify_facet("FSH level", "FSH level")
        self.assertEqual(ftype, "medical_physiological")
        self.assertEqual(obs_level, "not_observable")
        self.assertEqual(conf, "high")
        
        # 3. Biographical / Demographic
        ftype, obs_level, conf, _, _ = classify_facet("Nationality", "Nationality")
        self.assertEqual(ftype, "biographical_demographic")
        self.assertEqual(obs_level, "not_observable")
        
        # 4. Cognitive / Formal test
        ftype, obs_level, conf, _, _ = classify_facet("Intelligence Quotient (IQ)", "Intelligence Quotient (IQ)")
        self.assertEqual(ftype, "cognitive_skill")
        self.assertEqual(obs_level, "not_observable")
        
        # 5. Daily Habit / Activity
        ftype, obs_level, conf, _, _ = classify_facet("Public-transport km/week", "Public-transport km/week")
        self.assertEqual(ftype, "daily_habit_activity")
        self.assertEqual(obs_level, "not_observable")
        
        # 6. Personality Traits (Indirect vs Direct)
        ftype, obs_level, conf, _, _ = classify_facet("Risktaking", "Risktaking")
        self.assertEqual(ftype, "personality_trait")
        self.assertEqual(obs_level, "indirect")
        
        ftype, obs_level, conf, _, _ = classify_facet("Sarcasm", "Sarcasm")
        self.assertEqual(ftype, "personality_trait")
        self.assertEqual(obs_level, "direct")
        
    def test_avoid_partial_keyword_matches(self):
        # Originality has 'origin' which could match biographical, but should default to trait
        ftype, obs_level, _, _, _ = classify_facet("Originality", "Originality")
        self.assertEqual(ftype, "personality_trait")
        self.assertEqual(obs_level, "indirect")
        
        # Courageousness has 'age', should default to trait
        ftype, obs_level, _, _, _ = classify_facet("Courageousness", "Courageousness")
        self.assertEqual(ftype, "personality_trait")
        self.assertEqual(obs_level, "indirect")
        
        # Language use has 'age', should default to trait/direct
        ftype, obs_level, _, _, _ = classify_facet("Language use", "Language use")
        self.assertEqual(ftype, "personality_trait")
        self.assertEqual(obs_level, "direct")
        
        # Techniques has 'iq' as substring, should NOT be classified as cognitive_skill
        ftype, obs_level, _, _, _ = classify_facet("Types of Mindfulness Techniques Used", "Types of Mindfulness Techniques Used")
        self.assertEqual(ftype, "spiritual_religious")
        self.assertEqual(obs_level, "not_observable")

if __name__ == '__main__':
    unittest.main()
