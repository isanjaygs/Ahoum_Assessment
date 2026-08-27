import os
import re
import pandas as pd

def normalize_facet_name(raw_name: str) -> str:
    """
    Normalizes the facet name by:
    1. Stripping leading index numbers and specific prefixes (e.g. '800. Sufi practice: ...').
    2. Stripping leading/trailing spaces and quotes.
    3. Stripping trailing colons.
    """
    name = raw_name.strip().strip('"').strip("'")
    
    # Match prefixes like '800. Sufi practice: ' or '754. I Ching hexagram 36 resonance level'
    # Pattern: optional digits, followed by a dot, followed by optional spaces, 
    # and optionally a category prefix and colon (e.g., 'Sufi practice:')
    pattern = r'^\d+\.\s*(?:[A-Za-z0-9–’\'\"\s\-]+:\s*)?(.*)'
    match = re.match(pattern, name)
    if match:
        name = match.group(1).strip()
        
    # Strip trailing colons
    name = name.rstrip(':').strip()
    return name

def contains_keyword(text: str, keyword: str) -> bool:
    """
    Checks if keyword is in text, enforcing strict word boundaries for short or highly 
    ambiguous keywords that could match parts of other words (e.g., 'age' in 'language' 
    or 'originality', 'iq' in 'techniques').
    """
    bounded_keywords = [
        'age', 'origin', 'sex', 'pain', 'diet', 'fsh', 'sufi', 'aura', 'input', 'iq', 
        'gland', 'blood', 'mind', 'drug', 'data', 'fili', 'term'
    ]
    if keyword in bounded_keywords:
        return bool(re.search(rf'\b{keyword}\b', text))
    return keyword in text

def classify_facet(raw_name: str, normalized_name: str) -> tuple[str, str, str, bool, str]:
    """
    Classifies a facet into:
    (facet_type, observability_level, rule_confidence, needs_review, abstention_reason)
    
    Observability levels:
    - 'direct': Directly observable conversational styles/signals (sarcasm, brevity).
    - 'indirect': Inferable attitudes, mood or personality traits requiring evidence.
    - 'not_observable': Requires clinical/medical testing, biographical validation, or diary logging.
    """
    raw_lower = raw_name.lower().strip()
    norm_lower = normalized_name.lower()
    
    # 1. Malformed or Header-like Rows
    if raw_lower.endswith(':') or raw_lower in ['facets'] or re.match(
        r'.*(?:subcomponents|components|themes|parameters|points|facets|types|drivers|domain)$', norm_lower
    ):
        return (
            'malformed_header',
            'not_observable',
            'high',
            False,
            '[Abstention Policy] This is a category header or malformed entry, not a scoreable facet.'
        )

    # 2. Medical / Clinical / Physiological (Clinical scales, diagnoses, symptoms, lab values)
    med_strong = [
        'level', 'hormone', 'basophil', 'fsh', 'parathyroid', 'chromatin', 'serotonin', 
        'cardiovascular', 'apnea', 'gene', 'metabolic rate', 'macronutrient', 'basophil count', 
        'pain presence', 'sleep-disorder', 'diagnosis', 'disease', 'transporter', 'polygenic risk',
        'burnout symptoms', 'depression symptoms', 'clinical depression', 'depression (dep)', 
        'hypomania', 'hysteria', 'psychoticism', 'clinical', 'medical'
    ]
    med_medium = ['dietary', 'sleep', 'caffeine', 'health', 'acidity', 'pain']
    
    if any(contains_keyword(norm_lower, k) for k in med_strong):
        return (
            'medical_physiological',
            'not_observable',
            'high',
            False,
            '[Abstention Policy] This facet requires medical testing, laboratory values, or a clinical diagnosis.'
        )
        
    if any(contains_keyword(norm_lower, k) for k in med_medium):
        needs_review = True
        # Specific overrides
        if 'caffeine intake' in norm_lower or 'sleep-environment' in norm_lower or 'snacking' in norm_lower:
            facet_type = 'daily_habit_activity'
            obs_level = 'not_observable'
        else:
            facet_type = 'medical_physiological'
            obs_level = 'not_observable'
            
        return (
            facet_type,
            obs_level,
            'medium',
            needs_review,
            f'[Abstention Policy] This facet ({facet_type}) requires physiological measurements or diary logging.'
        )

    # 3. Spiritual / Religious / Astrology
    spiritual_strong = [
        'sufi', 'i ching', 'hexagram', 'kabbalah', 'quran', 'khatam', 'ridvan', 'reiki', 
        'scorpio', 'rising sign', 'gnostic', 'archon', 'yoga', 'vrata', 'mantra', 'bhagavad-gita', 
        'shabbat', 'dhikr', 'kirtan', 'tiferet', 'sephira', 'channeling', 'pilgrimage', 'scripture',
        'bible', 'buddhist', 'hindu', 'jewish', 'islamic', 'sikh', 'bahá’í', 'christian', 'religion',
        'zohar', 'seerah', 'sacred text', 'holiness'
    ]
    spiritual_medium = ['meditation', 'spiritual', 'spirituality', 'faith', 'belief', 'aura', 'mindful', 'mindfulness']
    
    if any(contains_keyword(norm_lower, k) for k in spiritual_strong):
        if any(cnt in norm_lower for cnt in ['count', 'cycles', 'hours', 'index', 'days', 'consistency', 'repetitions']):
            return (
                'spiritual_religious',
                'not_observable',
                'high',
                False,
                '[Abstention Policy] This facet requires tracking specific religious practices or attendance metrics.'
            )
        else:
            return (
                'spiritual_religious',
                'indirect',
                'medium',
                True,
                '[Abstention Policy] Spiritual beliefs can sometimes be discussed, but require explicit verbal evidence.'
            )
            
    if any(contains_keyword(norm_lower, k) for k in spiritual_medium):
        # Specific override for "types of techniques used"
        if 'techniques used' in norm_lower or 'types of' in norm_lower:
            return (
                'spiritual_religious',
                'not_observable',
                'high',
                False,
                '[Abstention Policy] Listing specific mindfulness techniques used requires diary verification.'
            )
        return (
            'spiritual_religious',
            'indirect',
            'medium',
            True,
            '[Abstention Policy] Requires verification of spiritual practices or beliefs.'
        )

    # 4. Cognitive / Formal Testing
    cognitive_strong = [
        'iq', 'intelligence quotient', 'filing skills', 'numerical reasoning', 'spatial perception', 
        'calculations', 'sequence identification', 'spelling accuracy', 'memory recall', 'auditory memory', 
        'sequential memory', 'memory for sounds', 'working memory index', 'faux pas recognition',
        'alphanumeric', 'filing'
    ]
    cognitive_medium = ['reasoning', 'memory', 'arithmetic', 'data analysis', 'computer skills', 'mathematical', 'data']
    
    if any(contains_keyword(norm_lower, k) for k in cognitive_strong):
        return (
            'cognitive_skill',
            'not_observable',
            'high',
            False,
            '[Abstention Policy] This facet requires standardized cognitive testing, tests of memory, or formal assessments.'
        )
    if any(contains_keyword(norm_lower, k) for k in cognitive_medium):
        return (
            'cognitive_skill',
            'indirect',
            'medium',
            True,
            '[Abstention Policy] General cognitive skills require high evidence thresholds to score from dialogue.'
        )

    # 5. Biographical / Demographic
    biographical_strong = [
        'nationality', 'childhood', 'commute time', 'passport', 'volunteer', 'open-source', 
        'demographic', 'age', 'gender', 'income', 'commute', 'subscription count', 'subscriber count',
        'identity diffusion'
    ]
    biographical_medium = ['background', 'origin', 'residence', 'nationality', 'education']
    
    if any(contains_keyword(norm_lower, k) for k in biographical_strong):
        return (
            'biographical_demographic',
            'not_observable',
            'high',
            False,
            '[Abstention Policy] This facet is a biographical or demographic fact that cannot be verified without direct self-report.'
        )
    if any(contains_keyword(norm_lower, k) for k in biographical_medium):
        return (
            'biographical_demographic',
            'not_observable',
            'medium',
            True,
            '[Abstention Policy] Biographical history is typically not observable in short conversation snippets.'
        )

    # 6. Daily Habit / Activity
    habit_strong = [
        'km/week', 'hours/week', 'sessions/week', 'visits/year', 'hours/day', 'sourcing %', 
        'meals', 'cardio', 'rehearsal', 'snacking', 'skipping', 'frequency', 'pet-enrichment', 
        'outdoors/day', 'transport', 'sourcing', 'eating habits', 'processed-food'
    ]
    habit_medium = ['cooking', 'diet', 'dance', 'subscription', 'travel', 'food', 'museum', 'choir']
    
    if any(contains_keyword(norm_lower, k) for k in habit_strong):
        return (
            'daily_habit_activity',
            'not_observable',
            'high',
            False,
            '[Abstention Policy] This is a specific daily habit or quantitative metric that requires behavioral tracking.'
        )
    if any(contains_keyword(norm_lower, k) for k in habit_medium):
        if 'arts' in norm_lower or 'appreciation' in norm_lower or 'interest' in norm_lower:
            return (
                'daily_habit_activity',
                'indirect',
                'medium',
                True,
                '[Abstention Policy] General interest in activities can be discussed, but require verbal declaration.'
            )
        return (
            'daily_habit_activity',
            'not_observable',
            'medium',
            True,
            '[Abstention Policy] Daily habits or hobbies are not directly observable without explicit verbal confirmation.'
        )

    # 7. Personality / Behavior (Default)
    # Direct observable conversational signals (style, immediate cues)
    direct_keywords = [
        'brevity', 'talkativeness', 'outspokenness', 'hesitation', 'clumsiness', 'sentence structure', 
        'spelling accuracy', 'language use', 'eye-contact', 'word count', 'response time', 'sarcasm', 
        'humour', 'politeness', 'disrespect', 'impudence', 'brazenness'
    ]
    
    if any(contains_keyword(norm_lower, k) for k in direct_keywords):
        return (
            'personality_trait',
            'direct',
            'high',
            False,
            ''
        )
        
    # Default is indirect observable personality trait
    return (
        'personality_trait',
        'indirect',
        'high',
        False,
        ''
    )

def audit_facets_pipeline(input_csv_path: str, output_csv_path: str):
    """
    Main preprocessing pipeline that reads the raw CSV, cleans it, 
    applies the taxonomy, and outputs the enriched CSV.
    """
    df = pd.read_csv(input_csv_path)
    df.columns = [c.strip() for c in df.columns]
    raw_col = df.columns[0]
    
    enriched_rows = []
    for idx, row in df.iterrows():
        raw_facet = str(row[raw_col]).strip()
        if not raw_facet or raw_facet.lower() == 'facets':
            continue
            
        normalized = normalize_facet_name(raw_facet)
        facet_type, observability_level, rule_confidence, needs_review, reason = classify_facet(raw_facet, normalized)
        
        # Check sensitivity
        sensitivity = 'low'
        norm_lower = normalized.lower()
        if any(contains_keyword(norm_lower, k) for k in ['drug', 'alcohol', 'violence', 'kink', 'sexual', 'apnea', 'depression', 'burnout', 'pain']):
            sensitivity = 'high'
        elif any(contains_keyword(norm_lower, k) for k in ['religion', 'spiritual', 'politics', 'income', 'ethical', 'lending', 'shabbat', 'quran', 'bible']):
            sensitivity = 'medium'
            
        enriched_rows.append({
            'raw_facet': raw_facet,
            'normalized_facet': normalized,
            'facet_type': facet_type,
            'observability_level': observability_level,
            'sensitivity': sensitivity,
            'rule_confidence': rule_confidence,
            'needs_review': needs_review,
            'abstention_reason': reason
        })
        
    enriched_df = pd.DataFrame(enriched_rows)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    enriched_df.to_csv(output_csv_path, index=False)
    obs   = enriched_df['observability_level'].value_counts()
    ftype = enriched_df['facet_type'].value_counts()
    sens  = enriched_df['sensitivity'].value_counts()
    nr    = enriched_df['needs_review'].value_counts()

    print(f"\n  ✓ Audited {len(enriched_df)} facets → {output_csv_path}\n")
    print("  Observability          Facet Type                    Sensitivity   Review")
    print("  ─────────────────────  ────────────────────────────  ────────────  ──────")
    rows = max(len(obs), len(ftype), len(sens), 2)
    obs_items   = list(obs.items())
    ftype_items = list(ftype.items())
    sens_items  = list(sens.items())
    nr_items    = [("needs review", nr.get(True, 0)), ("confident",    nr.get(False, 0))]
    for i in range(rows):
        o  = f"{obs_items[i][0]:12} {obs_items[i][1]:3}"   if i < len(obs_items)   else ""
        ft = f"{ftype_items[i][0]:26} {ftype_items[i][1]:3}" if i < len(ftype_items) else ""
        s  = f"{sens_items[i][0]:8} {sens_items[i][1]:3}"  if i < len(sens_items)  else ""
        nr_= f"{nr_items[i][0]:12} {nr_items[i][1]:3}"     if i < len(nr_items)    else ""
        print(f"  {o:<23}  {ft:<30}  {s:<14}  {nr_}")

if __name__ == '__main__':
    input_path = '/Users/sanjaygs/Documents/Study/Projects/Ahoum/Facets Assignment.csv'
    output_path = '/Users/sanjaygs/Documents/Study/Projects/Ahoum/data/facets_enriched.csv'
    audit_facets_pipeline(input_path, output_path)
