def score_non_observable_facets(non_observable_facets: list[dict]) -> list[dict]:
    """
    Directly bypasses the LLM for non-observable facets and generates 
    deterministic policy abstention results.
    
    The confidence levels are mapped directly from the audit's classification 
    rule confidence ('high' or 'medium'), and the evidence matches the pre-defined 
    abstention reason.
    """
    results = []
    for facet in non_observable_facets:
        confidence = facet.get('rule_confidence', 'high').lower()
        # Fallback if confidence value is not in low/medium/high format
        if confidence not in ['low', 'medium', 'high']:
            confidence = 'high'
            
        results.append({
            'facet': facet['normalized_facet'],
            'status': 'not_observable',
            'score': None,
            'confidence': confidence,
            'evidence': facet.get(
                'abstention_reason', 
                '[Abstention Policy] This facet requires biographical, medical, or physiological evidence and cannot be scored from conversation text.'
            )
        })
    return results
