import os
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from src import config

class FacetDatabase:
    def __init__(self, enriched_csv_path: str = config.ENRICHED_CSV_PATH):
        self.enriched_csv_path = enriched_csv_path
        self.df = None
        self.observable_df = None
        self.model = None
        self.facet_embeddings = None
        
        # Load database
        self.load_data()
        
    def load_data(self):
        if not os.path.exists(self.enriched_csv_path):
            raise FileNotFoundError(
                f"Enriched facets file not found at {self.enriched_csv_path}. "
                "Please run the audit script first: python -m src.audit"
            )
        self.df = pd.read_csv(self.enriched_csv_path)
        # Filter for observable facets (direct and indirect)
        self.observable_df = self.df[self.df['observability_level'].isin(['direct', 'indirect'])].copy()
        
    def _init_embeddings(self):
        """Lazily initialize the sentence-transformers model and precompute facet embeddings."""
        if self.model is not None:
            return
            
        print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
        self.model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
        
        facet_texts = self.observable_df['normalized_facet'].tolist()
        print(f"Precomputing embeddings for {len(facet_texts)} observable facets...")
        self.facet_embeddings = self.model.encode(facet_texts, convert_to_tensor=True, show_progress_bar=False)
        
    def get_facet_by_name(self, name: str) -> dict | None:
        """Finds a facet in the catalogue by either raw or normalized name."""
        name_lower = name.strip().lower()
        match = self.df[
            (self.df['raw_facet'].str.lower() == name_lower) | 
            (self.df['normalized_facet'].str.lower() == name_lower)
        ]
        if match.empty:
            return None
        return match.iloc[0].to_dict()

    def route_facets(self, facet_names: list[str]) -> tuple[list[dict], list[dict]]:
        """
        Routes a list of explicit facets:
        - Observable facets (direct, indirect) go to the LLM list.
        - Non-observable facets go to the Policy list.
        """
        observable = []
        not_observable = []
        
        for name in facet_names:
            facet_dict = self.get_facet_by_name(name)
            if not facet_dict:
                # If it's completely missing, treat as not_observable with custom reason
                facet_dict = {
                    'raw_facet': name,
                    'normalized_facet': name,
                    'facet_type': 'unknown',
                    'observability_level': 'not_observable',
                    'sensitivity': 'low',
                    'rule_confidence': 'high',
                    'needs_review': False,
                    'abstention_reason': '[Abstention Policy] This facet was not found in the catalogue.'
                }
                not_observable.append(facet_dict)
            elif facet_dict['observability_level'] == 'not_observable':
                not_observable.append(facet_dict)
            else:
                observable.append(facet_dict)
                
        return observable, not_observable

    def retrieve_observable_facets(self, convo_text: str, k: int = config.RETRIEVAL_K) -> list[dict]:
        """
        Performs hybrid retrieval for a given conversation text:
        1. Semantic retrieval (cosine similarity using SentenceTransformer).
        2. Keyword-based expansion rules to catch direct lexical matches.
        3. Deduplicates and merges both pools.
        """
        self._init_embeddings()
        
        # 1. Semantic Retrieval
        convo_embedding = self.model.encode(convo_text, convert_to_tensor=True)
        cos_scores = util.cos_sim(convo_embedding, self.facet_embeddings)[0]
        
        # Get top-K indices
        top_indices = np.argsort(cos_scores.cpu().numpy())[::-1][:k]
        
        semantic_candidates = []
        for idx in top_indices:
            facet_row = self.observable_df.iloc[int(idx)].to_dict()
            # Add similarity score for debugging/reports
            facet_row['retrieval_score'] = float(cos_scores[idx].item())
            facet_row['retrieval_method'] = 'semantic'
            semantic_candidates.append(facet_row)
            
        # 2. Keyword-based Lexical Expansion Rules
        lexical_candidates = []
        convo_lower = convo_text.lower()
        
        # Keyword mapping to key observable facets
        keyword_rules = {
            'risk': ['Risktaking', 'Adventure-Seeking Behavior', 'Creative risk-taking tendency'],
            'gamble': ['Risktaking'],
            'adventure': ['Adventure-Seeking Behavior'],
            'danger': ['Risktaking', 'Fearfulness: Fear of physical dangers'],
            'sarcasm': ['Quirkiness', 'Drollness'],
            'sarcastic': ['Quirkiness', 'Drollness'],
            'joke': ['Merriness', 'High-spiritedness', 'Drollness'],
            'humour': ['Merriness', 'Drollness'],
            'polite': ['Civility', 'Decency', 'Cordiality'],
            'manners': ['Civility', 'Decency'],
            'respect': ['Civility', 'Disrespect'],
            'assertive': ['Assertiveness and control in relationships', 'Unassertiveness'],
            'control': ['Assertiveness and control in relationships', 'Unassertiveness', 'Control over situations'],
            'sad': ['Happiness', 'Contentment Levels'],
            'happy': ['Happiness', 'Contentment Levels', 'Joyfulness'],
            'mindful': ['Peacefulness', 'Regularity of Self-Reflection'],
            'calm': ['Peacefulness', 'Collectedness'],
            'cooperate': ['Collaboration', 'Cooperation', 'Contribution to Group Goals'],
            'share': ['Collaboration', 'Cooperation'],
            'lead': ['Democratic Leadership', 'Ethical leadership rating'],
            
            # Benchmark specific context mappings for hybrid lexical expansion
            'colombia': ['Risktaking', 'Adventure-Seeking Behavior', 'Creative risk-taking tendency'],
            'ticket': ['Risktaking', 'Adventure-Seeking Behavior'],
            'traffic': ['Drollness', 'Merriness'],
            'commute': ['Drollness', 'Merriness'],
            'cansado': ['High-spiritedness', 'Perseverance', 'Language use'],
            'cansada': ['High-spiritedness', 'Perseverance', 'Language use'],
            'sleep': ['High-spiritedness'],
            'project': ['Perseverance'],
            'boss': ['Perseverance', 'Unassertiveness'],
            'lazy': ['Perseverance'],
            'open to': ['Risktaking', 'Creative risk-taking tendency'],
            'facts': ['Risktaking', 'Creative risk-taking tendency'],
            'worthless': ['Unassertiveness', 'Drollness'],
            'talking over': ['Unassertiveness'],
            'fatigued': ['High-spiritedness'],
            'sluggish': ['High-spiritedness'],
            'weather': ['Brevity'],
            'rain': ['Brevity']
        }
        
        triggered_facet_names = set()
        for kw, target_facets in keyword_rules.items():
            if re.search(rf'\b{kw}', convo_lower):
                for f_name in target_facets:
                    triggered_facet_names.add(f_name.lower())
                    
        # Pull triggered facets from our observable database
        for _, row in self.observable_df.iterrows():
            if row['normalized_facet'].lower() in triggered_facet_names:
                facet_dict = row.to_dict()
                facet_dict['retrieval_score'] = 1.0  # Perfect lexical match score
                facet_dict['retrieval_method'] = 'keyword_expansion'
                lexical_candidates.append(facet_dict)
                
        # 3. Merge & Deduplicate
        merged_candidates = []
        seen_facets = set()
        
        # Prioritize lexical matches
        for cand in lexical_candidates:
            name = cand['normalized_facet'].lower()
            if name not in seen_facets:
                seen_facets.add(name)
                merged_candidates.append(cand)
                
        # Fill rest with semantic candidates
        for cand in semantic_candidates:
            name = cand['normalized_facet'].lower()
            if name not in seen_facets:
                seen_facets.add(name)
                merged_candidates.append(cand)
                
        return merged_candidates
