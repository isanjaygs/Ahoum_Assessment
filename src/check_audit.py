import pandas as pd

df = pd.read_csv('data/facets_enriched.csv')

for ftype in df['facet_type'].unique():
    print(f"\n===== TYPE: {ftype} =====")
    sub = df[df['facet_type'] == ftype]
    # Print up to 40 examples
    for _, row in sub.head(40).iterrows():
        print(f" - {row['normalized_facet']} ({row['observability_level']}, review={row['needs_review']})")
