#!/usr/bin/env python
"""Quick test to verify Salbutamol alias fix"""
from app.services.drug_lookup import init_store, lookup_drug

init_store()
result = lookup_drug('salbutamol', disease='asthma', age=28)

print('=== SALBUTAMOL LOOKUP TEST ===')
print(f'Query: {result["query"]}')
print(f'Match Type: {result["match_type"]}')
print(f'Matches Found: {len(result["matches"])}')

if result['matches']:
    m = result['matches'][0]
    print(f'\n✓ SUCCESS: Found match!')
    print(f'  Generic Name: {m["generic_name"]}')
    print(f'  Score: {m["score"]:.1f}%')
    print(f'  Indications: {m["indications"][:150]}...')
else:
    print('\n✗ FAILED: No matches found')
    print(f'  Suggestions: {result["suggestions"]}')
