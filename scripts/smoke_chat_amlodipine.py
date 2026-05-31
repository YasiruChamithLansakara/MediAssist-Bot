from app.services.chat_service import build_chat_response
import json

r = build_chat_response(message='is it dangerous to take this?', disease='hypertension', age=52, drugs=['amlodipine'])
print('---ANSWER---')
print(r['answer'])
print('\n---MATCHED---')
print(json.dumps([m.get('best_match', {}).get('generic_name_clean') or m.get('query') for m in r['matched_drugs']], indent=2))
