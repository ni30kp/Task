#!/usr/bin/env python3
"""
Test multilingual functionality of the production system
"""

import requests
import json

def test_multilingual_responses():
    """Test multilingual responses for different languages"""
    
    base_url = "http://localhost:8000"
    
    test_queries = [
        {
            "query": "colle pour carrelage salle de bain mur blanc",
            "language": "French",
            "expected_terms": ["colle", "étanche", "carreaux", "céramique"]
        },
        {
            "query": "waterproof glue for bathroom tiles",
            "language": "English", 
            "expected_terms": ["waterproof", "adhesive", "tiles", "ceramic"]
        },
        {
            "query": "adhesivo para azulejos de baño",
            "language": "Spanish",
            "expected_terms": ["impermeable", "adhesivo", "azulejos"]
        }
    ]
    
    print("🌍 TESTING MULTILINGUAL SEMANTIC SEARCH")
    print("=" * 50)
    
    for test in test_queries:
        print(f"\n🔍 Testing {test['language']} Query:")
        print(f"Query: \"{test['query']}\"")
        
        try:
            response = requests.get(f"{base_url}/material-price", params={
                "query": test["query"],
                "limit": 2
            })
            
            if response.status_code == 200:
                results = response.json()
                
                if results:
                    print(f"✅ Found {len(results)} results")
                    
                    for i, material in enumerate(results[:1], 1):
                        print(f"\n{i}. {material['material_name']}")
                        print(f"   Description: {material['description']}")
                        print(f"   Confidence: {material['confidence_tier']} ({material['similarity_score']*100:.1f}%)")
                        print(f"   Price: €{material['unit_price']}/{material['unit']}")
                        print(f"   Region: {material['region']}")
                    
                    # Check for localization
                    material = results[0]
                    localized_name = any(term in material['material_name'].lower() for term in test['expected_terms'])
                    localized_desc = any(term in material['description'].lower() for term in test['expected_terms'])
                    
                    print(f"\n📊 Localization Status:")
                    print(f"   Name localized: {'✅ YES' if localized_name else '⚠️  PARTIAL'}")
                    print(f"   Description localized: {'✅ YES' if localized_desc else '⚠️  PARTIAL'}")
                    
                else:
                    print("❌ No results found")
            else:
                print(f"❌ Request failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 MULTILINGUAL TEST COMPLETE")

if __name__ == "__main__":
    test_multilingual_responses()
