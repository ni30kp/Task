#!/usr/bin/env python3
"""
Live API Testing for Donizo Semantic Pricing Engine
Test all endpoints with real scenarios
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test basic connectivity"""
    print("🏥 HEALTH CHECK")
    print("=" * 50)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    health = requests.get(f"{BASE_URL}/health")
    print(f"\nHealth Status: {health.status_code}")
    print(f"Health Response: {json.dumps(health.json(), indent=2)}")
    
    return response.status_code == 200

def test_material_search():
    """Test the /material-price endpoint"""
    print("\n🔍 MATERIAL SEARCH API TESTS")
    print("=" * 50)
    
    test_cases = [
        {
            "name": "Basic Search",
            "params": {"query": "white ceramic tiles bathroom"}
        },
        {
            "name": "Waterproof Adhesive",
            "params": {"query": "waterproof adhesive strong", "limit": 3}
        },
        {
            "name": "Regional Filter",
            "params": {"query": "paint wall", "region": "Île-de-France", "limit": 2}
        },
        {
            "name": "Quality Filter",
            "params": {"query": "tiles", "quality_min": 4, "limit": 3}
        },
        {
            "name": "French Query",
            "params": {"query": "colle carrelage salle de bain", "limit": 2}
        },
        {
            "name": "Vague Query",
            "params": {"query": "something for outdoor cement-ish", "limit": 2}
        }
    ]
    
    success_count = 0
    
    for test_case in test_cases:
        print(f"\n🧪 {test_case['name']}")
        start_time = time.time()
        
        try:
            response = requests.get(f"{BASE_URL}/material-price", params=test_case['params'])
            response_time = (time.time() - start_time) * 1000
            
            print(f"⚡ Response time: {response_time:.1f}ms")
            print(f"📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                results = response.json()
                print(f"📋 Found {len(results)} materials")
                
                if results:
                    top_result = results[0]
                    print(f"🎯 Top result: {top_result['material_name']}")
                    print(f"💰 Price: {top_result['unit_price']} {top_result['unit']}")
                    print(f"📍 Region: {top_result['region']}")
                    print(f"🎯 Confidence: {top_result['confidence_tier']} ({top_result['similarity_score']})")
                
                success_count += 1
            else:
                error = response.json()
                print(f"❌ Error: {error.get('detail', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print(f"\n✅ Search API: {success_count}/{len(test_cases)} tests passed")
    return success_count == len(test_cases)

def test_quote_generation():
    """Test the /generate-proposal endpoint"""
    print("\n💰 QUOTE GENERATION API TESTS")
    print("=" * 50)
    
    test_scenarios = [
        {
            "name": "Bathroom Renovation",
            "data": {
                "transcript": "Need waterproof glue and 60x60cm matte white wall tiles, better quality this time. For bathroom walls in Paris",
                "region": "Île-de-France"
            }
        },
        {
            "name": "Living Room Paint",
            "data": {
                "transcript": "I want to paint the living room walls, maybe 30 square meters. Something durable and washable. I'm in Marseille.",
                "region": "Provence-Alpes-Côte d'Azur"
            }
        },
        {
            "name": "General Renovation",
            "data": {
                "transcript": "Need materials for kitchen renovation, tiles and some adhesive",
                "region": "Belgium"
            }
        }
    ]
    
    success_count = 0
    generated_quotes = []
    
    for scenario in test_scenarios:
        print(f"\n🧪 {scenario['name']}")
        start_time = time.time()
        
        try:
            response = requests.post(f"{BASE_URL}/generate-proposal", json=scenario['data'])
            response_time = (time.time() - start_time) * 1000
            
            print(f"⚡ Response time: {response_time:.1f}ms")
            print(f"📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                quote = response.json()
                generated_quotes.append(quote)
                
                print(f"📋 Quote ID: {quote['quote_id']}")
                print(f"💰 Total: €{quote['total_estimate']}")
                print(f"📊 Confidence: {quote['confidence_score']}")
                print(f"🏷️  VAT Rate: {quote['vat_rate']*100:.0f}%")
                print(f"📍 Region: {quote['region']}")
                print(f"⏱️  Duration: {quote['estimated_duration']}")
                
                print(f"📋 Tasks ({len(quote['tasks'])}):")
                for task in quote['tasks']:
                    print(f"  • {task['label']}: €{task['margin_protected_price']}")
                    print(f"    Materials: {len(task['materials'])}, Labor: {task['labor_hours']}h")
                
                success_count += 1
            else:
                error = response.json()
                print(f"❌ Error: {error.get('detail', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print(f"\n✅ Quote API: {success_count}/{len(test_scenarios)} tests passed")
    return success_count == len(test_scenarios), generated_quotes

def test_feedback_system(quotes):
    """Test the /feedback endpoint"""
    print("\n📝 FEEDBACK SYSTEM API TESTS")
    print("=" * 50)
    
    if not quotes:
        print("⚠️  No quotes available for feedback testing")
        return False
    
    feedback_scenarios = [
        {
            "name": "Accurate Quote",
            "data": {
                "quote_id": quotes[0]['quote_id'],
                "user_type": "contractor",
                "verdict": "accurate",
                "comment": "Quote was very accurate, materials were exactly what I needed"
            }
        },
        {
            "name": "Overpriced Feedback",
            "data": {
                "quote_id": quotes[0]['quote_id'],
                "user_type": "client",
                "verdict": "overpriced", 
                "comment": "Seems a bit expensive for this region"
            }
        }
    ]
    
    if len(quotes) > 1:
        feedback_scenarios.append({
            "name": "Wrong Material",
            "data": {
                "quote_id": quotes[1]['quote_id'],
                "user_type": "contractor",
                "verdict": "wrong_material",
                "comment": "Suggested material not suitable for this application"
            }
        })
    
    success_count = 0
    
    for scenario in feedback_scenarios:
        print(f"\n🧪 {scenario['name']}")
        
        try:
            response = requests.post(f"{BASE_URL}/feedback", json=scenario['data'])
            
            print(f"📊 Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                print(f"✅ Status: {result['status']}")
                print(f"📋 Feedback ID: {result['feedback_id']}")
                print(f"💡 Message: {result['message']}")
                print(f"📈 Learning Impact: {result['learning_impact']}")
                print(f"🔧 Improvement Actions ({len(result['improvement_actions'])}):")
                
                for action in result['improvement_actions']:
                    print(f"  • {action}")
                
                success_count += 1
            else:
                error = response.json()
                print(f"❌ Error: {error.get('detail', 'Unknown error')}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")
    
    print(f"\n✅ Feedback API: {success_count}/{len(feedback_scenarios)} tests passed")
    return success_count == len(feedback_scenarios)

def test_stats_endpoint():
    """Test the /stats endpoint"""
    print("\n📊 SYSTEM STATISTICS")
    print("=" * 50)
    
    try:
        response = requests.get(f"{BASE_URL}/stats")
        
        if response.status_code == 200:
            stats = response.json()
            
            print(f"📦 Materials: {stats['materials']['total']}")
            print(f"💰 Quotes: {stats['quotes']['total']}")
            print(f"📝 Feedback: {stats['feedback']['total']}")
            
            if stats['quotes']['total'] > 0:
                print(f"💵 Total Quote Value: €{stats['quotes']['total_value']:.2f}")
                print(f"📊 Average Quote: €{stats['quotes']['average_value']:.2f}")
            
            print("\n🏷️  Top Categories:")
            for category, count in list(stats['materials']['by_category'].items())[:5]:
                print(f"  • {category}: {count}")
            
            print("\n🌍 Top Regions:")
            for region, count in list(stats['materials']['by_region'].items())[:5]:
                print(f"  • {region}: {count}")
            
            return True
        else:
            print(f"❌ Stats error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Stats exception: {e}")
        return False

def test_performance_benchmark():
    """Test performance requirements"""
    print("\n⚡ PERFORMANCE BENCHMARK")
    print("=" * 50)
    
    # Test search performance
    search_times = []
    for i in range(10):
        start_time = time.time()
        response = requests.get(f"{BASE_URL}/material-price", params={"query": "tiles bathroom"})
        if response.status_code == 200:
            search_times.append((time.time() - start_time) * 1000)
    
    # Test quote performance
    quote_times = []
    for i in range(5):
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/generate-proposal", json={
            "transcript": "Need materials for renovation",
            "region": "Île-de-France"
        })
        if response.status_code == 200:
            quote_times.append((time.time() - start_time) * 1000)
    
    if search_times and quote_times:
        avg_search = sum(search_times) / len(search_times)
        avg_quote = sum(quote_times) / len(quote_times)
        
        print(f"🔍 Average search time: {avg_search:.1f}ms")
        print(f"💰 Average quote time: {avg_quote:.1f}ms")
        
        search_pass = avg_search < 500
        quote_pass = avg_quote < 500
        
        print(f"✅ Search <500ms: {'PASS' if search_pass else 'FAIL'}")
        print(f"✅ Quote <500ms: {'PASS' if quote_pass else 'FAIL'}")
        
        return search_pass and quote_pass
    
    return False

def main():
    """Run comprehensive API testing"""
    print("🧪 DONIZO SEMANTIC PRICING ENGINE - LIVE API TESTING")
    print("Testing production-level application with real scenarios")
    print("=" * 70)
    
    start_time = time.time()
    
    # Run all tests
    tests_passed = 0
    total_tests = 6
    
    if test_health_check():
        tests_passed += 1
    
    if test_material_search():
        tests_passed += 1
    
    quote_success, quotes = test_quote_generation()
    if quote_success:
        tests_passed += 1
    
    if test_feedback_system(quotes):
        tests_passed += 1
    
    if test_stats_endpoint():
        tests_passed += 1
    
    if test_performance_benchmark():
        tests_passed += 1
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("🎉 API TESTING COMPLETE!")
    print(f"⚡ Total time: {total_time:.1f}s")
    print(f"✅ Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🚀 ALL TESTS PASSED - SYSTEM FULLY OPERATIONAL!")
        print("🌍 Ready to power pricing for every renovation job on Earth!")
    else:
        print(f"\n⚠️  Some tests failed. Check logs above.")
    
    print(f"\n🔗 Interactive API Documentation: {BASE_URL}/docs")
    print(f"🏥 Health Check: {BASE_URL}/health")
    print(f"📊 System Stats: {BASE_URL}/stats")

if __name__ == "__main__":
    main()
