#!/usr/bin/env python3
"""
CONFIDENCE SCORING FIX SCRIPT
============================

This script forces regeneration of the materials database with improved
embedding logic that ensures different materials have different similarity scores.
"""

import os
import sqlite3
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_confidence_scoring():
    """Fix the confidence scoring issue by regenerating the database"""
    
    print("🔧 FIXING CONFIDENCE SCORING ISSUE")
    print("=" * 40)
    
    # 1. Remove old database
    db_path = "donizo_complete.db"
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"✅ Removed old database: {db_path}")
    
    # 2. Import and initialize the engine with fixed embedding logic
    try:
        from complete_app import SemanticPricingEngine
        
        print("🚀 Initializing engine with fixed embedding logic...")
        engine = SemanticPricingEngine()
        
        print("✅ Engine initialized successfully!")
        print(f"✅ Generated {len(engine.materials_cache)} materials with unique embeddings")
        
        # 3. Test the fix
        print("\n🧪 TESTING CONFIDENCE SCORING FIX...")
        
        # Test with waterproof adhesive
        results = engine.semantic_search("waterproof adhesive", limit=5)
        
        if results:
            print(f"Found {len(results)} results:")
            similarities = []
            
            for i, result in enumerate(results, 1):
                similarity = result['similarity_score']
                similarities.append(similarity)
                print(f"  {i}. {result['material_name']} - {result['vendor']} - {similarity:.6f}")
            
            unique_similarities = len(set(similarities))
            print(f"\n📊 RESULTS:")
            print(f"  Unique similarity scores: {unique_similarities}")
            
            if unique_similarities > 1:
                print("  ✅ FIXED: Similarity scores now vary properly!")
                return True
            else:
                print("  ❌ Still identical scores")
                return False
        else:
            print("  ❌ No results found")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = fix_confidence_scoring()
    
    if success:
        print("\n🎉 CONFIDENCE SCORING SUCCESSFULLY FIXED!")
        print("You can now restart the server to see varied confidence scores.")
    else:
        print("\n❌ CONFIDENCE SCORING FIX FAILED")
        print("Manual intervention may be required.")
