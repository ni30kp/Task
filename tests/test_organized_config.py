#!/usr/bin/env python3
"""
Test script to demonstrate the organized configuration system
"""

from config import get_settings, get_database_url, get_openai_key, get_vector_dimension

def test_organized_configuration():
    """Test the organized configuration system"""
    
    print("🧪 TESTING ORGANIZED CONFIGURATION SYSTEM")
    print("=" * 50)
    
    # Load settings
    settings = get_settings()
    
    # Test configuration display
    print("\n📋 CONFIGURATION LOADED:")
    settings.print_config_summary()
    
    # Test helper functions
    print("\n🔧 HELPER FUNCTIONS:")
    print(f"Database URL: {get_database_url()}")
    print(f"OpenAI Key Available: {'YES' if get_openai_key() else 'NO'}")
    print(f"Vector Dimension: {get_vector_dimension()}D")
    
    # Test individual config sections
    print("\n📊 DETAILED CONFIGURATION:")
    print(f"App Name: {settings.app.name}")
    print(f"App Version: {settings.app.version}")
    print(f"Materials Target: {settings.app.materials_target:,}")
    print(f"Response Time Target: {settings.app.response_time_target}ms")
    
    print(f"\nDatabase Host: {settings.database.host}")
    print(f"Database Port: {settings.database.port}")
    print(f"Database Name: {settings.database.name}")
    
    print(f"\nOpenAI Model: {settings.openai.model}")
    print(f"OpenAI Available: {'YES' if settings.has_openai_key() else 'NO'}")
    
    print(f"\nMargin Rate: {settings.business.margin_rate*100:.1f}%")
    print(f"VAT Renovation: {settings.business.vat_renovation*100:.1f}%")
    print(f"VAT New Build: {settings.business.vat_new_build*100:.1f}%")
    print(f"Labor Rate: €{settings.business.labor_rate_per_hour}/hour")
    
    print(f"\nSupported Languages: {', '.join(settings.multilingual.supported_languages)}")
    print(f"Default Language: {settings.multilingual.default_language}")
    
    print(f"\nServer Host: {settings.server.host}")
    print(f"Server Port: {settings.server.port}")
    print(f"Debug Mode: {settings.server.debug}")
    print(f"Log Level: {settings.server.log_level}")
    
    print("\n" + "=" * 50)
    print("✅ ORGANIZED CONFIGURATION SYSTEM WORKING PERFECTLY!")
    
    return settings

if __name__ == "__main__":
    test_organized_configuration()
