#!/usr/bin/env python3
"""
Quick test for the fixed SQL generation
"""

import sys
import os
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent / "backend"
sys.path.insert(0, str(backend_path))

def test_sql_generation():
    """Test the fixed SQL generation"""
    try:
        from advanced_query_processor import AdvancedQueryProcessor
        
        processor = AdvancedQueryProcessor()
        
        # Test query that was failing
        test_query = "Show me the top 5 companies by number of orders"
        
        print(f"🔍 Testing query: {test_query}")
        
        # Generate SQL
        sql = processor._generate_sql_query(test_query)
        
        if sql:
            print(f"✅ Generated SQL:")
            print(sql)
            print()
            
            # Check for correct column names
            if "order_company_code" in sql.lower():
                print("✅ Correct column name used: order_company_code")
            else:
                print("❌ Incorrect column name - should use order_company_code")
                
            if sql.lower().startswith("select"):
                print("✅ Valid SELECT query")
            else:
                print("❌ Invalid query format")
                
        else:
            print("❌ Failed to generate SQL")
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Fixed SQL Generation")
    print("=" * 40)
    test_sql_generation()