import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_connection():
    """Get database connection"""
    try:
        connection_string = os.getenv('NEON_CONNECTION_STRING')
        if not connection_string:
            print("✗ NEON_CONNECTION_STRING not found in environment variables")
            return None
            
        engine = create_engine(connection_string)
        return engine
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return None

def test_basic_connection():
    """Test basic database connection"""
    print("🔗 Testing basic database connection...")
    engine = get_connection()
    if not engine:
        return False
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✓ Connected successfully!")
            print(f"PostgreSQL version: {version[:50]}...")
            return True
    except Exception as e:
        print(f"✗ Connection test failed: {e}")
        return False

def list_tables():
    """List all tables in the database"""
    print("\n📋 Listing all tables...")
    engine = get_connection()
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT table_name, 
                       (SELECT COUNT(*) FROM information_schema.columns 
                        WHERE table_name = t.table_name AND table_schema = 'public') as column_count
                FROM information_schema.tables t
                WHERE table_schema = 'public'
                ORDER BY table_name
            """))
            
            tables = result.fetchall()
            if tables:
                print("Available tables:")
                for table_name, col_count in tables:
                    print(f"  - {table_name} ({col_count} columns)")
                return [table[0] for table in tables]
            else:
                print("No tables found in the database")
                return []
                
    except Exception as e:
        print(f"✗ Error listing tables: {e}")
        return []

def get_table_info(table_name):
    """Get detailed information about a specific table"""
    print(f"\n🔍 Getting info for table '{table_name}'...")
    engine = get_connection()
    
    try:
        with engine.connect() as conn:
            # Get column information
            columns_query = text("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = :table_name AND table_schema = 'public'
                ORDER BY ordinal_position
            """)
            
            columns_result = conn.execute(columns_query, {"table_name": table_name})
            columns = columns_result.fetchall()
            
            if columns:
                print(f"Columns in '{table_name}':")
                for col_name, data_type, nullable in columns:
                    null_info = "NULL" if nullable == "YES" else "NOT NULL"
                    print(f"  - {col_name}: {data_type} ({null_info})")
            
            # Get row count
            count_result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = count_result.fetchone()[0]
            print(f"Total rows: {row_count}")
            
            return True
            
    except Exception as e:
        print(f"✗ Error getting table info: {e}")
        return False

def sample_data_from_table(table_name, limit=5):
    """Get sample data from a table"""
    print(f"\n📊 Sample data from '{table_name}' (first {limit} rows)...")
    engine = get_connection()
    
    try:
        # Use pandas for better formatting
        query = f"SELECT * FROM {table_name} LIMIT {limit}"
        df = pd.read_sql(query, engine)
        
        if not df.empty:
            print(df.to_string(index=False))
            return df
        else:
            print("No data found in table")
            return None
            
    except Exception as e:
        print(f"✗ Error retrieving sample data: {e}")
        return None

def test_specific_queries():
    """Test specific business logic queries"""
    print("\n🎯 Testing specific business queries...")
    engine = get_connection()
    
    queries = {
        "Total searches by status": """
            SELECT search_status, COUNT(*) as count
            FROM search
            GROUP BY search_status
            ORDER BY count DESC
        """,
        
        "Recent orders": """
            SELECT *
            FROM "order"
            ORDER BY order_id DESC
            LIMIT 5
        """,
        
        "Package information": """
            SELECT package_code, package_name, package_price
            FROM package
            ORDER BY package_price DESC
            LIMIT 5
        """,
        
        "Search types": """
            SELECT search_type_code, search_type_name, search_category
            FROM search_type
            LIMIT 10
        """
    }
    
    for query_name, query in queries.items():
        try:
            print(f"\n📈 {query_name}:")
            df = pd.read_sql(query, engine)
            if not df.empty:
                print(df.to_string(index=False))
            else:
                print("No results found")
        except Exception as e:
            print(f"✗ Error in query '{query_name}': {e}")

def interactive_query():
    """Allow user to run custom queries"""
    print("\n💬 Interactive Query Mode")
    print("Enter SQL queries (type 'exit' to quit):")
    
    engine = get_connection()
    
    while True:
        try:
            query = input("\nSQL> ").strip()
            
            if query.lower() in ['exit', 'quit', 'q']:
                break
                
            if not query:
                continue
                
            # Basic safety check
            if any(dangerous in query.lower() for dangerous in ['drop', 'delete', 'truncate', 'update', 'insert']):
                print("⚠️  Dangerous operations not allowed in this mode")
                continue
            
            df = pd.read_sql(query, engine)
            if not df.empty:
                print(df.to_string(index=False))
            else:
                print("Query executed successfully (no results)")
                
        except KeyboardInterrupt:
            print("\n👋 Exiting interactive mode...")
            break
        except Exception as e:
            print(f"✗ Query error: {e}")

def main():
    """Main function to run all tests"""
    print("🚀 PostgreSQL Data Retrieval Test")
    print("=" * 50)
    
    # Test connection
    if not test_basic_connection():
        return
    
    # List tables
    tables = list_tables()
    if not tables:
        return
    
    # Get info for each table
    for table in tables[:3]:  # Limit to first 3 tables
        get_table_info(table)
        sample_data_from_table(table, 3)
    
    # Test specific queries
    test_specific_queries()
    
    # Ask if user wants interactive mode
    print("\n" + "=" * 50)
    choice = input("Do you want to enter interactive query mode? (y/n): ").lower()
    if choice in ['y', 'yes']:
        interactive_query()
    
    print("\n✅ Data retrieval test completed!")

if __name__ == "__main__":
    main()