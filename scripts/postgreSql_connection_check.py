from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv()

def test_neon_connection():
    """Simple test to verify Neon connection works"""
    
    connection_string = os.getenv('NEON_CONNECTION_STRING')
    print(f"🔗 Testing connection...")
    print(f"Host: ep-small-smoke-a1bdr2u3-pooler.ap-southeast-1.aws.neon.tech")
    
    try:
        # Create engine
        engine = create_engine(connection_string)
        
        # Test connection
        with engine.connect() as conn:
            # Test basic query
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]
            print(f"✓ Connected successfully!")
            print(f"PostgreSQL version: {version}")
            
            # List existing tables
            result = conn.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result.fetchall()]
            print(f"Existing tables: {tables}")
            
    except Exception as e:
        print(f"✗ Connection failed: {e}")

if __name__ == "__main__":
    test_neon_connection()