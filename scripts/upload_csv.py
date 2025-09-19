import pandas as pd
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
import chardet

# Load environment variables
load_dotenv()

def detect_encoding(file_path):
    """Detect the encoding of a file"""
    try:
        with open(file_path, 'rb') as file:
            raw_data = file.read(10000)  # Read first 10KB for detection
            result = chardet.detect(raw_data)
            return result['encoding']
    except Exception as e:
        print(f"Error detecting encoding: {e}")
        return 'utf-8'

def read_csv_safely(file_path):
    """Read CSV with proper encoding handling"""
    encodings_to_try = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']
    
    # First try to detect encoding
    detected_encoding = detect_encoding(file_path)
    if detected_encoding and detected_encoding not in encodings_to_try:
        encodings_to_try.insert(0, detected_encoding)
    
    for encoding in encodings_to_try:
        try:
            print(f"  Trying encoding: {encoding}")
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"  ✓ Successfully read with {encoding} encoding")
            return df
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"  Error with {encoding}: {e}")
            continue
    
    print(f"  ✗ Could not read file with any encoding")
    return None

def clean_dataframe(df, table_name):
    # Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.replace(' ', '_')
        .str.replace('-', '_')
        .str.replace('.', '_')
    )
    
    # Replace NaN/NaT with None (SQL NULL)
    df = df.where(pd.notnull(df), None)
    
    # Also convert empty strings "" to None → NULL
    df = df.applymap(lambda x: None if isinstance(x, str) and x.strip() == "" else x)

    print(f"  Cleaned columns: {list(df.columns)}")
    return df

def test_connection():
    """Test database connection"""
    try:
        connection_string = os.getenv('NEON_CONNECTION_STRING')
        if not connection_string:
            print("✗ NEON_CONNECTION_STRING not found in environment variables")
            return None
            
        engine = create_engine(connection_string)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✓ Database connection successful!")
            return engine
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return None

def upload_csv_files():
    """Upload all CSV files to Neon PostgreSQL"""
    engine = test_connection()
    if not engine:
        return
    
    csv_files = {
        'search': '../data/csv data/search.csv',
        'order_request': '../data/csv data/order_request.csv', 
        'order_status': '../data/csv data/order_status.csv',
        'subject': '../data/csv data/subject.csv',
        'package': '../data/csv data/package.csv',
        'company': '../data/csv data/company.csv',
        'search_status': '../data/csv data/search_status.csv',
        'search_type': '../data/csv data/search_type.csv'
    }
    
    successful_uploads = []
    failed_uploads = []
    
    for table_name, csv_path in csv_files.items():
        try:
            print(f"\n📁 Processing {csv_path}...")
            
            # Check if file exists
            if not os.path.exists(csv_path):
                print(f"✗ File not found: {csv_path}")
                failed_uploads.append((table_name, "File not found"))
                continue
            
            # Read CSV with encoding handling
            df = read_csv_safely(csv_path)
            if df is None:
                failed_uploads.append((table_name, "Could not read file"))
                continue
                
            print(f"📊 CSV loaded: {len(df)} rows, {len(df.columns)} columns")
            
            # Clean dataframe
            df = clean_dataframe(df, table_name)
            
            # Show sample data
            print(f"  Sample data:")
            print(f"  {df.head(1).to_string()}")
            
            # Upload to database
            df.to_sql(
                table_name,
                engine,
                if_exists='replace',
                index=False,
                method='multi',
                chunksize=1000
            )
            
            print(f"✓ Successfully uploaded {len(df)} rows to table '{table_name}'")
            successful_uploads.append(table_name)
            
        except Exception as e:
            print(f"✗ Error uploading {csv_path}: {e}")
            failed_uploads.append((table_name, str(e)))
    
    # Summary
    print(f"\n📋 Upload Summary:")
    print(f"✓ Successful: {len(successful_uploads)} tables")
    for table in successful_uploads:
        print(f"  - {table}")
    
    if failed_uploads:
        print(f"✗ Failed: {len(failed_uploads)} tables")
        for table, error in failed_uploads:
            print(f"  - {table}: {error}")

def verify_upload():
    """Verify that data was uploaded correctly"""
    engine = create_engine(os.getenv('NEON_CONNECTION_STRING'))
    tables = ['search', 'order_request','order_status', 'subject', 'package','company','search_status','search_type']
    
    print("\n🔍 Verifying uploaded data:")
    print("-" * 50)
    
    with engine.connect() as conn:
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                count = result.fetchone()[0]
                print(f"Table '{table}': {count} rows")
                
                # Get column info
                columns_result = conn.execute(text(f"SELECT * FROM {table}"))
                if columns_result.rowcount > 0:
                    columns = list(columns_result.keys())
                    print(f"  Columns: {columns[:3]}{'...' if len(columns) > 3 else ''}")
                else:
                    print(f"  No data in table")
                
            except Exception as e:
                print(f"Table '{table}': Error - {e}")

if __name__ == "__main__":
    print("🚀 Starting CSV upload to Neon PostgreSQL...")
    
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠️  .env file not found. Make sure you have NEON_CONNECTION_STRING set.")
    
    upload_csv_files()
    verify_upload()
    print("\n✅ Process completed!")