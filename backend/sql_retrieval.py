import os
from sqlalchemy import create_engine, text
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Optional, List
import json

load_dotenv()

class SQLRetrieval:
    def __init__(self):
        """Initialize SQL retrieval system with database connection and schema"""
        self.engine = create_engine(os.getenv('NEON_CONNECTION_STRING'))
        
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel("gemini-1.5-flash")

        # Load the exact database schema
        schema_file = os.path.join(os.path.dirname(__file__), 'database_schema.sql')
        try:
            with open(schema_file, 'r') as f:
                self.schema = f.read()
        except FileNotFoundError:
            print(f"Warning: Schema file not found at {schema_file}")
            self.schema = self._get_default_schema()
    
    def _get_default_schema(self) -> str:
        """Fallback schema if file not found"""
        return """
        -- Background Check System Schema
        -- Tables: Subject, Company, Package, Search_Type, Order_Request, Search, Search_status, Order_status
        """
    
    def process_sql_question(self, question: str) -> Dict:
        """
        Main entry point - processes SQL question through complete pipeline
        
        Args:
            question (str): Natural language question that requires SQL query
            
        Returns:
            Dict: Complete response with data, visualization, and explanation
        """
        print(f"🔍 Processing SQL Question: {question}")
        
        try:
            # Step 1: Convert natural language to SQL query
            sql_query = self._generate_sql_query(question)
            if not sql_query:
                return self._error_response("Failed to generate SQL query")
            
            print(f"📝 Generated SQL: {sql_query}")
            
            # Step 2: Execute SQL query against PostgreSQL database
            data_results = self._execute_sql_query(sql_query)
            if data_results is None:
                return self._error_response("Failed to execute SQL query")
            
            print(f"📊 Retrieved {len(data_results)} rows")
            
            # Step 3: Create visualization if applicable
            visualization = self._create_visualization(data_results, question)
            
            # Step 4: Generate LLM explanation of results
            explanation = self._generate_explanation(data_results, question, sql_query)
            
            # Step 5: Return formatted response
            return self._format_response(question, sql_query, data_results, visualization, explanation)
            
        except Exception as e:
            print(f"❌ Error in SQL processing: {e}")
            return self._error_response(f"SQL processing failed: {str(e)}")
    
    def _generate_sql_query(self, question: str) -> Optional[str]:
        """
        Convert natural language question to SQL query using LLM + schema
        
        Args:
            question (str): Natural language question
            
        Returns:
            str: Generated SQL query or None if failed
        """
        
        # Create detailed prompt with schema and business context
        prompt = f"""
You are a SQL expert working with a Background Check System database.  
Your task is to translate user questions written in natural language into **valid PostgreSQL SQL queries with columns in smallcase only**.
=== DATABASE SCHEMA ===
{self.schema}

=====================================================================
📚 DATABASE SCHEMA
=====================================================================
The database contains these main tables:

1. **Subject**
   - Holds candidate (person) details.
   - Key columns: subject_id, subject_name, subject_alias, subject_contact, subject_address

2. **Company**
   - Stores company information that requests background checks.
   - Key columns: comp_id, comp_name, comp_code

3. **Package**
   - Defines background check packages offered to companies.
   - Key columns: package_code, package_name, package_price, comp_code (links to Company)

4. **Search_Type**
   - Stores the types of result after background checks (criminal, employment, education, etc.).
   - Key columns: search_type_code, search_type_name

5. **Search**
   - Represents individual background checks conducted for subjects.
   - Key columns: search_id, package_req_id, subject_id, search_type_code, search_status

6. **Order_Request**
   - Represents orders placed by companies for background checks on subjects.
   - Key columns: order_id, order_PackageId, order_SubjectID, order_CompanyCode, Order_Status

7. **Search_status**
   - Lookup table for different search statuses.
   - Key columns: Status_code, Status (e.g., Pending, Completed, Failed)

8. **Order_status**
   - Lookup table for different order statuses.
   - Key columns: Status_code, Status (e.g.,CANCELLED,DRAFT,RECORD FOUND,NO RECORD FOUND,PENDING,
Awaiting County Search,QUALITY,AWAITING ACTION,RELEASE NEEDED,OTHER INFORMATION NEEDED,RECORD FOUND,
PENDING - TYPE 6,PENDING - TYPE 7,PENDING - TYPE 8,RECORD FOUND - TYPE 2)


=====================================================================
🔗 TABLE RELATIONSHIPS
=====================================================================
- Search.subject_id → Subject.subject_id
- Search.search_type_code → Search_Type.search_type_code
- Search.search_status → Search_status.Status_code
- Order_Request.order_SubjectID → Subject.subject_id
- Order_Request.Order_status → Order_status.Status_code
- Order_Request.order_CompanyCode → Company.comp_code
- Package.comp_code → Company.comp_code

=====================================================================
🏢 BUSINESS CONTEXT
=====================================================================
This database supports a **Background Check System** where:
- Companies request background checks (via `Order_Request`).
- Each order is tied to a company, package, subject, and has a status.
- Each order triggers one or more searches (via `Search`).
- Searches are of different types (criminal, education, etc.) and have statuses.
- Results help track candidate verification progress for companies.

=====================================================================
⚖️ SQL GENERATION RULES
=====================================================================
1. Use **EXACT table names and column names** (case-sensitive).
2. Always **JOIN related tables** for meaningful queries.
3. Use **WHERE clauses** for filtering conditions (e.g., by company, subject, status).
4. For large datasets, always add **LIMIT 100** unless user explicitly requests all.
5. For status-related queries, always JOIN with `Search_status` or `Order_status`.
6. Use **COUNT, SUM, AVG, GROUP BY** when aggregating data.
7. Do not invent columns or tables not present in the schema.
8. Return **only the SQL query** (no explanations, no extra text).

=====================================================================
📝 EXAMPLE QUERIES
=====================================================================
- "How many searches": SELECT COUNT(*) FROM Search
- "Pending searches": SELECT s.*, ss.Status FROM Search s JOIN Search_status ss ON s.search_status = ss.Status_code WHERE ss.Status LIKE '%pending%'
- "Subject details": SELECT s.*, sub.subject_name FROM Search s JOIN Subject sub ON s.subject_id = sub.subject_id
1."Show all pending searches with subject names."
    query:"SELECT s.search_id, sub.subject_name, ss.Status
    FROM Search s
    JOIN Subject sub ON s.subject_id = sub.subject_id
    JOIN Search_status ss ON s.search_status = ss.Status_code
    WHERE ss.Status = 'Pending'
    LIMIT 100;"
2."List all orders placed by each company with their status.":
    query:"SELECT c.comp_name, o.order_id, os.Status FROM Order_Request o
    JOIN Company c ON o.order_CompanyCode = c.comp_code JOIN Order_status os ON o.Order_status = os.Status_code
    LIMIT 100;"
User Question: "{question}"

SQL Query:
"""
        
        try:
            response = self.client.generate_content(prompt)
            
            sql_query = response.text.strip()
            
            # Clean up common formatting issues
            sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
            
            # Basic validation
            if not sql_query or not sql_query.upper().startswith('SELECT'):
                print(f"Invalid SQL generated: {sql_query}")
                return None
                
            return sql_query
            
        except Exception as e:
            print(f"Error generating SQL: {e}")
            return None
    
    def _execute_sql_query(self, sql_query: str) -> Optional[pd.DataFrame]:
        """
        Execute SQL query against PostgreSQL database with safety checks
        
        Args:
            sql_query (str): SQL query to execute
            
        Returns:
            pd.DataFrame: Query results or None if failed
        """
        
        try:
            # Safety checks to prevent dangerous operations
            sql_lower = sql_query.lower().strip()
            dangerous_operations = [
                'drop', 'delete', 'truncate', 'alter', 'create', 
                'insert', 'update', 'grant', 'revoke'
            ]
            
            if any(op in sql_lower for op in dangerous_operations):
                raise ValueError("Potentially dangerous SQL operation detected")
            
            # Execute query using pandas for easy data manipulation
            df = pd.read_sql(sql_query, self.engine)
            
            # Basic result validation
            if df.empty:
                print("Query returned no results")
                return pd.DataFrame()  # Return empty DataFrame instead of None
            
            return df
            
        except Exception as e:
            print(f"SQL execution error: {e}")
            return None
    
    def _create_visualization(self, df: pd.DataFrame, question: str) -> Optional[str]:
        """
        Create appropriate visualization based on data structure
        
        Args:
            df (pd.DataFrame): Query results
            question (str): Original question for context
            
        Returns:
            str: Plotly JSON string or None
        """
        
        if df is None or df.empty:
            return None
        
        try:
            # Determine visualization type based on data structure
            num_rows, num_cols = df.shape
            
            # Single metric (1 row, 1 column with numeric data)
            if num_rows == 1 and num_cols == 1 and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
                value = df.iloc[0, 0]
                fig = go.Figure(go.Indicator(
                    mode = "number",
                    value = value,
                    title = {"text": question},
                ))
                return fig.to_json()
            
            # Two columns with categorical + numeric (good for bar charts)
            elif num_cols == 2 and pd.api.types.is_numeric_dtype(df.iloc[:, 1]):
                fig = px.bar(
                    df, 
                    x=df.columns[0], 
                    y=df.columns[1],
                    title=f"Results: {question}",
                    labels={
                        df.columns[0]: df.columns[0].replace('_', ' ').title(),
                        df.columns[1]: df.columns[1].replace('_', ' ').title()
                    }
                )
                fig.update_layout(xaxis_tickangle=-45)
                return fig.to_json()
            
            # Status/category distribution (pie chart)
            elif 'status' in ' '.join(df.columns).lower() and num_cols <= 3:
                status_col = [col for col in df.columns if 'status' in col.lower()][0]
                if len(df[status_col].unique()) <= 10:  # Not too many categories
                    status_counts = df[status_col].value_counts()
                    fig = px.pie(
                        values=status_counts.values,
                        names=status_counts.index,
                        title=f"Distribution: {question}"
                    )
                    return fig.to_json()
            
            # Table for detailed data (multiple columns, reasonable row count)
            elif num_cols >= 3 and num_rows <= 20:
                fig = go.Figure(data=[go.Table(
                    header=dict(
                        values=[col.replace('_', ' ').title() for col in df.columns],
                        fill_color='lightblue',
                        align='left',
                        font=dict(size=12, color='white')
                    ),
                    cells=dict(
                        values=[df[col] for col in df.columns],
                        fill_color='lightgray',
                        align='left',
                        font=dict(size=11)
                    )
                )])
                fig.update_layout(title=f"Results: {question}")
                return fig.to_json()
            
            return None
            
        except Exception as e:
            print(f"Visualization error: {e}")
            return None
    
    def _generate_explanation(self, df: pd.DataFrame, question: str, sql_query: str) -> str:
        """
        Generate business-friendly explanation using LLM
        
        Args:
            df (pd.DataFrame): Query results
            question (str): Original question
            sql_query (str): Executed SQL query
            
        Returns:
            str: Natural language explanation
        """
        
        if df is None:
            return "The query could not be executed due to an error."
        
        if df.empty:
            return "No results were found for your query. This might indicate that there's no data matching your criteria, or you may need to adjust your search parameters."
        
        # Prepare data summary for LLM
        data_summary = f"""
Query Results Summary:
- Total rows: {len(df)}
- Columns: {', '.join(df.columns)}

Sample Data (first 3 rows):
{df.head(3).to_string()}

Data Types:
{df.dtypes.to_string()}
"""
        
        # Add statistical insights for numeric columns
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        if len(numeric_cols) > 0:
            data_summary += f"\n\nNumeric Column Statistics:\n{df[numeric_cols].describe().to_string()}"
        
        prompt = f"""
You are a business analyst for a background check company. Provide a clear, actionable explanation of these SQL query results.

Original Question: "{question}"
SQL Query: {sql_query}

{data_summary}

Provide a business-focused explanation that includes:
1. Key findings and insights
2. Business implications for background check operations
3. Notable patterns or trends
4. Actionable recommendations if applicable

Guidelines:
- Keep it under 200 words
- Use business language, avoid technical jargon
- Focus on what these results mean for business operations
- Be specific about numbers and trends
- If it's a single metric, explain its significance

Explanation:
"""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"Error generating explanation: {e}")
            # Fallback explanation
            if len(df) == 1 and len(df.columns) == 1:
                value = df.iloc[0, 0]
                return f"The query returned a single result: {value}. This metric provides insight into your background check operations."
            else:
                return f"The query returned {len(df)} records with {len(df.columns)} data points each, providing insights into your background check system."
    
    def _format_response(self, question: str, sql_query: str, data: pd.DataFrame, 
                        visualization: Optional[str], explanation: str) -> Dict:
        """
        Format the complete response for return to calling system
        
        Returns:
            Dict: Structured response with all components
        """
        return {
            "type": "sql",
            "success": True,
            "question": question,
            "sql_query": sql_query,
            "data": {
                "records": data.to_dict('records') if not data.empty else [],
                "columns": list(data.columns) if not data.empty else [],
                "row_count": len(data),
                "summary_stats": self._get_data_stats(data)
            },
            "visualization": visualization,
            "explanation": explanation,
            "metadata": {
                "execution_time": "success",
                "data_source": "postgresql",
                "schema_used": True
            }
        }
    
    def _error_response(self, error_message: str) -> Dict:
        """Create standardized error response"""
        return {
            "type": "sql",
            "success": False,
            "error": error_message,
            "data": {"records": [], "columns": [], "row_count": 0},
            "visualization": None,
            "explanation": f"I encountered an error while processing your SQL question: {error_message}",
            "metadata": {"execution_time": "failed"}
        }
    
    def _get_data_stats(self, df: pd.DataFrame) -> Dict:
        """Get basic statistics about the data"""
        if df.empty:
            return {}
        
        stats = {
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "numeric_columns": len(df.select_dtypes(include=['number']).columns),
            "text_columns": len(df.select_dtypes(include=['object']).columns)
        }
        
        # Add column info
        stats["columns_info"] = {
            col: str(df[col].dtype) for col in df.columns
        }
        
        return stats

# ========================================================================================
# MAIN ENTRY POINT FOR SQL QUESTIONS
# ========================================================================================

def handle_sql_question(question: str) -> Dict:
    """
    Main entry point for SQL questions coming from the router
    
    This function is called when the LLM router determines a question requires SQL processing
    
    Args:
        question (str): Natural language question that needs SQL processing
        
    Returns:
        Dict: Complete response with data, visualization, and explanation
    """
    sql_retriever = SQLRetrieval()
    return sql_retriever.process_sql_question(question)

# ========================================================================================
# TESTING FUNCTIONS
# ========================================================================================

def test_sql_retrieval():
    """Test the SQL retrieval system with sample questions"""
    print("🧪 Testing SQL Retrieval System")
    print("=" * 50)
    
    # Test questions for background check system
    test_questions = [
        #"How many total searches are in the database?",
        #"Show me the breakdown of searches by status",
        #"show me record found searches?",
        #"Show me overdue of urgent order?"
        #"What are the top 5 most expensive packages?",
        #"How many orders does each company have?",
        "Show me all search types available",
        #"Which subjects have the most searches?",
        #"What are the different order statuses?",
        #"Show me recent searches with subject names"
        #"How many orders are pending?"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Testing: {question}")
        
        try:
            result = handle_sql_question(question)
            
            if result["success"]:
                print(f"   ✅ Success: {result['data']['row_count']} rows")
                print(f"   📝 SQL: {result['sql_query'][:80]}...")
                print(f"   💡 Explanation: {result['explanation'][:100]}...")
            else:
                print(f"   ❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"   💥 Exception: {e}")
    
    print(f"\n✅ SQL retrieval testing completed!")

if __name__ == "__main__":
    # Run tests when file is executed directly
    test_sql_retrieval()