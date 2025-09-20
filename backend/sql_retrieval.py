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
            res=self._format_response(question, sql_query, data_results, visualization, explanation)
            print(res)
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
Your task is to translate user questions written in natural language into **valid PostgreSQL SQL queries with columns name in smallcase only**.
=== DATABASE SCHEMA ===
{self.schema}

=====================================================================
📚 DATABASE SCHEMA
=====================================================================
The database contains these main tables:

1. **Subject**
   - Holds candidate (person) details.
   - Key columns: subject_id, subject_name, subject_alias, subject_contact, subject_address
    sample values in this table are:
        subject_id,subject_name,subject_alias,subject_contact,subject_address1,subject_address2,sbj_city
        4832213,Maximillian Cains,,7639898647,18 Kahuaina Place,,Cromwell
        5058939,Sandi Stimer,,6318854424,603 Contees St.,,Salem
        5095445,Natividad Tito,,,14 Dews St.,,Marlin
        5208968,Jane Acerno,,5121986942,91 Puerto Rd.,,Vicksburg
        5252901,Fedele Willock,,2079349344,567 Corkins Rd.,#411,Carrollton
   
2. **Company**
   - Stores company information that requests background checks.
   - Key columns: comp_id, comp_name, comp_code
    sample values in this table are:
        comp_id,comp_name,comp_code
        8763,"Abbvie, Inc.",ABVIE
        6600,Activision International,ACTIT
        10030,AIDS Healthcare Foundation,AHF
        5082,Alliance Healthcare Services,AHS

3. **Package**
   - Defines background check packages offered to companies.
   - Key columns: package_code, package_name, package_price, comp_code (links to Company)
    sample values in this table are:
        package_code,package_name,package_price,comp_code
        41,GEO Group Package Three,46.69,GEO
        2227,General Package,18.15,AMZTA
        2318,Retail Associates,21.14,ROSS
        3056,Medical Verification Package,45.7,AMAZT
        3870,Salary Package,38.9,ROCKT
        3872,Hourly Package,16.64,ROCKT

4. **Search_Type**
   - Stores the types of result after background checks (criminal, employment, education, etc.).
   - Key columns: search_type_code, search_type_name
    sample values in this table are:
        search_type_code,search_type,search_type_category
        9PK,CUSTOMIZED PACKAGE,PKG
        ADJ,ADJUDICATION,G
        CBSV,CONSENT-BASED SOCIAL SECURITY NUMBER VERIFICATION,G
        CRTU,CREDIT REPORT- TRANS UNION,G
        D10,10 PANEL LAB DRUG TEST,G
        D10A,10 PANEL + ADULTERANT,G
        DH5,HAIR 5 PANEL DRUG TEST,G
5. **Search**
   - Represents individual background checks conducted for subjects.
   - Key columns: search_id, package_req_id, subject_id, search_type_code, search_status
    sample values in this table are:
        search_id,package_req_id,subject_id,search_type_code,search_status,county_name,state_code,pkg_code,sub_status
        25692215,Y4716269,4832213,MVR,R,NONE,IN,5923,
        27123290,Y4934241,5058939,EMP,R,NONE,,5805,Discrepancy Found
        27368764,Y4969373,5095445,ADJ,R,NONE,,41,
        28112714,Y5079200,5208968,SON,N,NONE,,2227,
        28112715,Y5079200,5208968,OFAC,N,NONE,,2227,
        28112716,Y5079200,5208968,ADJ,R,NONE,,2227,
        28112718,Y5079200,5208968,NCRIM,R,NONE,,2227,
        28416276,Y5121814,5252901,ADJ,R,NONE,,7194,
        29022564,Y5207165,5340770,EMP,R,NONE,,5367,Developed Information
6. **Order_Request**
   - Represents orders placed by companies for background checks on subjects.
   - Key columns: order_id, order_Package_id, order_Subject_id, order_company_code, order_status
    sample values in this table are:
        order_id,order_package_id,order_subject_id,order_company_code,order_status,order_packcage_code,order_date,estimated_completion,actual_completion,days_elapsed,is_overdue,priority_level
        1040815,Y4716269,4832213,NEST,P,5923,20:47.4,20:47.4,,79,TRUE,Standard
        1258599,Y4934241,5058939,TFS,P,5805,20:47.4,20:47.4,,57,TRUE,Standard
        1293667,Y4969373,5095445,GEO,P,41,20:47.4,20:47.4,,19,TRUE,Standard
        1403444,Y5079200,5208968,AMZTA,P,2227,20:47.4,20:47.4,,49,TRUE,Standard
        1446058,Y5121814,5252901,ACTIT,D,7194,20:47.4,20:47.4,20:47.4,60,FALSE,Rush
        1531402,Y5207165,5340770,TREM,P,5367,20:47.4,20:47.4,,118,TRUE,Urgent
        3574933,Y7251702,7419549,AMZSF,P,2227,20:47.4,20:47.4,,89,TRUE,Standard
        3588649,Y7265424,7433399,TMOB,D,12069,20:47.4,20:47.4,20:47.4,43,FALSE,Standard
        3713223,Y7390015,7559643,AMZSF,P,2227,20:47.4,20:47.4,,9,TRUE,Urgent
7. **Search_status**
   - Lookup table for different search statuses.
   - Key columns: status_code, status (e.g., Pending, Completed, Failed)
    sample values in this table are:
        status_code,status
        C,CANCELLED
        D,DRAFT
        F,RECORD FOUND
        N,NO RECORD FOUND
        P,PENDING
        P11,Awaiting County Search
        P13,QUALITY
        P14,AWAITING ACTION
        P4,RELEASE NEEDED
        P5,OTHER INFORMATION NEEDED
        R,RECORD FOUND
        P6,PENDING - TYPE 6
        P7,PENDING - TYPE 7
        P8,PENDING - TYPE 8
        R2,RECORD FOUND - TYPE 2

8. **Order_status**
   - Lookup table for different order statuses.
   - Key columns: Status_code, Status 
    sample values in this table are:
        status_code,status_description
        P,PENDING
        D,COMPLETED


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
            chat = self.client.start_chat(history=[])
            response = chat.send_message(
            prompt,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 300
            }
            )
            return response.text.strip()
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

def generate_sql_query(question: str) -> str:
    """
    Standalone function to generate SQL query from natural language question
    
    This function is used by processuserquery.py to generate SQL queries
    
    Args:
        question (str): Natural language question
        
    Returns:
        str: Generated SQL query or None if failed
    """
    sql_retriever = SQLRetrieval()
    return sql_retriever._generate_sql_query(question)

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
        "What are the top 5 most expensive packages?",
        #"How many orders does each company have?",
        #"Show me all search types available",
        #"Which subjects have the most searches?",
        #"What are the different order statuses?",
        #"Show me recent searches with subject names"
        #"How many orders are pending?"
        #"summarize last years order"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n{i}. Testing: {question}")
        
        try:
            result = handle_sql_question(question)
            
            if result.get("success"):
                print("\n" + "="*60)
                print(f"📋 QUESTION: {result['question']}")
                print(f"📝 SQL QUERY: {result['sql_query']}")
                print("="*60)
                
                # Display the data table
                data_records = result['data']['records']
                if data_records:
                    df = pd.DataFrame(data_records)
                    print("📊 QUERY RESULTS:")
                    print(df.to_string(index=False))
                    print(f"\n📈 Total rows: {len(df)}")
                    print(data_records)
                else:
                    print("📊 No data returned")
                
                # Display explanation
                print(f"\n💡 EXPLANATION:")
                print(result['explanation'])
                print("="*60)
            else:
                print(f"   ❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"   💥 Exception: {e}")
    
    print(f"\n✅ SQL retrieval testing completed!")

if __name__ == "__main__":
    # Run tests when file is executed directly
    test_sql_retrieval()