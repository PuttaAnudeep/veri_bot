import os
import json
import pandas as pd
import google.generativeai as genai
from sqlalchemy import create_engine
from dotenv import load_dotenv
from typing import Dict, Optional, List, Tuple, Any
import traceback
import re
import time
import random
from functools import wraps

load_dotenv()

def llm_retry_with_rate_limit(max_retries: int = 3, base_delay: float = 3.0):
    """
    Decorator for LLM calls with retry logic and rate limiting
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds (will add randomization)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    # Add rate limiting delay (3-4 seconds with randomization)
                    if attempt > 0:  # Don't delay on first attempt
                        delay = base_delay + random.uniform(0.5, 1.5)  # 3-4.5 seconds
                        print(f"Waiting {delay:.1f}s before retry attempt {attempt}...")
                        time.sleep(delay)
                    elif hasattr(wrapper, '_last_call_time'):
                        # Ensure minimum gap between calls
                        time_since_last = time.time() - wrapper._last_call_time
                        if time_since_last < base_delay:
                            sleep_time = base_delay - time_since_last + random.uniform(0.2, 0.8)
                            print(f"Rate limiting: waiting {sleep_time:.1f}s...")
                            time.sleep(sleep_time)
                    
                    # Record call time
                    wrapper._last_call_time = time.time()
                    
                    # Make the actual LLM call
                    result = func(*args, **kwargs)
                    
                    if attempt > 0:
                        print(f"LLM call succeeded on attempt {attempt + 1}")
                    
                    return result
                    
                except Exception as e:
                    last_exception = e
                    print(f"LLM call attempt {attempt + 1} failed: {str(e)}")
                    
                    if attempt == max_retries:
                        break
                    
                    # Exponential backoff for retries
                    if attempt > 0:
                        backoff_delay = base_delay * (2 ** attempt) + random.uniform(0.5, 1.5)
                        print(f"Backing off for {backoff_delay:.1f}s before next retry...")
                        time.sleep(backoff_delay)
            
            print(f"All {max_retries + 1} LLM call attempts failed. Last error: {last_exception}")
            raise last_exception or Exception("LLM call failed after all retries")
        
        return wrapper
    return decorator

class AdvancedQueryProcessor:
    def __init__(self):
        """Initialize the advanced query processing system"""
        self.engine = create_engine(os.getenv('NEON_CONNECTION_STRING'))
        
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel("gemini-1.5-flash")
        
        # Load database schema
        schema_file = os.path.join(os.path.dirname(__file__), 'database_schema.sql')
        try:
            with open(schema_file, 'r') as f:
                self.schema = f.read()
        except FileNotFoundError:
            print(f"Warning: Schema file not found at {schema_file}")
            self.schema = self._get_default_schema()
        
        # Load chart configurations
        config_file = os.path.join(os.path.dirname(__file__), 'chart_config.json')
        try:
            with open(config_file, 'r') as f:
                self.chart_config = json.load(f)
        except FileNotFoundError:
            print(f"Warning: Chart config file not found at {config_file}")
            self.chart_config = self._get_default_chart_config()
        
        # Initialize rate limiting
        self._last_llm_call = 0
    
    def _get_default_schema(self) -> str:
        """Fallback schema if file not found"""
        return """
        -- Background Check System Schema
        -- Tables: Subject, Company, Package, Search_Type, Order_Request, Search, Search_status, Order_status
        """
    
    def _get_default_chart_config(self) -> Dict:
        """Fallback chart config if file not found"""
        return {
            "chart_types": {
                "bar": {"name": "Bar Chart", "description": "Compare categories"},
                "line": {"name": "Line Chart", "description": "Show trends"},
                "pie": {"name": "Pie Chart", "description": "Show proportions"}
            }
        }
    
    def process_user_query(self, user_query: str) -> Dict:
        """
        Streamlined processing pipeline:
        1. Generate SQL query from natural language
        2. Execute SQL and create DataFrame
        3. Suggest visualization type using LLM
        4. Transform data for frontend consumption
        5. Return structured response
        """
        print(f"🚀 Processing user query: {user_query}")
        
        try:
            # Step 1: Generate SQL query using LLM
            sql_query = self._generate_sql_query(user_query)
            if not sql_query:
                return self._error_response("Failed to generate SQL query")
            print(f"📝 Generated SQL: {sql_query}")
            
            # Step 2: Execute SQL and create DataFrame
            df = self._execute_sql_query(sql_query)
            if df is None or df.empty:
                return self._error_response("No data returned from query")
            print(f"📊 Retrieved DataFrame with shape: {df.shape}")
            
            # Step 3: Suggest visualization type using LLM
            suggested_chart, confidence, reasoning = self._suggest_visualization(df, user_query)
            print(f"🎯 Suggested chart: {suggested_chart} (confidence: {confidence})")
            
            # Step 4: Transform data for the suggested chart
            chart_data = self._prepare_chart_data(df, suggested_chart, user_query)
            
            # Step 5: Generate business explanation
            explanation = self._generate_explanation(df, user_query, sql_query, suggested_chart)
            
            return {
                "success": True,
                "query": user_query,
                "sql_query": sql_query,
                "data": {
                    "raw_data": df.to_dict('records'),
                    "columns": list(df.columns),
                    "row_count": len(df)
                },
                "chart": {
                    "type": suggested_chart,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "data": chart_data,
                    "config": self._get_chart_config(suggested_chart)
                },
                "explanation": explanation
            }
            
        except Exception as e:
            print(f"❌ Error in query processing: {e}")
            return self._error_response(f"Query processing failed: {str(e)}")
    
    def _prepare_chart_data(self, df: pd.DataFrame, chart_type: str, user_query: str) -> Dict:
        """
        Use LLM to dynamically transform data for optimal chart visualization
        Returns structured data ready for Plotly or other chart libraries
        """
        try:
            if df.empty:
                return {"error": "No data to visualize"}
            
            print(f"🔄 Generating dynamic transformation for {chart_type} chart...")
            
            # Step 1: Generate transformation code using LLM
            transformation_code = self._generate_data_transformation_code(df, chart_type, user_query)
            print(f"📝 Generated transformation code")
            
            # Step 2: Execute the transformation
            processed_df = self._execute_data_transformation(df, transformation_code)
            print(f"✅ Transformed data shape: {processed_df.shape}")
            
            # Step 3: Convert to frontend-ready format
            return self._format_chart_data_for_frontend(processed_df, chart_type)
            
        except Exception as e:
            print(f"❌ Error in dynamic chart preparation: {e}")
            # Fallback to basic formatting
            return self._format_chart_data_for_frontend(df, chart_type)
    
    def _format_chart_data_for_frontend(self, df: pd.DataFrame, chart_type: str) -> Dict:
        """
        Format the processed DataFrame into frontend-ready chart data structure
        """
        try:
            if df.empty:
                return {"error": "No data to display"}
            
            # Common data structure for all chart types
            chart_data = {
                "data": [],
                "layout": {
                    "title": f"{chart_type.title()} Chart",
                    "autosize": True,
                    "margin": {"l": 40, "r": 40, "t": 60, "b": 40}
                }
            }
            
            if chart_type == "bar":
                # Expect 2 columns: categories and values
                if len(df.columns) >= 2:
                    x_col, y_col = df.columns[0], df.columns[1]
                    chart_data["data"] = [{
                        "type": "bar",
                        "x": df[x_col].tolist(),
                        "y": df[y_col].tolist(),
                        "name": y_col.replace('_', ' ').title()
                    }]
                    chart_data["layout"]["xaxis"] = {"title": x_col.replace('_', ' ').title()}
                    chart_data["layout"]["yaxis"] = {"title": y_col.replace('_', ' ').title()}
                    
            elif chart_type == "pie":
                # Expect 2 columns: labels and values
                if len(df.columns) >= 2:
                    labels_col, values_col = df.columns[0], df.columns[1]
                    chart_data["data"] = [{
                        "type": "pie",
                        "labels": df[labels_col].tolist(),
                        "values": df[values_col].tolist(),
                        "name": "Distribution"
                    }]
                    
            elif chart_type == "line":
                # Expect 2 columns: x-axis (sequential) and y-axis (values)
                if len(df.columns) >= 2:
                    x_col, y_col = df.columns[0], df.columns[1]
                    chart_data["data"] = [{
                        "type": "scatter",
                        "mode": "lines+markers",
                        "x": df[x_col].tolist(),
                        "y": df[y_col].tolist(),
                        "name": y_col.replace('_', ' ').title()
                    }]
                    chart_data["layout"]["xaxis"] = {"title": x_col.replace('_', ' ').title()}
                    chart_data["layout"]["yaxis"] = {"title": y_col.replace('_', ' ').title()}
                    
            elif chart_type == "scatter":
                # Expect 2 numerical columns
                if len(df.columns) >= 2:
                    x_col, y_col = df.columns[0], df.columns[1]
                    chart_data["data"] = [{
                        "type": "scatter",
                        "mode": "markers",
                        "x": df[x_col].tolist(),
                        "y": df[y_col].tolist(),
                        "name": "Data Points"
                    }]
                    chart_data["layout"]["xaxis"] = {"title": x_col.replace('_', ' ').title()}
                    chart_data["layout"]["yaxis"] = {"title": y_col.replace('_', ' ').title()}
                    
            elif chart_type == "table":
                # Table format for Plotly
                chart_data["data"] = [{
                    "type": "table",
                    "header": {
                        "values": [col.replace('_', ' ').title() for col in df.columns],
                        "fill": {"color": "lightblue"},
                        "align": "center"
                    },
                    "cells": {
                        "values": [df[col].tolist() for col in df.columns],
                        "fill": {"color": "lightgray"},
                        "align": "center"
                    }
                }]
                
            else:
                # Default: convert to bar chart
                if len(df.columns) >= 2:
                    x_col, y_col = df.columns[0], df.columns[1]
                    chart_data["data"] = [{
                        "type": "bar",
                        "x": df[x_col].tolist(),
                        "y": df[y_col].tolist(),
                        "name": "Data"
                    }]
            
            return chart_data
            
        except Exception as e:
            print(f"❌ Error formatting chart data: {e}")
            return {
                "error": f"Failed to format {chart_type} chart data: {str(e)}",
                "raw_data": df.to_dict('records') if not df.empty else []
            }
    
    def _get_chart_config(self, chart_type: str) -> Dict:
        """Get configuration for the chart type"""
        return self.chart_config["chart_types"].get(chart_type, {})
        """Prepare data for pie chart"""
        if len(df.columns) < 2:
            # If only one column, use value counts
            col = df.columns[0]
            value_counts = df[col].value_counts()
            return {
                "chart_type": "pie",
                "labels": value_counts.index.tolist(),
                "values": value_counts.values.tolist()
            }
        
        # Use first column as labels, second as values
        labels_col = df.columns[0]
        values_col = df.columns[1]
        
        return {
            "chart_type": "pie",
            "labels": df[labels_col].tolist(),
            "values": df[values_col].tolist()
        }
    
    def _prepare_scatter_data(self, df: pd.DataFrame) -> Dict:
        """Prepare data for scatter plot"""
        if len(df.columns) < 2:
            return {"error": "Scatter plot requires at least 2 columns"}
        
        x_col = df.columns[0]
        y_col = df.columns[1]
        
        # Add color column if available
        color_col = df.columns[2] if len(df.columns) > 2 else None
        
        result = {
            "chart_type": "scatter",
            "x": df[x_col].tolist(),
            "y": df[y_col].tolist(),
            "x_label": x_col.replace('_', ' ').title(),
            "y_label": y_col.replace('_', ' ').title()
        }
        
        if color_col:
            result["color"] = df[color_col].tolist()
            result["color_label"] = color_col.replace('_', ' ').title()
        
        return result
    
    def _prepare_table_data(self, df: pd.DataFrame) -> Dict:
        """Prepare data for table display"""
        return {
            "chart_type": "table",
            "headers": [col.replace('_', ' ').title() for col in df.columns],
            "rows": df.head(100).values.tolist()  # Limit rows for performance
        }
    
    def _prepare_gauge_data(self, df: pd.DataFrame) -> Dict:
        """Prepare data for gauge chart"""
        if df.empty:
            return {"error": "No data for gauge"}
        
        # Use first numerical column
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) == 0:
            return {"error": "No numerical data for gauge"}
        
        col = numeric_cols[0]
        value = float(df[col].iloc[0]) if len(df) > 0 else 0
        
        return {
            "chart_type": "gauge",
            "value": value,
            "title": col.replace('_', ' ').title(),
            "max_value": float(df[col].max()) if len(df) > 1 else value * 1.2
        }
    
    def _get_chart_config(self, chart_type: str) -> Dict:
        """Get configuration for the chart type"""
        return self.chart_config["chart_types"].get(chart_type, {})
    
    @llm_retry_with_rate_limit(max_retries=3, base_delay=3.0)
    def _generate_sql_query(self, user_query: str) -> Optional[str]:
        """Generate SQL query using LLM with enhanced prompting"""
        prompt = f"""
You are a SQL expert working with a Background Check System database.
Your task is to translate user questions into **valid PostgreSQL SQL queries** with **exact column names**.

=== DATABASE SCHEMA ===
IMPORTANT: Use EXACT column names as shown:

1. **company** table:
   - comp_id, comp_name, comp_code

2. **order_request** table:
   - order_id, order_package_id, order_subject_id, order_company_code, order_status

3. **subject** table:
   - subject_id, subject_name, subject_alias, subject_contact

4. **search** table:
   - search_id, subject_id, search_type_code, search_status

5. **package** table:
   - package_code, package_name, package_price, comp_code

Key relationships:
- order_request.order_company_code = company.comp_code
- order_request.order_subject_id = subject.subject_id

=== USER QUESTION ===
"{user_query}"

=== CRITICAL RULES ===
1. Use LOWERCASE table names: company, order_request, subject, search, package
2. Use EXACT column names with underscores: order_company_code (NOT order_CompanyCode)
3. Always add LIMIT 100 for performance
4. Use appropriate JOINs for meaningful results
5. Return ONLY the SQL query (no explanations, no markdown)

SQL Query:
"""
        
        response = self.client.generate_content(prompt)
        sql_query = response.text.strip()
        
        # Clean up formatting
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()
        
        # Remove any extra text that might be added
        lines = sql_query.split('\n')
        sql_lines = []
        for line in lines:
            if line.strip() and not line.strip().startswith('--') and not line.strip().startswith('/*'):
                sql_lines.append(line)
        sql_query = '\n'.join(sql_lines)
        
        if not sql_query or not sql_query.upper().startswith('SELECT'):
            print(f"Invalid SQL generated: {sql_query}")
            return None
            
        return sql_query
    
    def _execute_sql_query(self, sql_query: str) -> Optional[pd.DataFrame]:
        """Execute SQL query with safety checks"""
        try:
            # Safety checks
            sql_lower = sql_query.lower().strip()
            dangerous_operations = [
                'drop', 'delete', 'truncate', 'alter', 'create', 
                'insert', 'update', 'grant', 'revoke'
            ]
            
            if any(op in sql_lower for op in dangerous_operations):
                raise ValueError("Potentially dangerous SQL operation detected")
            
            df = pd.read_sql(sql_query, self.engine)
            return df
            
        except Exception as e:
            print(f"SQL execution error: {e}")
            return None
    
    @llm_retry_with_rate_limit(max_retries=3, base_delay=3.0)
    def _suggest_visualization(self, df: pd.DataFrame, user_query: str) -> Tuple[str, str, str]:
        """Use LLM to suggest the most appropriate visualization type"""
        
        # Prepare data analysis for LLM
        data_analysis = self._analyze_dataframe(df)
        
        # Get available chart types
        available_charts = list(self.chart_config["chart_types"].keys())
        chart_descriptions = {
            chart: self.chart_config["chart_types"][chart]["description"] 
            for chart in available_charts
        }
        
        prompt = f"""
You are a data visualization expert. Analyze the following data and suggest the BEST chart type.

=== USER QUERY ===
"{user_query}"

=== DATA ANALYSIS ===
{data_analysis}

=== AVAILABLE CHART TYPES ===
{json.dumps(chart_descriptions, indent=2)}

=== DETAILED CHART CONFIGURATIONS ===
{json.dumps(self.chart_config["chart_types"], indent=2)}

=== INSTRUCTIONS ===
1. Consider the user's intent from their query
2. Analyze the data structure and types
3. Choose the MOST appropriate chart from the available options
4. Provide a confidence score (High/Medium/Low)
5. Give clear reasoning for your choice

Respond in this exact format:
CHART_TYPE: [chart_name]
CONFIDENCE: [High/Medium/Low]
REASONING: [Your detailed reasoning]
"""
        
        response = self.client.generate_content(prompt)
        response_text = response.text.strip()
        
        # Parse response
        chart_type = self._extract_field(response_text, "CHART_TYPE", "bar")
        confidence = self._extract_field(response_text, "CONFIDENCE", "Medium")
        reasoning = self._extract_field(response_text, "REASONING", "Default reasoning")
        
        # Validate chart type
        if chart_type not in available_charts:
            chart_type = "bar"  # Default fallback
        
        return chart_type, confidence, reasoning
    
    def _analyze_dataframe(self, df: pd.DataFrame) -> str:
        """Analyze DataFrame structure for LLM decision making"""
        if df.empty:
            return "DataFrame is empty"
        
        analysis = []
        analysis.append(f"Shape: {df.shape}")
        analysis.append(f"Columns: {list(df.columns)}")
        
        # Analyze column types
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            unique_count = df[col].nunique()
            
            analysis.append(f"  {col}: {dtype}, {non_null_count} non-null, {unique_count} unique values")
            
            # Sample values
            if unique_count <= 10:
                sample_values = df[col].value_counts().head(5).to_dict()
                analysis.append(f"    Sample values: {sample_values}")
            else:
                sample_values = df[col].head(3).tolist()
                analysis.append(f"    Sample values: {sample_values}")
        
        # Basic statistics for numerical columns
        numerical_cols = df.select_dtypes(include=['number']).columns
        if len(numerical_cols) > 0:
            analysis.append(f"Numerical columns statistics:")
            for col in numerical_cols:
                stats = df[col].describe()
                analysis.append(f"  {col}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
        
        return "\\n".join(analysis)
    
    @llm_retry_with_rate_limit(max_retries=3, base_delay=3.0)
    def _generate_explanation(self, df: pd.DataFrame, user_query: str, sql_query: str, chart_type: str) -> str:
        """Generate comprehensive explanation using LLM"""
        
        data_summary = self._analyze_dataframe(df) if df is not None else "No data processed"
        
        prompt = f"""
You are a business analyst. Provide a clear explanation of the query results.

=== USER QUERY ===
"{user_query}"

=== DATA SUMMARY ===
{data_summary}

=== CHOSEN VISUALIZATION ===
{chart_type} - {self.chart_config["chart_types"].get(chart_type, {}).get("description", "")}

=== INSTRUCTIONS ===
Provide a business-focused explanation that includes:
1. Key findings from the data
2. Why this visualization type was chosen
3. Business insights and implications

Keep it clear and concise (under 200 words).

Explanation:
"""
        
        response = self.client.generate_content(prompt)
        return response.text.strip()
    
    @llm_retry_with_rate_limit(max_retries=3, base_delay=3.0)
    def _generate_data_transformation_code(self, df: pd.DataFrame, chart_type: str, user_query: str) -> str:
        """Generate Python code to transform DataFrame for optimal chart visualization"""
        
        data_analysis = self._analyze_dataframe(df)
        chart_config = self._get_chart_config(chart_type)
        
        prompt = f"""
You are a data transformation expert. Generate Python pandas code to transform a DataFrame for optimal chart visualization.

=== USER QUERY ===
"{user_query}"

=== CURRENT DATA ANALYSIS ===
{data_analysis}

=== TARGET CHART TYPE ===
{chart_type} - {chart_config.get("description", "")}

Requirements: {chart_config.get("data_requirements", "")}

=== TRANSFORMATION GUIDELINES ===

For BAR charts:
- Need categorical x-axis and numerical y-axis
- Consider aggregation (groupby, sum, count, mean)
- Sort by values (ascending/descending)
- Limit to top N items if too many categories
- Handle missing values

For PIE charts:
- Need categories and their values/counts
- Aggregate if necessary 
- Limit to top categories (usually 5-8)
- Ensure positive values only

For LINE charts:
- Need sequential data (time, dates, ordered categories)
- Sort by x-axis values
- Handle time series if dates present

For SCATTER charts:
- Need two numerical columns
- Consider correlation analysis
- Handle outliers if necessary

For TABLE charts:
- Clean column names
- Handle missing values
- Sort meaningfully
- Limit rows if too many

For HISTOGRAM/BOX charts:
- Focus on numerical distributions
- Handle outliers
- Consider binning

=== CRITICAL RULES ===
1. Work with existing DataFrame 'df'
2. Create new DataFrame called 'processed_df'
3. Use only pandas operations (pd, df methods)
4. Handle edge cases (empty data, all nulls, etc.)
5. Ensure final DataFrame matches chart requirements
6. NO import statements
7. NO print statements
8. Keep column names simple and descriptive
9. Handle data type conversions if needed
10. Code must be executable with exec()

=== EXAMPLES ===
Bar chart transformation:
```python
# Group and aggregate data
processed_df = df.groupby('category_column')['value_column'].sum().reset_index()
processed_df = processed_df.sort_values('value_column', ascending=False).head(10)
processed_df.columns = ['Category', 'Total']
```

Pie chart transformation:
```python
# Create categories and values
processed_df = df['category_column'].value_counts().head(8).reset_index()
processed_df.columns = ['Category', 'Count']
```

Line chart transformation:
```python
# Time series or sequential data
processed_df = df.sort_values('date_column').copy()
processed_df = processed_df[['date_column', 'value_column']].dropna()
```

Generate ONLY the transformation code (no explanations, no markdown):
"""
        
        response = self.client.generate_content(prompt)
        code = response.text.strip()
        
        # Clean up the code
        code = code.replace("```python", "").replace("```", "").strip()
        
        # Remove any explanations or comments
        lines = []
        for line in code.split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('"""') and not line.startswith("'''"):
                lines.append(line)
        
        code = '\n'.join(lines)
        
        # Add safety wrapper
        safe_code = f"""
try:
    {code}
    # Ensure processed_df exists
    if 'processed_df' not in locals() or processed_df is None:
        processed_df = df.copy()
    # Ensure it's not empty
    if processed_df.empty:
        processed_df = df.copy()
except Exception as e:
    print(f"Data transformation error: {{e}}")
    processed_df = df.copy()
"""
        
        return safe_code
    
    def _execute_data_transformation(self, df: pd.DataFrame, transformation_code: str) -> pd.DataFrame:
        """Safely execute the LLM-generated transformation code"""
        try:
            # Create safe execution environment
            exec_globals = {
                'df': df.copy(),
                'pd': pd,
                'processed_df': None
            }
            
            # Execute the transformation code
            exec(transformation_code, exec_globals)
            
            # Get the processed DataFrame
            processed_df = exec_globals.get('processed_df', df.copy())
            
            # Validation checks
            if processed_df is None or processed_df.empty:
                print("Transformation resulted in empty DataFrame, using original")
                return df.copy()
            
            # Ensure reasonable size (limit rows for performance)
            if len(processed_df) > 1000:
                processed_df = processed_df.head(1000)
                print(f"Limited DataFrame to 1000 rows for performance")
            
            return processed_df
            
        except Exception as e:
            print(f"Error executing transformation: {e}")
            return df.copy()
    
    def _extract_field(self, text: str, field_name: str, default: str) -> str:
        """Extract field value from formatted LLM response"""
        try:
            pattern = f"{field_name}:\\s*(.+?)(?:\\n|$)"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
            return default
        except Exception:
            return default
    
    def _error_response(self, error_message: str) -> Dict:
        """Create standardized error response"""
        return {
            "success": False,
            "error": error_message,
            "query": "",
            "sql_query": "",
            "data": {"raw_data": [], "columns": [], "row_count": 0},
            "chart": {"type": "", "confidence": "Low", "reasoning": error_message, "data": {}, "config": {}},
            "explanation": f"Error: {error_message}"
        }

# Main entry point
def process_advanced_query(user_query: str) -> Dict:
    """
    Main entry point for advanced query processing
    This is called by the FastAPI endpoint
    """
    processor = AdvancedQueryProcessor()
    return processor.process_user_query(user_query)

# Testing function
def test_advanced_processing():
    """Test the advanced processing system"""
    print("🧪 Testing Advanced Query Processing System")
    print("=" * 60)
    
    test_queries = [
        "Show me the top 5 companies by number of orders",
        "What's the distribution of search statuses?",
        "How many orders are completed vs pending?",
        "Show me the most expensive packages"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n{i}. Testing: {query}")
        
        try:
            result = process_advanced_query(query)
            
            if result.get("success"):
                print(f"   ✅ Success!")
                print(f"   📝 SQL: {result['sql_query']}")
                print(f"   📊 Chart: {result['visualization']['suggested_chart']}")
                print(f"   🎯 Confidence: {result['visualization']['confidence']}")
                print(f"   📈 Data points: {result['data']['row_count']}")
            else:
                print(f"   ❌ Error: {result['error']}")
                
        except Exception as e:
            print(f"   💥 Exception: {e}")
    
    print(f"\\n✅ Advanced processing testing completed!")

if __name__ == "__main__":
    test_advanced_processing()