from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import json
from pathlib import Path

# Import our custom query processor
from advanced_query_processor import process_advanced_query

app = FastAPI(
    title="VeriBot API",
    description="AI-Powered Data Visualization & Query Processing API",
    version="1.0.0"
)

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    chart_type: Optional[str] = None  # Optional: user can specify preferred chart

class QueryResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    message: Optional[str] = None

# Mount static files for frontend
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

# Health check endpoint
@app.get("/")
async def root():
    """Root endpoint - serve the frontend"""
    try:
        frontend_file = frontend_path / "index.html"
        if frontend_file.exists():
            with open(frontend_file, 'r', encoding='utf-8') as f:
                return HTMLResponse(content=f.read())
        else:
            return {"message": "VeriBot API is running! Frontend not found.", "status": "healthy"}
    except Exception as e:
        return {"message": f"VeriBot API is running! Error loading frontend: {str(e)}", "status": "healthy"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "VeriBot API is running successfully"}

# Main query processing endpoint
@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    """
    Process user query through the advanced pipeline:
    1. Generate SQL from natural language
    2. Execute SQL and get data
    3. Suggest visualization type
    4. Transform data for visualization
    5. Create chart and return results
    """
    try:
        if not request.query or not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process the query through our advanced system
        result = process_advanced_query(request.query.strip())
        
        if result["success"]:
            return QueryResponse(
                success=True,
                data=result,
                message="Query processed successfully"
            )
        else:
            return QueryResponse(
                success=False,
                data=result,
                message=f"Query processing failed: {result.get('error', 'Unknown error')}"
            )
    
    except HTTPException:
        raise
    except Exception as e:
        error_result = {
            "type": "api_error",
            "success": False,
            "error": str(e),
            "original_query": request.query,
            "sql_query": "",
            "data": {"original_df": [], "processed_df": [], "columns": [], "row_count": 0},
            "visualization": {"suggested_chart": "", "confidence": "Low", "reasoning": f"API Error: {str(e)}", "chart_data": None},
            "manipulation_code": "",
            "explanation": f"An API error occurred while processing your query: {str(e)}",
            "metadata": {"processing_steps": ["API Error"]}
        }
        
        return QueryResponse(
            success=False,
            data=error_result,
            message=f"Internal server error: {str(e)}"
        )

# Get available chart types
@app.get("/api/charts")
async def get_available_charts():
    """Get list of available chart types and their configurations"""
    try:
        config_file = Path(__file__).parent / "chart_config.json"
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                chart_config = json.load(f)
            
            # Extract just the basic info for frontend display
            chart_types = []
            for chart_id, config in chart_config["chart_types"].items():
                chart_types.append({
                    "id": chart_id,
                    "name": config["name"],
                    "description": config["description"],
                    "icon": config["icon"],
                    "use_cases": config["use_cases"]
                })
            
            return {
                "success": True,
                "chart_types": chart_types,
                "total_charts": len(chart_types)
            }
        else:
            # Fallback chart types
            fallback_charts = [
                {"id": "bar", "name": "Bar Chart", "description": "Compare categories", "icon": "📊", "use_cases": ["Comparisons"]},
                {"id": "line", "name": "Line Chart", "description": "Show trends", "icon": "📈", "use_cases": ["Trends"]},
                {"id": "pie", "name": "Pie Chart", "description": "Show proportions", "icon": "🥧", "use_cases": ["Proportions"]},
                {"id": "scatter", "name": "Scatter Plot", "description": "Show relationships", "icon": "⚬", "use_cases": ["Relationships"]},
                {"id": "table", "name": "Data Table", "description": "Show raw data", "icon": "📋", "use_cases": ["Raw Data"]}
            ]
            
            return {
                "success": True,
                "chart_types": fallback_charts,
                "total_charts": len(fallback_charts),
                "message": "Using fallback chart configuration"
            }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading chart configurations: {str(e)}")

# Test endpoint for SQL generation only
@app.post("/api/test-sql")
async def test_sql_generation(request: QueryRequest):
    """Test endpoint that only generates SQL without executing it"""
    try:
        from advanced_query_processor import AdvancedQueryProcessor
        
        processor = AdvancedQueryProcessor()
        sql_query = processor._generate_sql_query(request.query)
        
        return {
            "success": True,
            "query": request.query,
            "generated_sql": sql_query,
            "message": "SQL generated successfully (not executed)"
        }
    
    except Exception as e:
        return {
            "success": False,
            "query": request.query,
            "error": str(e),
            "message": "SQL generation failed"
        }

# Get database schema
@app.get("/api/schema")
async def get_database_schema():
    """Get the database schema for reference"""
    try:
        schema_file = Path(__file__).parent / "database_schema.sql"
        if schema_file.exists():
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_content = f.read()
            
            return {
                "success": True,
                "schema": schema_content,
                "message": "Database schema retrieved successfully"
            }
        else:
            return {
                "success": False,
                "schema": "",
                "message": "Database schema file not found"
            }
    
    except Exception as e:
        return {
            "success": False,
            "schema": "",
            "error": str(e),
            "message": "Error reading database schema"
        }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "error": str(exc)}
    )

# CORS middleware for development
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup event
@app.on_event("startup")
async def startup_event():
    print("🚀 VeriBot API starting up...")
    print("📊 Advanced Query Processing System initialized")
    print("🎯 Chart configuration loaded")
    print("✅ API ready to process queries!")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    print("🛑 VeriBot API shutting down...")
    print("👋 Goodbye!")

if __name__ == "__main__":
    import uvicorn
    
    # Run the API
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )