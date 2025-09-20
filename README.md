# VeriBot - AI-Powered Data Visualization Platform

🤖 **VeriBot** is an intelligent data visualization platform that transforms natural language queries into SQL queries, executes them against your database, and automatically suggests and creates the most appropriate visualizations using AI.

## 🌟 Key Features

### 🎯 **Multi-LLM Pipeline**
- **Query Understanding**: Converts natural language to SQL queries
- **Visualization Intelligence**: AI suggests the best chart type from 10+ options
- **Data Transformation**: Automatically generates Python code to transform data for visualization
- **Smart Execution**: Safely executes transformations on the same DataFrame

### 📊 **10 Chart Types Available**
1. **Bar Chart** 📊 - Compare categories and values
2. **Line Chart** 📈 - Show trends over time
3. **Pie Chart** 🥧 - Display proportional data
4. **Scatter Plot** ⚬ - Show relationships between variables
5. **Heatmap** 🔥 - Visualize data density
6. **Histogram** 📋 - Show data distribution
7. **Box Plot** 📦 - Display statistical summaries
8. **Gauge Chart** 🎯 - Show single metric progress
9. **Data Table** 📋 - Display structured data
10. **Sunburst Chart** ☀️ - Show hierarchical data

### 🧠 **AI-Powered Processing**
- **LLM 1**: Natural language → SQL query generation
- **LLM 2**: Data analysis → Chart type suggestion
- **LLM 3**: DataFrame transformation code generation
- **LLM 4**: Business insights explanation generation

## 🏗️ Architecture

```
User Query → SQL Generation → Data Extraction → Chart Suggestion → Data Transformation → Visualization
     ↓              ↓              ↓               ↓                 ↓                ↓
  Frontend → Advanced Query → PostgreSQL → AI Analysis → Python Exec → Plotly Chart
```

## 📁 Project Structure

```
veri_bot/
├── frontend/                 # Web interface
│   ├── index.html           # Main UI with query textbox & chart previews
│   ├── styles.css           # Modern responsive styling
│   └── script.js            # Frontend logic & API communication
├── backend/                 # Python backend
│   ├── api_server.py        # FastAPI server with all endpoints
│   ├── advanced_query_processor.py  # Multi-LLM processing pipeline
│   ├── sql_retrieval.py     # Original SQL processing (legacy)
│   ├── chart_config.json    # Chart type configurations & requirements
│   └── database_schema.sql  # Database schema for SQL generation
├── data/                    # CSV data files
├── scripts/                 # Utility scripts
├── requirements.txt         # Python dependencies
└── start_veribot.sh        # Startup script
```

## 🚀 Quick Start

### 1. Prerequisites
```bash
# Required
- Python 3.8+
- PostgreSQL database
- Gemini API key

# Optional (for development)
- Node.js (for frontend development)
- Git
```

### 2. Environment Setup
```bash
# Clone and navigate
cd veri_bot

# Create environment file
cat > backend/.env << EOF
NEON_CONNECTION_STRING=postgresql://username:password@host:port/database
GEMINI_API_KEY=your_gemini_api_key_here
EOF
```

### 3. Start the Application
```bash
# Using the startup script (recommended)
./start_veribot.sh

# Or manually
cd backend
pip install -r ../requirements.txt
python api_server.py
```

### 4. Access the Application
- **Web Interface**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 🎮 Usage Examples

### Natural Language Queries
```
"Show me the top 5 companies by number of orders"
→ Generates bar chart of companies vs order counts

"What's the distribution of search statuses?"  
→ Creates pie chart of status proportions

"How do package prices vary over time?"
→ Produces line chart showing price trends

"Show me the correlation between order priority and completion time"
→ Creates scatter plot with correlation analysis
```

### API Usage
```python
import requests

# Submit a query
response = requests.post('http://localhost:8000/api/query', 
    json={'query': 'Show me pending orders by company'})

result = response.json()
if result['success']:
    # Access the data
    data = result['data']['processed_df']  # Transformed data
    chart = result['data']['visualization']['chart_data']  # Plotly JSON
    explanation = result['data']['explanation']  # AI insights
```

## 🔧 Technical Details

### Multi-LLM Processing Pipeline

1. **SQL Generation**
   ```python
   user_query → LLM + database_schema → SQL query
   ```

2. **Data Extraction**
   ```python
   SQL query → PostgreSQL → pandas DataFrame
   ```

3. **Chart Suggestion**
   ```python
   DataFrame + user_intent → LLM + chart_config → suggested_chart_type
   ```

4. **Data Transformation**
   ```python
   DataFrame + chart_requirements → LLM → Python transformation code
   ```

5. **Safe Execution**
   ```python
   exec(transformation_code, {'df': dataframe}) → processed_df
   ```

6. **Visualization Creation**
   ```python
   processed_df + chart_type → Plotly → interactive_chart
   ```

### Chart Configuration System

Each chart type has detailed configuration:
```json
{
  "bar": {
    "data_requirements": {
      "min_columns": 2,
      "column_types": ["categorical", "numerical"]
    },
    "use_cases": ["comparing quantities", "rankings"],
    "plotly_config": {
      "chart_type": "bar",
      "x_axis": "categorical_column",
      "y_axis": "numerical_column"
    }
  }
}
```

### Safety Features
- SQL injection prevention
- Code execution sandboxing
- Error handling at each step
- Fallback visualizations
- Data size limits

## 🛠️ API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/api/query` | POST | Process natural language query |
| `/api/charts` | GET | Get available chart types |
| `/api/test-sql` | POST | Test SQL generation only |
| `/api/schema` | GET | Get database schema |
| `/health` | GET | Health check |

## 📊 Sample Database Schema

The system works with a background check database:

```sql
-- Main entities
- Subject (candidates for background checks)
- Company (clients requesting checks) 
- Package (background check packages)
- Search (individual background checks)
- Order_Request (orders for background checks)

-- Lookup tables  
- Search_Type (types of checks)
- Search_status (check statuses)
- Order_status (order statuses)
```

## 🎨 Frontend Features

- **Modern UI**: Clean, responsive design with gradient backgrounds
- **Real-time Processing**: Loading indicators and status updates
- **Interactive Charts**: Plotly-powered visualizations with zoom, pan, hover
- **Data Tables**: Sortable, searchable data display
- **Error Handling**: User-friendly error messages and retry options
- **Mobile Responsive**: Works on all device sizes

## 🔧 Development

### Adding New Chart Types

1. Update `chart_config.json`:
```json
{
  "new_chart": {
    "name": "New Chart",
    "description": "Chart description", 
    "data_requirements": {...},
    "plotly_config": {...}
  }
}
```

2. Add visualization logic in `advanced_query_processor.py`:
```python
elif chart_type == "new_chart":
    fig = create_new_chart(df)
```

3. Update frontend chart preview in `index.html`

### Custom LLM Prompts

Modify prompts in `advanced_query_processor.py`:
- `_generate_sql_query()` - SQL generation prompt
- `_suggest_visualization()` - Chart suggestion prompt  
- `_generate_manipulation_code()` - Data transformation prompt

## 🐛 Troubleshooting

### Common Issues

1. **Database Connection Failed**
   - Check `NEON_CONNECTION_STRING` in `.env`
   - Verify database is accessible
   - Test connection with `psql` or similar

2. **LLM API Errors**
   - Verify `GEMINI_API_KEY` in `.env`
   - Check API quota and rate limits
   - Try with a simpler query

3. **Chart Generation Fails**
   - Check data structure compatibility
   - Review error logs for specific issues
   - Falls back to table view automatically

4. **Dependencies Missing**
   - Run `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Debug Mode
```bash
# Enable detailed logging
export LOG_LEVEL=DEBUG
python api_server.py
```

## 📈 Performance Considerations

- **Query Limits**: SQL queries auto-limited to 1000 rows
- **Chart Data**: Large datasets automatically sampled
- **Memory Usage**: DataFrames cleared after processing
- **Caching**: Consider adding Redis for repeated queries

## 🔒 Security

- SQL injection prevention through query validation
- Code execution in controlled environment
- Input sanitization on all endpoints
- CORS configured for development (adjust for production)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Google Gemini** for AI capabilities
- **Plotly** for interactive visualizations
- **FastAPI** for the web framework
- **PostgreSQL** for robust data storage

---

**VeriBot** - Transforming data questions into insights, one query at a time! 🚀