import React, { useState, useEffect } from 'react';
import {
  Container,
  Typography,
  Paper,
  TextField,
  Button,
  Grid,
  Card,
  CardContent,
  Box,
  Alert,
  CircularProgress,
  Chip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';
import {
  ExpandMore as ExpandMoreIcon,
  Analytics as AnalyticsIcon,
  Send as SendIcon,
  AutoGraph as AutoGraphIcon,
  DataObject as DataObjectIcon,
  Insights as InsightsIcon,
} from '@mui/icons-material';
import Plot from 'react-plotly.js';
import axios from 'axios';
import './App.css';

// Chart type icons mapping
const chartIcons = {
  bar: '📊',
  line: '📈', 
  pie: '🥧',
  scatter: '⚬',
  heatmap: '🔥',
  histogram: '📋',
  box: '📦',
  gauge: '🎯',
  table: '📋',
  sunburst: '☀️'
};

function App() {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [availableCharts, setAvailableCharts] = useState([]);

  const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // Load available charts on component mount
  useEffect(() => {
    loadAvailableCharts();
  }, []);

  const loadAvailableCharts = async () => {
    try {
      const response = await axios.get(`${apiUrl}/api/charts`);
      if (response.data.success) {
        setAvailableCharts(response.data.chart_types);
      }
    } catch (error) {
      console.error('Error loading charts:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await axios.post(`${apiUrl}/api/query`, {
        query: query.trim()
      });

      if (response.data.success) {
        setResult(response.data.data);
      } else {
        setError(response.data.message || 'Query processing failed');
      }
    } catch (error) {
      setError(error.message || 'Network error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleSampleQuery = (sampleQuery) => {
    setQuery(sampleQuery);
  };

  const sampleQueries = [
    "Show me the top 5 companies by number of orders",
    "What's the distribution of search statuses?",
    "How many orders are completed vs pending?",
    "Show me the most expensive packages",
    "Which subjects have the most searches?"
  ];

  return (
    <div className="App">
      {/* Header */}
      <Box
        sx={{
          background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
          color: 'white',
          py: 4,
          mb: 4
        }}
      >
        <Container maxWidth="lg">
          <Typography variant="h2" component="h1" gutterBottom fontWeight="bold">
            🤖 VeriBot
          </Typography>
          <Typography variant="h5" component="p">
            AI-Powered Data Visualization & Query Processing
          </Typography>
        </Container>
      </Box>

      <Container maxWidth="lg">
        {/* Query Input Section */}
        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AnalyticsIcon /> Ask me anything about your data:
          </Typography>
          
          <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
            <TextField
              fullWidth
              multiline
              rows={3}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="e.g., Show me the top 5 companies by number of orders, or What's the distribution of search statuses?"
              variant="outlined"
              sx={{ mb: 2 }}
            />
            
            <Button
              type="submit"
              variant="contained"
              size="large"
              disabled={loading || !query.trim()}
              startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
              sx={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                '&:hover': {
                  background: 'linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%)',
                }
              }}
            >
              {loading ? 'Analyzing...' : 'Analyze'}
            </Button>
          </Box>

          {/* Sample Queries */}
          <Box sx={{ mt: 3 }}>
            <Typography variant="body2" color="text.secondary" gutterBottom>
              Try these examples:
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
              {sampleQueries.map((sample, index) => (
                <Chip
                  key={index}
                  label={sample}
                  variant="outlined"
                  clickable
                  onClick={() => handleSampleQuery(sample)}
                  sx={{ mb: 1 }}
                />
              ))}
            </Box>
          </Box>
        </Paper>

        {/* Available Charts Section */}
        <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
          <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <AutoGraphIcon /> Available Chart Types
          </Typography>
          
          <Grid container spacing={2}>
            {availableCharts.map((chart) => (
              <Grid item xs={12} sm={6} md={4} lg={3} key={chart.id}>
                <Card 
                  sx={{ 
                    height: '100%',
                    cursor: 'pointer',
                    transition: 'transform 0.2s, box-shadow 0.2s',
                    '&:hover': {
                      transform: 'translateY(-2px)',
                      boxShadow: 4
                    }
                  }}
                >
                  <CardContent sx={{ textAlign: 'center', p: 2 }}>
                    <Typography variant="h3" sx={{ mb: 1 }}>
                      {chartIcons[chart.id] || '📊'}
                    </Typography>
                    <Typography variant="h6" gutterBottom>
                      {chart.name}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {chart.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Paper>

        {/* Error Display */}
        {error && (
          <Alert severity="error" sx={{ mb: 4 }}>
            {error}
          </Alert>
        )}

        {/* Results Section */}
        {result && (
          <Paper elevation={3} sx={{ p: 3, mb: 4 }}>
            <Typography variant="h5" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <InsightsIcon /> Query Results
            </Typography>

            {/* Results Metadata */}
            <Box sx={{ mb: 3, display: 'flex', gap: 2, flexWrap: 'wrap' }}>
              <Chip icon={<DataObjectIcon />} label={`${result.data?.row_count || 0} records`} />
              <Chip 
                label={`${result.chart?.type || 'Unknown'} chart`}
                color="primary"
              />
            </Box>

            {/* Chart Suggestion */}
            {result.chart?.type && (
              <Alert 
                severity="info" 
                sx={{ mb: 3 }}
                icon={<span style={{ fontSize: '1.5rem' }}>🤖</span>}
              >
                <Typography variant="subtitle1" fontWeight="bold">
                  AI Recommended Chart: {result.chart.type.toUpperCase()}
                </Typography>
                <Typography variant="body2">
                  {result.chart.reasoning || 'Chart selected based on data analysis'}
                </Typography>
              </Alert>
            )}

            {/* Visualization */}
            {result.chart?.data && (
              <Box sx={{ mb: 3 }}>
                <Typography variant="h6" gutterBottom>
                  📈 Visualization
                </Typography>
                <Paper variant="outlined" sx={{ p: 2 }}>
                  <Plot
                    data={result.chart.data.data}
                    layout={{
                      ...result.chart.data.layout,
                      autosize: true,
                    }}
                    style={{ width: '100%', height: '500px' }}
                    useResizeHandler={true}
                  />
                </Paper>
              </Box>
            )}

            {/* Expandable Sections */}
            <Box>
              {/* SQL Query */}
              {result.sql_query && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="h6">📝 Generated SQL Query</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Paper 
                      variant="outlined" 
                      sx={{ 
                        p: 2, 
                        backgroundColor: '#f5f5f5',
                        fontFamily: 'monospace',
                        whiteSpace: 'pre-wrap',
                        overflow: 'auto'
                      }}
                    >
                      {result.sql_query}
                    </Paper>
                  </AccordionDetails>
                </Accordion>
              )}

              {/* Data Table */}
              {result.data?.raw_data && result.data.raw_data.length > 0 && (
                <Accordion>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="h6">📊 Raw Data</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <TableContainer component={Paper} variant="outlined" sx={{ maxHeight: 400 }}>
                      <Table stickyHeader size="small">
                        <TableHead>
                          <TableRow>
                            {Object.keys(result.data.raw_data[0] || {}).map((column) => (
                              <TableCell key={column} sx={{ fontWeight: 'bold' }}>
                                {column.replace('_', ' ').toUpperCase()}
                              </TableCell>
                            ))}
                          </TableRow>
                        </TableHead>
                        <TableBody>
                          {result.data.raw_data.slice(0, 100).map((row, index) => (
                            <TableRow key={index} hover>
                              {Object.values(row).map((value, cellIndex) => (
                                <TableCell key={cellIndex}>
                                  {value !== null && value !== undefined ? String(value) : ''}
                                </TableCell>
                              ))}
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </TableContainer>
                    {result.data.raw_data.length > 100 && (
                      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
                        ... and {result.data.raw_data.length - 100} more rows
                      </Typography>
                    )}
                  </AccordionDetails>
                </Accordion>
              )}

              {/* AI Explanation */}
              {result.explanation && (
                <Accordion defaultExpanded>
                  <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                    <Typography variant="h6">🧠 AI Analysis</Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Paper 
                      variant="outlined" 
                      sx={{ 
                        p: 2, 
                        backgroundColor: '#f8f9fa',
                        borderLeft: '4px solid #4caf50'
                      }}
                    >
                      <Typography variant="body1" sx={{ lineHeight: 1.7 }}>
                        {result.explanation}
                      </Typography>
                    </Paper>
                  </AccordionDetails>
                </Accordion>
              )}
            </Box>
          </Paper>
        )}
      </Container>

      {/* Footer */}
      <Box
        sx={{
          backgroundColor: 'rgba(0, 0, 0, 0.1)',
          py: 3,
          mt: 6,
          textAlign: 'center'
        }}
      >
        <Container maxWidth="lg">
          <Typography variant="body2" color="text.secondary">
            © 2025 VeriBot. Powered by AI for intelligent data visualization.
          </Typography>
        </Container>
      </Box>
    </div>
  );
}

export default App;
