const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const NIM_API_BASE = 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const MODEL_MAPPING = {
  'gpt-4dv4': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4g5': 'z-ai/glm-5.1',
  'gpt-4k5': 'moonshotai/kimi-k2.5',
  'gpt-4k6': 'moonshotai/kimi-k2.6',
  'gpt-4m35': 'mistralai/mistral-medium-3.5-128b',
  'claude-3-sonnet': 'z-ai/glm4.7',
  'mm27': 'mistralai/mistral-small-4-119b-2603'
};

app.get('/health', (req, res) => res.json({ status: 'ok' }));

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map(id => ({ id, object: 'model', created: Date.now(), owned_by: 'proxy' }))
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    const { model, messages, temperature, max_tokens, stream, top_p } = req.body;
    const nimModel = MODEL_MAPPING[model] || model;

    const nimRequest = {
      model: nimModel,
      messages,
      ...(temperature !== undefined && { temperature }),
      ...(max_tokens !== undefined && { max_tokens }),
      ...(top_p !== undefined && { top_p }),
      stream: stream || false
    };

    const response = await axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
      headers: { 'Authorization': `Bearer ${NIM_API_KEY}`, 'Content-Type': 'application/json' },
      responseType: stream ? 'stream' : 'json',
      timeout: 300000
    });

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      response.data.on('data', chunk => res.write(chunk));
      response.data.on('end', () => res.end());
      response.data.on('error', () => res.end());
    } else {
      res.json(response.data);
    }

  } catch (error) {
    res.status(error.response?.status || 500).json({
      error: { message: error.message, type: 'invalid_request_error', code: error.response?.status || 500 }
    });
  }
});

app.all('*', (req, res) => res.status(404).json({ error: { message: `Endpoint ${req.path} not found` } }));

app.listen(PORT, () => console.log(`Proxy running on port ${PORT}`));
