// server.js - OpenAI to NVIDIA NIM API Proxy

const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// NVIDIA NIM API configuration
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

// false = hide reasoning completely from client output.
// true = show reasoning inside one clean <think>...</think> block.
// Keep this false if you want no think leakage.
const SHOW_REASONING = false;

// Enables model-side thinking params for supported models.
const ENABLE_THINKING_MODE = true;

// Token limits
const DEFAULT_MAX_TOKENS = 9024;
const MAX_ALLOWED_TOKENS = 32768;

// Model mapping
const MODEL_MAPPING = {
  'gpt-4dv4': 'deepseek-ai/deepseek-v4-pro',
  'gpt-4g5': 'z-ai/glm-5.1',
  'gpt-4k5': 'moonshotai/kimi-k2.5',
  'gpt-4k6': 'moonshotai/kimi-k2.6',
  'gpt-4m35': 'mistralai/mistral-medium-3.5-128b',
  'claude-3-sonnet': 'z-ai/glm4.7',
  'mm27': 'mistralai/mistral-small-4-119b-2603'
};

// Model helpers
const isKimi = (m = '') => m.toLowerCase().includes('kimi');
const isDeepseek = (m = '') => m.toLowerCase().includes('deepseek');
const isGlm = (m = '') => m.toLowerCase().includes('glm');

/**
 * Stateful filter for streamed <think> leakage.
 *
 * Why this exists:
 * Kimi may stream <think> tags through delta.content, not just reasoning_content.
 * Those tags can be split across chunks:
 *
 *   chunk 1: "<thi"
 *   chunk 2: "nk>The hidden..."
 *
 * A normal regex per chunk will miss that. This filter keeps state across chunks.
 */
class ThinkStreamFilter {
  constructor({ showReasoning = false } = {}) {
    this.showReasoning = Boolean(showReasoning);
    this.insideThink = false;
    this.pending = '';
    this.reasoningOpened = false;
  }

  isPossiblePartialTag(text) {
    const lower = text.toLowerCase();

    return (
      lower === '<' ||
      lower === '</' ||
      '<think>'.startsWith(lower) ||
      '</think>'.startsWith(lower) ||
      '<thinking>'.startsWith(lower) ||
      '</thinking>'.startsWith(lower)
    );
  }

  filter(input = '') {
    let text = this.pending + String(input);
    this.pending = '';

    let output = '';
    let i = 0;

    while (i < text.length) {
      const rest = text.slice(i);
      const lowerRest = rest.toLowerCase();

      // Save possible partial tags at the end of the chunk.
      if (rest.startsWith('<') && this.isPossiblePartialTag(rest)) {
        this.pending = rest;
        break;
      }

      // Opening tags
      if (lowerRest.startsWith('<think>')) {
        this.insideThink = true;

        if (this.showReasoning && !this.reasoningOpened) {
          output += '<think>\n';
          this.reasoningOpened = true;
        }

        i += '<think>'.length;
        continue;
      }

      if (lowerRest.startsWith('<thinking>')) {
        this.insideThink = true;

        if (this.showReasoning && !this.reasoningOpened) {
          output += '<think>\n';
          this.reasoningOpened = true;
        }

        i += '<thinking>'.length;
        continue;
      }

      // Closing tags
      if (lowerRest.startsWith('</think>')) {
        this.insideThink = false;

        if (this.showReasoning && this.reasoningOpened) {
          output += '\n</think>\n\n';
          this.reasoningOpened = false;
        }

        i += '</think>'.length;
        continue;
      }

      if (lowerRest.startsWith('</thinking>')) {
        this.insideThink = false;

        if (this.showReasoning && this.reasoningOpened) {
          output += '\n</think>\n\n';
          this.reasoningOpened = false;
        }

        i += '</thinking>'.length;
        continue;
      }

      // Normal character handling
      if (!this.insideThink || this.showReasoning) {
        output += text[i];
      }

      i += 1;
    }

    return output;
  }

  flush() {
    this.pending = '';

    if (this.showReasoning && this.reasoningOpened) {
      this.reasoningOpened = false;
      this.insideThink = false;
      return '\n</think>\n\n';
    }

    this.insideThink = false;
    return '';
  }
}

function removeThinkBlocks(text = '') {
  const filter = new ThinkStreamFilter({ showReasoning: false });
  return filter.filter(String(text)) + filter.flush();
}

function cleanReasoning(text = '') {
  return String(text)
    .replace(/<\/?think>/gi, '')
    .replace(/<\/?thinking>/gi, '')
    .trim();
}

function getMaxTokens(maxTokens) {
  const requested = Number.isFinite(Number(maxTokens))
    ? Number(maxTokens)
    : DEFAULT_MAX_TOKENS;

  return Math.min(Math.max(requested, 1), MAX_ALLOWED_TOKENS);
}

async function resolveNimModel(model) {
  if (MODEL_MAPPING[model]) {
    return MODEL_MAPPING[model];
  }

  // Try direct model passthrough first.
  try {
    const testResponse = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      {
        model,
        messages: [{ role: 'user', content: 'test' }],
        max_tokens: 1,
        stream: false
      },
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json'
        },
        validateStatus: (status) => status < 500,
        timeout: 30000
      }
    );

    if (testResponse.status >= 200 && testResponse.status < 300) {
      return model;
    }
  } catch (_) {
    // Ignore and fall back.
  }

  const modelLower = String(model || '').toLowerCase();

  if (
    modelLower.includes('gpt-4') ||
    modelLower.includes('claude-opus') ||
    modelLower.includes('405b')
  ) {
    return 'meta/llama-3.1-405b-instruct';
  }

  if (
    modelLower.includes('claude') ||
    modelLower.includes('gemini') ||
    modelLower.includes('70b')
  ) {
    return 'meta/llama-3.1-70b-instruct';
  }

  return 'meta/llama-3.1-8b-instruct';
}

function buildNimRequest({
  nimModel,
  messages,
  temperature,
  max_tokens,
  stream
}) {
  const nimRequest = {
    model: nimModel,
    messages: Array.isArray(messages) ? messages : [],
    max_tokens: getMaxTokens(max_tokens),
    stream: Boolean(stream)
  };

  if (typeof temperature === 'number') {
    nimRequest.temperature = temperature;
  }

  // Kimi
  if (ENABLE_THINKING_MODE && isKimi(nimModel)) {
    nimRequest.chat_template_kwargs = { thinking: true };
    nimRequest.include_reasoning = true;
  }

  // DeepSeek
  if (ENABLE_THINKING_MODE && isDeepseek(nimModel)) {
    nimRequest.chat_template_kwargs = { thinking: true };
  }

  // GLM
  if (ENABLE_THINKING_MODE && isGlm(nimModel)) {
    nimRequest.chat_template_kwargs = { enable_thinking: true };
  }

  return nimRequest;
}

async function postToNim(nimRequest, stream) {
  let retries = 3;

  while (retries > 0) {
    try {
      return await axios.post(
        `${NIM_API_BASE}/chat/completions`,
        nimRequest,
        {
          headers: {
            Authorization: `Bearer ${NIM_API_KEY}`,
            'Content-Type': 'application/json'
          },
          responseType: stream ? 'stream' : 'json',
          timeout: 300000
        }
      );
    } catch (err) {
      retries -= 1;

      if (retries === 0) {
        throw err;
      }

      const status = err.response?.status;

      if (status === 502 || status === 503 || status === 504) {
        await new Promise((resolve) => setTimeout(resolve, 2000));
        continue;
      }

      throw err;
    }
  }

  throw new Error('NIM request failed after retries');
}

function createOpenAIError(status, message, type = 'invalid_request_error') {
  return {
    error: {
      message,
      type,
      code: status
    }
  };
}

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE
  });
});

// List models endpoint, OpenAI-compatible
app.get('/v1/models', (req, res) => {
  const models = Object.keys(MODEL_MAPPING).map((model) => ({
    id: model,
    object: 'model',
    created: Math.floor(Date.now() / 1000),
    owned_by: 'nvidia-nim-proxy'
  }));

  res.json({
    object: 'list',
    data: models
  });
});

// Chat completions endpoint
app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res
        .status(500)
        .json(createOpenAIError(500, 'NIM_API_KEY is not configured', 'server_error'));
    }

    const {
      model,
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;

    if (!model) {
      return res
        .status(400)
        .json(createOpenAIError(400, 'Missing required field: model'));
    }

    if (!Array.isArray(messages)) {
      return res
        .status(400)
        .json(createOpenAIError(400, 'Missing or invalid required field: messages'));
    }

    const nimModel = await resolveNimModel(model);

    // Do not inject jailbreak/unlock system prompts.
    const finalMessages = [...messages];

    const nimRequest = buildNimRequest({
      nimModel,
      messages: finalMessages,
      temperature,
      max_tokens,
      stream
    });

    const response = await postToNim(nimRequest, Boolean(stream));

    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');
      res.flushHeaders?.();

      let buffer = '';

      // Filters delta.content leakage.
      const contentFilter = new ThinkStreamFilter({ showReasoning: false });

      // Only used if SHOW_REASONING is true.
      const reasoningFilter = new ThinkStreamFilter({ showReasoning: SHOW_REASONING });

      const sendChunk = (data) => {
        res.write(`data: ${JSON.stringify(data)}\n\n`);
      };

      const sendTextChunk = (text) => {
        if (!text) return;

        sendChunk({
          id: `chatcmpl-${Date.now()}`,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now() / 1000),
          model,
          choices: [
            {
              index: 0,
              delta: { content: text },
              finish_reason: null
            }
          ]
        });
      };

      const flushFilters = () => {
        const flushedContent = contentFilter.flush();
        const flushedReasoning = SHOW_REASONING ? reasoningFilter.flush() : '';

        if (SHOW_REASONING) {
          sendTextChunk(flushedReasoning + flushedContent);
        } else {
          sendTextChunk(flushedContent);
        }
      };

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();

        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const rawLine of lines) {
          const line = rawLine.trimEnd();

          if (!line.startsWith('data: ')) {
            continue;
          }

          const payload = line.slice(6).trim();

          if (payload === '[DONE]') {
            flushFilters();
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {
            const data = JSON.parse(payload);
            const choice = data.choices?.[0];
            const delta = choice?.delta;

            if (!delta) {
              sendChunk(data);
              continue;
            }

            const reasoningRaw = delta.reasoning_content || '';
            const contentRaw = delta.content || '';

            delete delta.reasoning_content;

            let output = '';

            // If explicit reasoning_content exists, either hide it or wrap it cleanly.
            if (reasoningRaw && SHOW_REASONING) {
              const reasoning = cleanReasoning(reasoningRaw);

              if (reasoning) {
                output += reasoningFilter.filter(`<think>${reasoning}</think>`);
              }
            }

            // Always filter normal content because Kimi may leak <think> there.
            if (contentRaw) {
              output += contentFilter.filter(contentRaw);
            }

            delta.content = output;

            // Avoid sending empty chunks caused by hidden reasoning.
            if (!delta.content && !choice.finish_reason) {
              continue;
            }

            sendChunk(data);
          } catch (err) {
            console.error('Stream parse error:', err.message);
            // Do not forward malformed chunks because they may contain raw reasoning.
          }
        }
      });

      response.data.on('end', () => {
        flushFilters();
        res.end();
      });

      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        flushFilters();
        res.end();
      });

      return;
    }

    // Non-streaming response transform
    const openaiResponse = {
      id: response.data.id || `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: response.data.created || Math.floor(Date.now() / 1000),
      model,
      choices: (response.data.choices || []).map((choice, index) => {
        let fullContent = removeThinkBlocks(choice.message?.content || '');

        if (SHOW_REASONING && choice.message?.reasoning_content) {
          const reasoning = cleanReasoning(choice.message.reasoning_content);

          if (reasoning) {
            fullContent = `<think>\n${reasoning}\n</think>\n\n${fullContent}`;
          }
        }

        return {
          index: choice.index ?? index,
          message: {
            role: choice.message?.role || 'assistant',
            content: fullContent
          },
          finish_reason: choice.finish_reason || 'stop'
        };
      }),
      usage: response.data.usage || {
        prompt_tokens: 0,
        completion_tokens: 0,
        total_tokens: 0
      }
    };

    res.json(openaiResponse);
  } catch (error) {
    console.error('Proxy error:', error.message);

    const status = error.response?.status || 500;
    const message =
      error.response?.data?.error?.message ||
      error.response?.data?.message ||
      error.message ||
      'Internal server error';

    res.status(status).json({
      error: {
        message,
        type: error.response?.data?.error?.type || 'invalid_request_error',
        code: status
      }
    });
  }
});

// Catch-all for unsupported endpoints
app.all('*', (req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});

app.listen(PORT, () => {
  console.log(`OpenAI to NVIDIA NIM Proxy running on port ${PORT}`);
  console.log(`Health check: http://localhost:${PORT}/health`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ENABLED' : 'DISABLED'}`);
  console.log(`Thinking mode: ${ENABLE_THINKING_MODE ? 'ENABLED' : 'DISABLED'}`);
});
