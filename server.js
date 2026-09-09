const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));

const NIM_API_BASE =
  process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY = process.env.NIM_API_KEY;

// JAI model name -> NVIDIA model
const MODEL = 'moonshotai/kimi-k3';
const JAI_MODEL_NAME = 'gpt-4k6';

// Maximum output tokens.
// Kimi K3 supports up to 16,384 output tokens on NVIDIA's endpoint.
const MAX_OUTPUT_TOKENS = 16384;


// ================================
// HEALTH CHECK
// ================================

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    model: MODEL
  });
});


// ================================
// MODEL LIST
// ================================

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: [
      {
        id: JAI_MODEL_NAME,
        object: 'model',
        created: Math.floor(Date.now() / 1000),
        owned_by: 'nvidia-nim-proxy'
      }
    ]
  });
});


// ================================
// CHAT COMPLETIONS
// ================================

app.post('/v1/chat/completions', async (req, res) => {

  try {

    const {
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;

    // ----------------------------
    // Validate messages
    // ----------------------------

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({
        error: {
          message: 'No messages were provided.',
          type: 'invalid_request_error',
          code: 400
        }
      });
    }

    // ----------------------------
    // Output token handling
    // ----------------------------

    let outputTokens = Number(max_tokens);

    if (!Number.isFinite(outputTokens) || outputTokens <= 0) {
      outputTokens = MAX_OUTPUT_TOKENS;
    }

    outputTokens = Math.min(
      outputTokens,
      MAX_OUTPUT_TOKENS
    );

    // ----------------------------
    // NVIDIA request
    // ----------------------------

    const nimRequest = {
      model: MODEL,
      messages: messages,
      max_tokens: outputTokens,
      temperature:
        typeof temperature === 'number'
          ? temperature
          : 1,
      reasoning_effort: 'max',
      stream: Boolean(stream)
    };

    console.log(
      `JAI request -> ${MODEL} | max_tokens=${outputTokens} | stream=${Boolean(stream)}`
    );

    const response = await axios.post(
      `${NIM_API_BASE}/chat/completions`,
      nimRequest,
      {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
          Accept: stream
            ? 'text/event-stream'
            : 'application/json'
        },

        responseType: stream ? 'stream' : 'json',

        timeout: 300000
      }
    );


    // ================================
    // STREAMING
    // ================================

    if (stream) {

      res.setHeader(
        'Content-Type',
        'text/event-stream'
      );

      res.setHeader(
        'Cache-Control',
        'no-cache'
      );

      res.setHeader(
        'Connection',
        'keep-alive'
      );

      let buffer = '';

      response.data.on('data', chunk => {

        buffer += chunk.toString();

        const lines = buffer.split('\n');

        buffer = lines.pop() || '';

        for (const line of lines) {

          if (!line.startsWith('data: ')) {
            continue;
          }

          // Pass DONE through
          if (line.trim() === 'data: [DONE]') {
            res.write('data: [DONE]\n\n');
            continue;
          }

          try {

            const data = JSON.parse(
              line.slice(6)
            );

            const delta =
              data.choices?.[0]?.delta;

            if (delta) {

              // Kimi can return reasoning separately.
              // Convert it into <think> tags for JAI.

              const reasoning =
                delta.reasoning_content ||
                delta.reasoning ||
                delta.thinking_content ||
                '';

              const content =
                delta.content || '';

              delete delta.reasoning;
              delete delta.reasoning_content;
              delete delta.thinking_content;

              let output = '';

              if (reasoning) {
                output += `<think>\n${reasoning}`;
              }

              if (content) {
                output += content;
              }

              if (output) {
                delta.content = output;
              }
            }

            res.write(
              `data: ${JSON.stringify(data)}\n\n`
            );

          } catch (e) {
            console.error(
              'Stream parsing error:',
              e.message
            );
          }
        }
      });

      response.data.on('end', () => {
        res.end();
      });

      response.data.on('error', err => {
        console.error(
          'NVIDIA stream error:',
          err.message
        );

        res.end();
      });

      return;
    }


    // ================================
    // NON-STREAMING
    // ================================

    const data = response.data;

    res.json({
      id:
        data.id ||
        `chatcmpl-${Date.now()}`,

      object: 'chat.completion',

      created:
        data.created ||
        Math.floor(Date.now() / 1000),

      model: JAI_MODEL_NAME,

      choices: data.choices || [],

      usage:
        data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
    });

  } catch (error) {

    console.error(
      'NVIDIA ERROR:',
      error.response?.data || error.message
    );

    const status =
      error.response?.status || 500;

    const nvidiaError =
      error.response?.data;

    res.status(status).json({
      error: {
        message:
          nvidiaError?.detail ||
          nvidiaError?.error?.message ||
          error.message ||
          'NVIDIA request failed',

        type: 'nvidia_api_error',

        code: status,

        details: nvidiaError || null
      }
    });
  }
});


// ================================
// 404
// ================================

app.use((req, res) => {
  res.status(404).json({
    error: {
      message: `Endpoint ${req.path} not found`,
      type: 'invalid_request_error',
      code: 404
    }
  });
});


// ================================
// START
// ================================

app.listen(PORT, () => {

  console.log(
    `Kimi K3 NVIDIA proxy running on port ${PORT}`
  );

  console.log(
    `NVIDIA model: ${MODEL}`
  );

  console.log(
    `JAI model name: ${JAI_MODEL_NAME}`
  );

});
