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

const MODEL = 'moonshotai/kimi-k3';
const JAI_MODEL_NAME = 'gpt-4k6';

const MAX_OUTPUT_TOKENS = 32384;


// ========================================
// HEALTH CHECK
// ========================================

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    model: MODEL
  });
});


// ========================================
// MODELS
// ========================================

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


// ========================================
// CHAT COMPLETIONS
// ========================================

app.post('/v1/chat/completions', async (req, res) => {

  try {

    const {
      messages,
      temperature,
      max_tokens,
      stream
    } = req.body;


    // ----------------------------------------
    // Validate request
    // ----------------------------------------

    if (!Array.isArray(messages) || messages.length === 0) {

      return res.status(400).json({
        error: {
          message: 'No messages were provided.',
          type: 'invalid_request_error',
          code: 400
        }
      });

    }


    // ----------------------------------------
    // Output token limit
    // ----------------------------------------

    let outputTokens = Number(max_tokens);

    if (!Number.isFinite(outputTokens) || outputTokens <= 0) {
      outputTokens = MAX_OUTPUT_TOKENS;
    }

    outputTokens = Math.min(
      outputTokens,
      MAX_OUTPUT_TOKENS
    );


    // ----------------------------------------
    // NVIDIA request
    // ----------------------------------------

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
      `JAI -> Kimi K3 | max_tokens=${outputTokens} | stream=${Boolean(stream)}`
    );


    // ----------------------------------------
    // Send to NVIDIA
    // ----------------------------------------

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

        responseType: stream
          ? 'stream'
          : 'json',

        timeout: 300000
      }
    );


    // ========================================
    // STREAMING
    // ========================================

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

      res.flushHeaders();


      let buffer = '';

      // Tracks whether we have entered Kimi's
      // reasoning section.
      let thinkStarted = false;

      // Tracks whether reasoning has ended.
      let thinkEnded = false;


      response.data.on('data', chunk => {

        buffer += chunk.toString();

        const lines = buffer.split('\n');

        // Keep incomplete line for next chunk
        buffer = lines.pop() || '';


        for (const line of lines) {

          if (!line.startsWith('data: ')) {
            continue;
          }


          // --------------------------------------
          // DONE
          // --------------------------------------

          if (line.trim() === 'data: [DONE]') {

            // Close the think block if Kimi ended
            // while still reasoning.
            if (thinkStarted && !thinkEnded) {

              res.write(
                `data: ${JSON.stringify({
                  choices: [
                    {
                      delta: {
                        content: '</think>\n'
                      }
                    }
                  ]
                })}\n\n`
              );

            }

            res.write('data: [DONE]\n\n');

            continue;
          }


          // --------------------------------------
          // Parse SSE JSON
          // --------------------------------------

          try {

            const data = JSON.parse(
              line.slice(6)
            );

            const delta =
              data.choices?.[0]?.delta;


            if (delta) {

              // NVIDIA/Kimi may use any of these
              // reasoning fields.
              const reasoning =
                delta.reasoning_content ||
                delta.reasoning ||
                delta.thinking_content ||
                '';


              // ==================================
              // REASONING TOKEN
              // ==================================

              if (reasoning) {

                // FIRST reasoning chunk
                if (!thinkStarted) {

                  thinkStarted = true;

                  delta.content =
                    '<think>\n' + reasoning;

                }

                // ALL subsequent reasoning chunks
                else {

                  delta.content =
                    reasoning;

                }

              }


              // ==================================
              // NORMAL RESPONSE TOKEN
              // ==================================

              else if (
                delta.content &&
                thinkStarted &&
                !thinkEnded
              ) {

                delta.content =
                  '</think>\n' +
                  delta.content;

                thinkEnded = true;

              }


              // Remove NVIDIA-specific reasoning
              // fields so JAI doesn't display them
              // separately.

              delete delta.reasoning;
              delete delta.reasoning_content;
              delete delta.thinking_content;

            }


            // --------------------------------------
            // Send modified SSE chunk to JAI
            // --------------------------------------

            res.write(
              `data: ${JSON.stringify(data)}\n\n`
            );


          } catch (err) {

            console.error(
              'Stream parsing error:',
              err.message
            );

          }

        }

      });


      // ----------------------------------------
      // Stream finished
      // ----------------------------------------

      response.data.on('end', () => {

        res.end();

      });


      // ----------------------------------------
      // Stream error
      // ----------------------------------------

      response.data.on('error', err => {

        console.error(
          'NVIDIA stream error:',
          err.message
        );

        res.end();

      });


      return;
    }


    // ========================================
    // NON-STREAMING
    // ========================================

    const data = response.data;


    res.json({
      id:
        data.id ||
        `chatcmpl-${Date.now()}`,

      object:
        'chat.completion',

      created:
        data.created ||
        Math.floor(Date.now() / 1000),

      model:
        JAI_MODEL_NAME,

      choices:
        data.choices || [],

      usage:
        data.usage || {
          prompt_tokens: 0,
          completion_tokens: 0,
          total_tokens: 0
        }
    });


  } catch (error) {

    // ========================================
    // NVIDIA ERROR
    // ========================================

    const status =
      error.response?.status || 500;

    const upstream =
      error.response?.data;


    console.error(
      'NVIDIA ERROR:',
      upstream || error.message
    );


    res.status(status).json({
      error: {

        message:
          upstream?.detail ||
          upstream?.error?.message ||
          error.message ||
          'NVIDIA request failed',

        type:
          'nvidia_api_error',

        code:
          status,

        details:
          upstream || null

      }
    });

  }

});


// ========================================
// 404
// ========================================

app.use((req, res) => {

  res.status(404).json({

    error: {

      message:
        `Endpoint ${req.path} not found`,

      type:
        'invalid_request_error',

      code:
        404

    }

  });

});


// ========================================
// START SERVER
// ========================================

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
