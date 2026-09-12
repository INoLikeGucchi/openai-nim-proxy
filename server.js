const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json({ limit: '50mb' }));


// ========================================
// NVIDIA NIM CONFIG
// ========================================

const NIM_API_BASE =
  process.env.NIM_API_BASE ||
  'https://integrate.api.nvidia.com/v1';

const NIM_API_KEY =
  process.env.NIM_API_KEY?.trim();


// ========================================
// MODELS
// ========================================

// Primary model
const MODEL = 'moonshotai/kimi-k3';

// Fallback model
const FALLBACK_MODEL =
  'deepseek-ai/deepseek-v4-pro-0813';

// Model name exposed to JanitorAI
const JAI_MODEL_NAME = 'gpt-4k6';


// ========================================
// TOKEN LIMITS
// ========================================

// KEEPING YOUR ORIGINAL LIMIT
const MAX_OUTPUT_TOKENS = 32384;

// NVIDIA currently documents DeepSeek V4 Pro 0813
// with a maximum max_tokens value of 16384.
const FALLBACK_MAX_OUTPUT_TOKENS = 16384;


// ========================================
// API KEY CHECK
// ========================================

if (!NIM_API_KEY) {

  console.error(
    'ERROR: NIM_API_KEY is missing!'
  );

} else {

  console.log(
    `NVIDIA API key loaded: ${NIM_API_KEY.slice(0, 8)}...`
  );

}


// ========================================
// REQUEST LOGGER
// ========================================

app.use((req, res, next) => {

  console.log(
    `[REQUEST] ${req.method} ${req.path}`
  );

  next();

});


// ========================================
// HEALTH CHECK
// ========================================

app.get('/health', (req, res) => {

  res.json({

    status: 'ok',

    model: MODEL,

    fallback_model: FALLBACK_MODEL

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

        created:
          Math.floor(Date.now() / 1000),

        owned_by:
          'nvidia-nim-proxy'

      }

    ]

  });

});


// ========================================
// BUILD NVIDIA REQUEST
// ========================================

function buildNimRequest(
  model,
  messages,
  temperature,
  outputTokens,
  stream
) {

  return {

    model,

    messages,

    max_tokens: outputTokens,

    temperature:
      typeof temperature === 'number'
        ? temperature
        : 1,

    // Both Kimi and DeepSeek support reasoning.
    reasoning_effort: 'max',

    stream: Boolean(stream)

  };

}


// ========================================
// SEND REQUEST TO NVIDIA
// ========================================

async function sendToNvidia(
  model,
  messages,
  temperature,
  outputTokens,
  stream
) {

  const nimRequest = buildNimRequest(
    model,
    messages,
    temperature,
    outputTokens,
    stream
  );


  console.log(
    `NVIDIA -> ${model} | max_tokens=${outputTokens} | stream=${Boolean(stream)}`
  );


  return await axios.post(

    `${NIM_API_BASE}/chat/completions`,

    nimRequest,

    {

      headers: {

        'Authorization':
          `Bearer ${NIM_API_KEY}`,

        'Content-Type':
          'application/json',

        'Accept':
          stream
            ? 'text/event-stream'
            : 'application/json'

      },

      responseType:
        stream
          ? 'stream'
          : 'json',

      timeout: 300000

    }

  );

}


// ========================================
// CHAT COMPLETIONS
// ========================================

app.post(
  '/v1/chat/completions',
  async (req, res) => {

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

      if (
        !Array.isArray(messages) ||
        messages.length === 0
      ) {

        return res.status(400).json({

          error: {

            message:
              'No messages were provided.',

            type:
              'invalid_request_error',

            code: 400

          }

        });

      }


      // ----------------------------------------
      // Output token limit
      // ----------------------------------------

      let outputTokens =
        Number(max_tokens);


      if (
        !Number.isFinite(outputTokens) ||
        outputTokens <= 0
      ) {

        outputTokens =
          MAX_OUTPUT_TOKENS;

      }


      // KEEP ORIGINAL KIMI LIMIT
      outputTokens =
        Math.min(
          outputTokens,
          MAX_OUTPUT_TOKENS
        );


      // ----------------------------------------
      // Which model are we using?
      // ----------------------------------------

      let activeModel = MODEL;

      let response;


      // ========================================
      // PRIMARY: KIMI K3
      // ========================================

      try {

        response =
          await sendToNvidia(

            MODEL,

            messages,

            temperature,

            outputTokens,

            stream

          );


        console.log(
          `NVIDIA PRIMARY SUCCESS: ${MODEL}`
        );


      } catch (kimiError) {

        // ======================================
        // KIMI FAILED
        // ======================================

        console.error(
          '========================================'
        );

        console.error(
          'KIMI K3 FAILED'
        );

        console.error(
          'Status:',
          kimiError.response?.status ||
          'NO RESPONSE'
        );

        console.error(
          'Error:',
          kimiError.response?.data ||
          kimiError.message
        );

        console.error(
          'Switching to DeepSeek V4 Pro...'
        );

        console.error(
          '========================================'
        );


        // ======================================
        // FALLBACK: DEEPSEEK V4 PRO
        // ======================================

        activeModel =
          FALLBACK_MODEL;


        // DeepSeek V4 Pro currently documents
        // 16384 as its max max_tokens value.
        const fallbackTokens =
          Math.min(
            outputTokens,
            FALLBACK_MAX_OUTPUT_TOKENS
          );


        try {

          response =
            await sendToNvidia(

              FALLBACK_MODEL,

              messages,

              temperature,

              fallbackTokens,

              stream

            );


          console.log(
            `NVIDIA FALLBACK SUCCESS: ${FALLBACK_MODEL}`
          );


        } catch (fallbackError) {

          console.error(
            '========================================'
          );

          console.error(
            'DEEPSEEK FALLBACK ALSO FAILED'
          );

          console.error(
            'Status:',
            fallbackError.response?.status ||
            'NO RESPONSE'
          );

          console.error(
            'Error:',
            fallbackError.response?.data ||
            fallbackError.message
          );

          console.error(
            '========================================'
          );


          throw fallbackError;

        }

      }


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

        res.setHeader(
          'X-Accel-Buffering',
          'no'
        );

        res.flushHeaders();


        let buffer = '';

        let thinkStarted = false;

        let thinkEnded = false;


        response.data.on(
          'data',
          chunk => {

            buffer +=
              chunk.toString();


            const lines =
              buffer.split('\n');


            // Keep incomplete line
            buffer =
              lines.pop() || '';


            for (
              const line of lines
            ) {

              if (
                !line.startsWith(
                  'data: '
                )
              ) {

                continue;

              }


              // --------------------------------
              // DONE
              // --------------------------------

              if (
                line.trim() ===
                'data: [DONE]'
              ) {

                if (
                  thinkStarted &&
                  !thinkEnded
                ) {

                  res.write(

                    `data: ${JSON.stringify({

                      choices: [

                        {

                          delta: {

                            content:
                              '</think>\n'

                          }

                        }

                      ]

                    })}\n\n`

                  );

                }


                res.write(
                  'data: [DONE]\n\n'
                );

                continue;

              }


              // --------------------------------
              // PARSE JSON
              // --------------------------------

              try {

                const data =
                  JSON.parse(
                    line.slice(6)
                  );


                const delta =
                  data.choices?.[0]?.delta;


                if (delta) {


                  // ============================
                  // REASONING
                  // ============================

                  const reasoning =
                    delta.reasoning_content ||
                    delta.reasoning ||
                    delta.thinking_content ||
                    '';


                  if (reasoning) {


                    // First reasoning chunk
                    if (!thinkStarted) {

                      thinkStarted =
                        true;


                      delta.content =
                        '<think>\n' +
                        reasoning;

                    }


                    // Subsequent reasoning
                    else {

                      delta.content =
                        reasoning;

                    }

                  }


                  // ============================
                  // NORMAL RESPONSE
                  // ============================

                  else if (

                    delta.content &&

                    thinkStarted &&

                    !thinkEnded

                  ) {

                    delta.content =
                      '</think>\n' +
                      delta.content;

                    thinkEnded =
                      true;

                  }


                  // ============================
                  // REMOVE REASONING FIELDS
                  // ============================

                  delete delta.reasoning;

                  delete delta.reasoning_content;

                  delete delta.thinking_content;

                }


                // --------------------------------
                // Send modified SSE
                // --------------------------------

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

          }
        );


        // --------------------------------------
        // STREAM END
        // --------------------------------------

        response.data.on(
          'end',
          () => {

            res.end();

          }
        );


        // --------------------------------------
        // STREAM ERROR
        // --------------------------------------

        response.data.on(
          'error',
          err => {

            console.error(
              `NVIDIA ${activeModel} stream error:`,
              err.message
            );

            res.end();

          }
        );


        return;

      }


      // ========================================
      // NON-STREAMING
      // ========================================

      const data =
        response.data;


      res.json({

        id:
          data.id ||
          `chatcmpl-${Date.now()}`,

        object:
          'chat.completion',

        created:
          data.created ||
          Math.floor(
            Date.now() / 1000
          ),

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
      // FINAL ERROR
      // ========================================

      const status =
        error.response?.status ||
        500;


      const upstream =
        error.response?.data;


      console.error(
        'FINAL NVIDIA ERROR:',
        upstream ||
        error.message
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

  }
);


// ========================================
// 404
// ========================================

app.use(
  (req, res) => {

    res.status(404).json({

      error: {

        message:
          `Endpoint ${req.path} not found`,

        type:
          'invalid_request_error',

        code: 404

      }

    });

  }
);


// ========================================
// START SERVER
// ========================================

app.listen(
  PORT,
  () => {

    console.log(
      '========================================'
    );

    console.log(
      'NVIDIA NIM RP PROXY STARTED'
    );

    console.log(
      `Port: ${PORT}`
    );

    console.log(
      `Primary: ${MODEL}`
    );

    console.log(
      `Fallback: ${FALLBACK_MODEL}`
    );

    console.log(
      `JAI Model: ${JAI_MODEL_NAME}`
    );

    console.log(
      `Kimi Max Tokens: ${MAX_OUTPUT_TOKENS}`
    );

    console.log(
      `DeepSeek Fallback Max Tokens: ${FALLBACK_MAX_OUTPUT_TOKENS}`
    );

    console.log(
      '========================================'
    );

  }
);
