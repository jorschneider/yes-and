// api/improv.mjs — Vercel Node serverless proxy for "Yes, And" live mode.
//
// Holds the Anthropic API key server-side so it never ships to the browser.
// Two modes:
//   mode:'actor'    -> plays the AI ACTOR opposite the human PLAYER (latency-sensitive => Sonnet)
//   mode:'director' -> scores the scene like a grease-pencil prompt-book (quality => Opus)
//
// Request:  POST JSON { mode:'actor'|'director', transcript:[{name,text}], persona? }
// Response (actor):    { line, stageDirection }
// Response (director): { status, platform:{...}, tilt:{...}, game:{...} }

import Anthropic from "@anthropic-ai/sdk";

// Latency-sensitive partner play vs. quality judging — per spec.
const ACTOR_MODEL = "claude-sonnet-4-6";
const DIRECTOR_MODEL = "claude-opus-4-8";

// ===== THEORY ENGINE (actor system prompt) — large + stable => prompt-cached =====
const ACTOR_SYSTEM = `You are the AI ACTOR in a two-person improv scene. The human is the PLAYER. You speak exactly ONE character line at a time, in response to the PLAYER's most recent line, continuing the same scene.

You perform from the craft canon of Keith Johnstone (Impro) and the Upright Citizens Brigade. These constraints are non-negotiable:

ACCEPT THE OFFER ("Yes, and"). Treat whatever the PLAYER just established — names, place, relationship, the emotional fact, any object — as true. Never negate, never block, never argue the premise away. Build on it.

UNDER-WRITE: ONE NEW GIFT PER LINE. Add exactly one new piece of information — a detail, a name, a feeling, a complication. Do not flood the scene with five new facts; that buries your partner. One clean gift, generously placed.

NO PLATFORM-STALLING QUESTIONS. Do not hand the work back with open questions like "What is this place?" / "Who are you?" / "What should we do?" — those make the PLAYER do your job. If you would ask a question, make a STATEMENT instead: answer with an offer. (A small rhetorical or in-character question that still gives a gift is fine; an interrogation that stalls the platform is not.)

ESTABLISH, THEN HEIGHTEN THE GAME. Early on, help set the base reality — WHO the two of you are, WHAT you're doing, WHERE you are. Once the first unusual thing (the "tilt") appears, treat it as the engine of the scene. Find the GAME — the repeatable comedic pattern that one unusual thing implies — and HEIGHTEN it: raise the SAME unusual thing a floor higher each time. Do not stack a brand-new absurdity on top; explore and escalate the one you have. Play to the top of your intelligence.

MAKE YOUR PARTNER LOOK GOOD. Your job is to make the PLAYER's choices look smart and inevitable. React honestly first, then build. Justify their moves; never make them wrong.

VOICE & LENGTH. Stay in character. One to two sentences, conversational, performable out loud. No meta-commentary, no narration of what you're "doing," no quotation marks around the whole line, no speaker label.

OUTPUT FORMAT. Return ONLY a compact JSON object, no prose around it, no markdown fence:
{"line": "<your spoken character line>", "stageDirection": "<a SHORT optional parenthetical action like 'sliding the card in' or 'grinning' — empty string if none>"}
Do not put the stage direction inside "line". Keep stageDirection a few words at most.`;

// ===== DIRECTOR RUBRIC (director system prompt) — large + stable => prompt-cached =====
const DIRECTOR_SYSTEM = `You are the DIRECTOR watching an improv scene unfold, scoring it turn by turn like a grease-pencil prompt-book. You judge against the same canon the actors play from: Keith Johnstone (status, platform, the tilt, "make your partner look good") and UCB (base reality, the first unusual thing, the game of the scene).

Given the full transcript so far, assess FOUR things about the scene AS IT STANDS RIGHT NOW:

1. STATUS — who currently has the high status / is "up" in the scene. Answer 'PLAYER' or 'ACTOR'. If genuinely even, pick whoever was last up.

2. PLATFORM — is the base reality established yet? The platform is set once WHO (the two people / relationship), WHAT (the activity), and WHERE (the location) are all reasonably clear. Report established:true/false and baseReality: a short phrase naming the who/what/where as established so far (e.g. "two co-workers clocking in at a flood-damaged workplace"). If not yet set, baseReality is your best read of what's been offered.

3. TILT — has the first unusual thing landed? The tilt is the first clear break from ordinary routine — the thing that is surprising and not-yet-explained. Report landed:true/false and line: the verbatim (or near-verbatim) transcript line where the tilt lands, or "" if it hasn't yet.

4. GAME — has the game of the scene been named/found? The game is the repeatable comedic pattern the tilt implies — the engine that every later beat can pay forward and heighten. Report stated:true/false and oneLine: a single crisp sentence naming the pattern (e.g. "the time-clock pays out emotional labor by the hour"), or "" if it hasn't emerged.

Be honest and a little strict, like a good director. Early in a scene most of these will be false — that's correct; don't flatter the scene.

You MUST call the record_beats tool with your assessment. Do not write any prose.`;

// JSON-schema tool the director must fill — strict structured output.
const DIRECTOR_TOOL = {
  name: "record_beats",
  description:
    "Record the director's turn-by-turn read of the scene: status, platform, tilt, and game.",
  input_schema: {
    type: "object",
    properties: {
      status: {
        type: "string",
        enum: ["PLAYER", "ACTOR"],
        description: "Who currently has the high status / is up in the scene.",
      },
      platform: {
        type: "object",
        properties: {
          established: {
            type: "boolean",
            description: "True once who + what + where are all reasonably clear.",
          },
          baseReality: {
            type: "string",
            description: "Short phrase naming the who/what/where established so far.",
          },
        },
        required: ["established", "baseReality"],
        additionalProperties: false,
      },
      tilt: {
        type: "object",
        properties: {
          landed: {
            type: "boolean",
            description: "True once the first unusual thing has clearly appeared.",
          },
          line: {
            type: "string",
            description:
              "The verbatim transcript line where the tilt lands, or empty string if not yet.",
          },
        },
        required: ["landed", "line"],
        additionalProperties: false,
      },
      game: {
        type: "object",
        properties: {
          stated: {
            type: "boolean",
            description: "True once the repeatable comedic pattern has been found.",
          },
          oneLine: {
            type: "string",
            description:
              "Single sentence naming the game's pattern, or empty string if not yet.",
          },
        },
        required: ["stated", "oneLine"],
        additionalProperties: false,
      },
    },
    required: ["status", "platform", "tilt", "game"],
    additionalProperties: false,
  },
};

// ---- helpers --------------------------------------------------------------

function sendJson(res, code, payload) {
  res.statusCode = code;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify(payload));
}

// Body may already be parsed by the platform, or arrive as a raw stream.
async function readBody(req) {
  if (req.body && typeof req.body === "object") return req.body;
  if (typeof req.body === "string") {
    try { return JSON.parse(req.body); } catch { return {}; }
  }
  const chunks = [];
  for await (const chunk of req) chunks.push(chunk);
  if (!chunks.length) return {};
  try { return JSON.parse(Buffer.concat(chunks).toString("utf8")); } catch { return {}; }
}

// Render the transcript as alternating turns the model can read.
function transcriptToMessages(transcript) {
  return transcript
    .filter((t) => t && typeof t.text === "string" && t.text.trim())
    .map((t) => ({
      role: t.name === "AI ACTOR" ? "assistant" : "user",
      content: `${t.name}: ${t.text.trim()}`,
    }));
}

function firstTextBlock(message) {
  const block = (message.content || []).find((b) => b.type === "text");
  return block ? block.text : "";
}

function parseLooseJson(text) {
  if (!text) return null;
  try { return JSON.parse(text); } catch { /* fall through */ }
  const m = text.match(/\{[\s\S]*\}/);
  if (m) { try { return JSON.parse(m[0]); } catch { /* ignore */ } }
  return null;
}

// ---- handler --------------------------------------------------------------

export default async function handler(req, res) {
  if (req.method !== "POST") {
    return sendJson(res, 405, { error: "Use POST." });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return sendJson(res, 500, {
      error:
        "ANTHROPIC_API_KEY is not set on the server. Live mode needs the key configured in the deployment's environment.",
    });
  }

  let body;
  try {
    body = await readBody(req);
  } catch {
    return sendJson(res, 400, { error: "Could not read request body." });
  }

  const mode = body && body.mode;
  const transcript = Array.isArray(body && body.transcript) ? body.transcript : [];
  if (mode !== "actor" && mode !== "director") {
    return sendJson(res, 400, { error: "mode must be 'actor' or 'director'." });
  }
  if (!transcript.length) {
    return sendJson(res, 400, { error: "transcript is required and must be non-empty." });
  }

  const client = new Anthropic({ apiKey });
  const messages = transcriptToMessages(transcript);

  try {
    if (mode === "actor") {
      const personaNote =
        body.persona && typeof body.persona === "string"
          ? `\n\nSCENE PERSONA / STARTING NOTE FOR THE ACTOR: ${body.persona.trim()}`
          : "";
      const message = await client.messages.create({
        model: ACTOR_MODEL,
        max_tokens: 400,
        system: [
          {
            type: "text",
            text: ACTOR_SYSTEM + personaNote,
            cache_control: { type: "ephemeral" },
          },
        ],
        messages,
      });

      const parsed = parseLooseJson(firstTextBlock(message));
      const line =
        parsed && typeof parsed.line === "string" && parsed.line.trim()
          ? parsed.line.trim()
          : firstTextBlock(message).trim();
      const stageDirection =
        parsed && typeof parsed.stageDirection === "string"
          ? parsed.stageDirection.trim()
          : "";
      if (!line) {
        return sendJson(res, 502, { error: "The actor model returned an empty line." });
      }
      return sendJson(res, 200, { line, stageDirection });
    }

    // mode === "director": force the structured tool call.
    const message = await client.messages.create({
      model: DIRECTOR_MODEL,
      max_tokens: 1024,
      thinking: { type: "adaptive" },
      output_config: { effort: "high" },
      system: [
        {
          type: "text",
          text: DIRECTOR_SYSTEM,
          cache_control: { type: "ephemeral" },
        },
      ],
      tools: [DIRECTOR_TOOL],
      tool_choice: { type: "tool", name: "record_beats" },
      messages,
    });

    const toolUse = (message.content || []).find((b) => b.type === "tool_use");
    const beats = toolUse && toolUse.input ? toolUse.input : null;
    if (!beats || typeof beats !== "object") {
      return sendJson(res, 502, { error: "The director model did not return structured beats." });
    }

    // Normalize defensively so the client always gets the documented shape.
    const out = {
      status: beats.status === "ACTOR" ? "ACTOR" : "PLAYER",
      platform: {
        established: !!(beats.platform && beats.platform.established),
        baseReality:
          (beats.platform && typeof beats.platform.baseReality === "string"
            ? beats.platform.baseReality
            : "") || "",
      },
      tilt: {
        landed: !!(beats.tilt && beats.tilt.landed),
        line:
          (beats.tilt && typeof beats.tilt.line === "string" ? beats.tilt.line : "") || "",
      },
      game: {
        stated: !!(beats.game && beats.game.stated),
        oneLine:
          (beats.game && typeof beats.game.oneLine === "string" ? beats.game.oneLine : "") || "",
      },
    };
    return sendJson(res, 200, out);
  } catch (err) {
    const status = err && typeof err.status === "number" ? err.status : 500;
    const detail = err && err.message ? err.message : "Unknown error calling the Anthropic API.";
    return sendJson(res, status >= 400 && status < 600 ? status : 500, {
      error: `Anthropic API call failed: ${detail}`,
    });
  }
}
