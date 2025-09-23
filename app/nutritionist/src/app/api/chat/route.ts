import { NextRequest } from "next/server";
import OpenAI from "openai";
import { createClient } from "@supabase/supabase-js";

export const runtime = "nodejs";          // ensure Node runtime (service role key, Node SDKs)
export const dynamic = "force-dynamic";   // no caching of answers

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY! });

// Service role key is server-only; never import this file on the client.
const supabase = createClient(
  process.env.SUPABASE_URL!,
  process.env.SUPABASE_SERVICE_ROLE_KEY!,
  { auth: { persistSession: false, autoRefreshToken: false } }
);

async function embedQuery(query: string) {
  const resp = await openai.embeddings.create({
    model: "text-embedding-3-small",   // 1536-dim; matches your table
    input: query
  });
  return resp.data[0].embedding;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const message = (body?.message ?? "").toString().trim();

    if (!message) {
      return new Response(JSON.stringify({ error: "Empty query" }), {
        status: 400,
        headers: { "content-type": "application/json" }
      });
    }

    // 1) Embed the query
    const queryEmb = await embedQuery(message);

    // 2) Retrieve from Supabase (constrain to this PDF)
    const { data: chunks, error } = await supabase.rpc("match_documents", {
      query_embedding: queryEmb,
      match_count: 8,
      filter: { source: "data/datasets/rag/human-nutrition-text.pdf" },
    });

    if (error) throw error;

    // Optional: log retrieval for debugging in server logs
    // console.log("retrieved", (chunks ?? []).map((c: any) => ({
    //   p: c.metadata?.page, sim: Number(c.similarity).toFixed(3),
    //   prev: c.content.slice(0, 100)
    // })));

    // 3) Build the context (show page numbers)
    const context = (chunks ?? [])
      .map((c: any, i: number) => `[${i + 1}] (Page ${c.metadata?.page ?? "?"}) ${c.content}`)
      .join("\n\n");

    // If nothing relevant was found, short-circuit with a helpful reply
    if (!context) {
      return new Response(JSON.stringify({
        answer:
          "I couldn’t find this in the provided document. Try rephrasing or asking about a different section.",
        sources: []
      }), { status: 200, headers: { "content-type": "application/json" }});
    }

    // 4) Ask the model with strict instructions
    const completion = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      temperature: 0.2,
      messages: [
        {
          role: "system",
          content:
            "You are a strict RAG assistant. Answer ONLY using the CONTEXT. " +
            "If the answer is not present, say: 'I couldn’t find this in the provided document.' " +
            "Cite sources like [1], [2] and include page numbers (e.g., p. X) next to each claim."
        },
        { role: "user", content: `QUESTION: ${message}\n\nCONTEXT:\n${context}` }
      ]
    });

    return new Response(JSON.stringify({
      answer: completion.choices[0]?.message?.content ?? "",
      sources: chunks ?? []
    }), { status: 200, headers: { "content-type": "application/json" }});

  } catch (err: any) {
    console.error("api/chat error:", err?.message || err);
    return new Response(JSON.stringify({ error: err?.message || "Unknown error" }), {
      status: 500,
      headers: { "content-type": "application/json" }
    });
  }
}
