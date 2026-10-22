import { json } from '@sveltejs/kit';
import { openaiClient } from '$lib/openai';

const generateLLMResponse = async (query: string, items): Promise<string> => {
  try {
    const systemPrompt = `
      You are an intelligent AI assistant specialized in answering questions about ancient Pompeii.
      Your purpose is to provide helpful, insightful responses based on the user's queries, supported by a collection of documents that include images, captions, volume numbers, page numbers, and specific locations.
      Your goal is to uncover new discoveries and offer new insights into Pompeii, highlighting lesser-known aspects.
      For location-based queries, provide detailed information that includes the regio, insula, property, and room numbers.
      When listing items or explaining results, include the item ID, regio, insula, property, and room number in your response (bulleted list format of ID: regio, insula, property, room).
      Limit the number of items you list to 3 or fewer to avoid overwhelming the user (unless the user requests more).
      Do not answer questions that do not pertain to ancient Pompeii.
      Use line breaks to separate sentences when starting a new thought.
    `;
    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: query },
      ...items.map((item) => ({ role: 'system', content: JSON.stringify(item) }))
    ];

    const response = await openaiClient.chat.completions.create({
      model: 'gpt-35-turbo',
      messages
    });
    return response.choices[0].message.content;
  } catch (err) {
    console.error('Error:', err);
    return '';
  }
};

export const POST = async ({ request }) => {
  try {
    const { query, items } = await request.json();
    if (!query || !items) {
      throw new Error('Missing required parameters');
    }

    const llmResponse = await generateLLMResponse(
      query,
      items.map((item) => {
        return {
          id: item.id,
          imageURL: item.imageURL,
          caption_en: item.caption_en,
          volume: item.volume,
          page: item.page,
          regio: item.location.regio,
          insula: item.location.insula,
          property: item.location.property,
          room: item.location.room
        };
      })
    );

    return json({ llmResponse });
  } catch (err) {
    console.error('Error:', err.message);
    return json({ error: err.message }, { status: 500 });
  }
};
