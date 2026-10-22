import { json } from '@sveltejs/kit';
import { cosmosContainer } from '$lib/cosmos';

export const GET = async ({ params }) => {
  const { resources: items } = await cosmosContainer.items
    .query({
      query: `SELECT c.id, c.imageURL, c.caption, c.caption_en, c.volume, c.page, c.location, c.imageVector, c.captionVector FROM c WHERE c.id = @itemID`,
      parameters: [{ name: '@itemID', value: params.id }]
    })
    .fetchAll();
  if (items.length === 0) {
    return json({ error: 'Item not found', status: 404 });
  }
  const item = items[0];

  const { resources: similarImages } = await cosmosContainer.items
    .query({
      query: `SELECT c.id, c.imageURL, c.caption_en, c.volume, c.page, c.location FROM c ORDER BY VectorDistance(c.imageVector, @embedding) OFFSET 1 LIMIT 8`,
      parameters: [{ name: '@embedding', value: item.imageVector }]
    })
    .fetchAll();

  const { resources: similarCaptions } = await cosmosContainer.items
    .query({
      query: `SELECT c.id, c.imageURL, c.caption_en, c.volume, c.page, c.location FROM c ORDER BY VectorDistance(c.captionVector, @embedding) OFFSET 1 LIMIT 8`,
      parameters: [{ name: '@embedding', value: item.captionVector }]
    })
    .fetchAll();

  return json({ item, similarImages, similarCaptions });
};
