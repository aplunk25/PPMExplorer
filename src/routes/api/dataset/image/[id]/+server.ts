import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { error } from '@sveltejs/kit';

export const GET = async ({ params }) => {
  const id = params.id;
  if (!/^Volume_\d+_Page_\d+_image_\d+$/.test(id)) {
    throw error(400, 'Invalid image ID.');
  }

  try {
    const image = await readFile(path.join(process.cwd(), '350Package', 'data', 'images', `${id}.jpg`));
    return new Response(image, {
      headers: {
        'Content-Type': 'image/jpeg',
        'Cache-Control': 'public, max-age=3600'
      }
    });
  } catch {
    throw error(404, 'Image not found.');
  }
};
