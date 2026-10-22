import { json } from '@sveltejs/kit';
import { spawn } from 'node:child_process';
import { readFileSync } from 'node:fs';
import path from 'node:path';

type SampleLocation = { name: string; scope: string; lat: number; lon: number };

// Written by 350Package/export_locations.py
const locations: Record<string, SampleLocation> = JSON.parse(
  readFileSync(
    path.join(process.cwd(), '350Package', 'embeddings', 'random_samples', 'locations.json'),
    'utf8'
  )
);

type Candidate = {
  rank: number;
  image_id: string;
  image_path: string;
  associated_caption: string;
  image_similarity: number;
  caption_similarity: number;
  image_in_radius: boolean;
  caption_in_radius: boolean;
  included_by: string;
};

const runSearch = (query: string, imageRadius: number, captionRadius: number) =>
  new Promise<Candidate[]>((resolve, reject) => {
    const root = process.cwd();
    const python =
      process.env.PYTHON_EXECUTABLE || (process.platform === 'win32' ? 'python' : 'python3');
    const child = spawn(
      python,
      [
        path.join(root, '350Package', 'search_candidates.py'),
        query,
        '--image-radius',
        String(imageRadius),
        '--caption-radius',
        String(captionRadius),
        '--json'
      ],
      { cwd: path.join(root, '350Package'), windowsHide: true }
    );
    let stdout = '';
    let stderr = '';
    child.stdout.setEncoding('utf8').on('data', (chunk: string) => (stdout += chunk));
    child.stderr.setEncoding('utf8').on('data', (chunk: string) => (stderr += chunk));
    child.on('error', reject);
    child.on('close', (code) => {
      if (code !== 0) {
        reject(new Error(stderr || `CLIP search exited with code ${code}`));
        return;
      }
      const resultLine = stdout.split(/\r?\n/).find((line) => line.startsWith('JSON_RESULT='));
      if (!resultLine) {
        reject(new Error('CLIP search did not return results.'));
        return;
      }
      try {
        resolve(JSON.parse(resultLine.slice('JSON_RESULT='.length)) as Candidate[]);
      } catch (error) {
        reject(error);
      }
    });
  });

export const POST = async ({ request }) => {
  try {
    const {
      query,
      imageMinimumSimilarity = 0.22,
      captionMinimumSimilarity = 0.7
    } = await request.json();
    if (typeof query !== 'string' || !query.trim()) {
      return json({ error: 'Enter a search query.' }, { status: 400 });
    }
    if (
      typeof imageMinimumSimilarity !== 'number' ||
      imageMinimumSimilarity < -1 ||
      imageMinimumSimilarity > 1 ||
      typeof captionMinimumSimilarity !== 'number' ||
      captionMinimumSimilarity < -1 ||
      captionMinimumSimilarity > 1
    ) {
      return json({ error: 'Similarity thresholds must be between -1 and 1.' }, { status: 400 });
    }

    const candidates = await runSearch(
      query.trim(),
      1 - imageMinimumSimilarity,
      1 - captionMinimumSimilarity
    );
    return json({
      items: candidates.map((candidate) => ({
        ...candidate,
        id: candidate.image_id,
        imageURL: `/api/dataset/image/${encodeURIComponent(candidate.image_id)}`,
        caption_en: candidate.associated_caption,
        volume: Number(candidate.image_id.match(/Volume_(\d+)/)?.[1] ?? 0),
        page: Number(candidate.image_id.match(/Page_(\d+)/)?.[1] ?? 0),
        location: locations[candidate.image_id] ?? null
      }))
    });
  } catch (error) {
    console.error('CLIP search failed:', error);
    return json(
      { error: 'CLIP search failed. Check the Python and OpenCLIP setup, then try again.' },
      { status: 500 }
    );
  }
};
