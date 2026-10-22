<script lang="ts">
  import { SlideToggle } from '@skeletonlabs/skeleton';
  import Map from '$lib/components/Map.svelte';
  import type { PPMGeoJSONFeatureCollection } from '$lib/types';

  type SearchItem = {
    id: string;
    imageURL: string;
    caption_en: string;
    volume: number;
    page: number;
    rank: number;
    image_similarity: number;
    caption_similarity: number;
    included_by: string;
  };

  let items: SearchItem[] = [];
  let query = '';
  let imageMinimumSimilarity = 0.22;
  let captionMinimumSimilarity = 0.7;
  let errorMessage = '';
  let showRegios = true;
  let itemsLoading = false;
  let hasSearched = false;
  const geojson: PPMGeoJSONFeatureCollection = { type: 'FeatureCollection', features: [] };

  const search = async () => {
    if (!query.trim()) return;
    itemsLoading = true;
    errorMessage = '';
    hasSearched = true;

    try {
      const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, imageMinimumSimilarity, captionMinimumSimilarity })
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Search failed.');
      items = data.items;
    } catch (error) {
      console.error(error);
      errorMessage = error instanceof Error ? error.message : 'Search failed. Please try again.';
      items = [];
    } finally {
      itemsLoading = false;
    }
  };
</script>

<main class="container mx-auto p-6">
  <div class="grid min-h-[calc(100vh-8rem)] grid-cols-1 gap-6 lg:grid-cols-5">
    <section class="flex min-h-0 flex-col border-r border-primary-500 p-5 lg:col-span-2">
      <h1 class="mb-4 text-2xl font-bold">Search the sample collection</h1>
      <p class="mb-5 text-sm opacity-80">
        Search 350 linked Pompeii image and caption pairs with OpenCLIP ViT-B/32. A pair appears
        when its image or caption meets its similarity threshold.
      </p>

      <label class="mb-2 block">
        <span>Text query</span>
        <input
          class="input mt-1 block w-full"
          type="text"
          bind:value={query}
          placeholder="Describe an image or scene"
          on:keydown={(event) => event.key === 'Enter' && search()}
        />
      </label>
      <div class="mb-4 grid grid-cols-2 gap-4">
        <label class="block text-sm">
          <span>Image minimum similarity: {imageMinimumSimilarity.toFixed(2)}</span>
          <input class="mt-2 block w-full" type="range" min="0" max="1" step="0.01" bind:value={imageMinimumSimilarity} />
        </label>
        <label class="block text-sm">
          <span>Caption minimum similarity: {captionMinimumSimilarity.toFixed(2)}</span>
          <input class="mt-2 block w-full" type="range" min="0" max="1" step="0.01" bind:value={captionMinimumSimilarity} />
        </label>
      </div>
      <button
        type="button"
        class="variant-filled-secondary btn mb-5 w-full text-white"
        disabled={itemsLoading || !query.trim()}
        on:click={search}
      >
        {itemsLoading ? 'Searching…' : 'Search'}
      </button>

      {#if errorMessage}
        <p class="mb-4 text-red-500">{errorMessage}</p>
      {/if}
      <div class="min-h-0 flex-1 space-y-5 overflow-y-auto">
        {#if !itemsLoading && hasSearched && !errorMessage && items.length === 0}
          <p>No image or caption pairs met those thresholds.</p>
        {/if}
        {#each items as item}
          <article class="card variant-ghost-tertiary mx-auto w-[95%] p-4">
            <header class="flex justify-center">
              <img class="max-h-72 max-w-full object-contain" src={item.imageURL} alt={item.id} />
            </header>
            <div class="space-y-3 p-3">
              <h2 class="font-bold">
                Volume {item.volume}, Page {item.page} (Image {item.id.split('_')[5]})
              </h2>
              <p>{item.caption_en}</p>
              <p class="text-sm opacity-80">
                Image similarity {item.image_similarity.toFixed(3)} · Caption similarity
                {item.caption_similarity.toFixed(3)} · Matched by {item.included_by}
              </p>
            </div>
          </article>
        {/each}
      </div>
    </section>

    <section class="flex flex-col p-5 lg:col-span-3">
      <SlideToggle
        name="show-regios"
        bind:checked={showRegios}
        active="bg-secondary-500"
        size="sm"
      >
        Show Regios
      </SlideToggle>
      <div class="mt-4 min-h-80 flex-1">
        <Map {geojson} center={[14.4884, 40.75103]} zoom={14.5} {showRegios} />
      </div>
      <p class="mt-3 text-sm opacity-75">
        The sample metadata has no location geometry, so results are not highlighted on the map yet.
      </p>
    </section>
  </div>
</main>
